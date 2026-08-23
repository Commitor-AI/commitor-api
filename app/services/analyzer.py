import logging
import time
from collections import OrderedDict
from typing import Any

from fastapi import HTTPException, status

from app.config import Settings, get_settings
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse, ChangeGroup, ModelTier
from app.services.openrouter import (
    ChatResult,
    OpenRouterError,
    assistant_message,
    get_openrouter_client,
)

logger = logging.getLogger(__name__)

_JSON_INSTRUCTION = (
    "Respond with a single JSON object and nothing else:\n"
    '{"groups": [{"files": ["path"], "partial_files": [{"path": "path", "hunks": [1]}], '
    '"commit_message": "conventional commit message", '
    '"rationale": "why these changes belong together"}], "confidence": 0.0-1.0}\n'
    "Rules for splitting inside a single file:\n"
    "- Hunks are numbered 1-based per file, in the order they appear in the diff.\n"
    "- To assign only part of a file to a group, list the file in that group's "
    "partial_files with the hunk numbers; do not also list it in files.\n"
    "- Binary, deleted, new, and renamed files must always be assigned whole "
    "(never in partial_files).\n"
    "- Every hunk of every changed file must be assigned exactly once across all "
    "groups — no gaps, no duplicates.\n"
    "- If the whole diff is one logical change, return a single group with all "
    "files in `files` and leave `partial_files` empty."
)

_FAST_SYSTEM_PROMPT = (
    "You analyze git diffs and decide whether the changes form one logical commit or "
    "should be split into multiple commits. Group only changes that are clearly related. "
    "You may split a single file's hunks across groups when parts of it belong to "
    "different logical changes.\n" + _JSON_INSTRUCTION
)

_REASONING_SYSTEM_PROMPT = (
    "You are a meticulous reviewer deciding how a git diff should be split into commits. "
    "Think carefully through each hunk: what code path it belongs to, which other hunks it depends on, "
    "and whether the changes could land independently without breaking the build. "
    "Only group changes when there is a concrete dependency or shared purpose; when in doubt, split. "
    "Weigh file paths, import changes, shared types, and call sites as evidence of coupling. "
    "When two hunks of the SAME file belong to different logical changes, assign them to "
    "different groups via partial_files instead of lumping them together.\n" + _JSON_INSTRUCTION
)

_RECHECK_FOLLOW_UP = (
    "Re-examine your proposed grouping. For each group you kept together, confirm the changes "
    "could not be committed independently. If any group mixes unrelated concerns, revise the plan. "
    "Respond with the same JSON format."
)


def _extract_files(diff: str) -> list[str]:
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :]
            if path not in files:
                files.append(path)
    return files


def _diff_line_count(diff: str) -> int:
    return sum(1 for line in diff.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))


def _group_by_top_level(files: list[str]) -> list[ChangeGroup]:
    buckets: OrderedDict[str, list[str]] = OrderedDict()
    for path in files:
        top = path.split("/", 1)[0] if "/" in path else "(root)"
        buckets.setdefault(top, []).append(path)
    return [
        ChangeGroup(
            files=paths,
            commit_message=f"chore: update {top}",
            rationale=f"All changed files live under '{top}'",
        )
        for top, paths in buckets.items()
    ]


def _is_large(diff: str, files: list[str], settings: Settings) -> bool:
    return len(files) > settings.analyze_escalation_files or _diff_line_count(diff) > settings.analyze_escalation_diff_lines


def _heuristic_pass(payload: AnalyzeRequest, settings: Settings) -> tuple[AnalyzeResponse | None, list[str]]:
    files = _extract_files(payload.diff)
    if not files or _is_large(payload.diff, files, settings):
        return None, files
    groups = _group_by_top_level(files)
    if len(groups) == 1:
        response = AnalyzeResponse(groups=groups, confidence=0.8, model_tier=ModelTier.local)
        return response, files
    return None, files


def _parse_groups(result: ChatResult) -> tuple[list[ChangeGroup], float]:
    raw_groups = result.content.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("missing 'groups' list in model output")
    groups = [ChangeGroup.model_validate(item) for item in raw_groups]
    confidence = float(result.content.get("confidence", 0.0))
    return groups, max(0.0, min(1.0, confidence))


async def analyze_diff(payload: AnalyzeRequest, settings: Settings | None = None) -> AnalyzeResponse:
    settings = settings or get_settings()
    started = time.monotonic()

    local_response, files = _heuristic_pass(payload, settings)
    if local_response is not None:
        logger.info(
            "analyze tier=local model=none files=%d diff_lines=%d latency=%.2fs total_latency=%.2fs",
            len(files),
            _diff_line_count(payload.diff),
            0.0,
            time.monotonic() - started,
        )
        return local_response

    client = get_openrouter_client()
    user_prompt = payload.diff if payload.context is None else f"Context: {payload.context}\n\nDiff:\n{payload.diff}"

    fast_result: ChatResult | None = None
    try:
        fast_result = await client.chat_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": _FAST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        groups, confidence = _parse_groups(fast_result)
        if confidence >= settings.analyze_confidence_threshold:
            logger.info(
                "analyze tier=fast model=%s files=%d confidence=%.2f latency=%.2fs prompt_tokens=%s completion_tokens=%s total_latency=%.2fs",
                fast_result.model,
                len(files),
                confidence,
                fast_result.latency_seconds,
                fast_result.usage.get("prompt_tokens"),
                fast_result.usage.get("completion_tokens"),
                time.monotonic() - started,
            )
            return AnalyzeResponse(groups=groups, confidence=confidence, model_tier=ModelTier.fast)
    except (OpenRouterError, ValueError) as exc:
        logger.warning("fast model pass failed (%s), escalating to reasoning model", exc)

    try:
        reasoning_result = await client.chat_completion_with_reasoning(
            system_prompt=_REASONING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        groups, confidence = _parse_groups(reasoning_result)
        history = [assistant_message(reasoning_result.raw_content, reasoning_result.reasoning_details)]

        if confidence < 0.9:
            recheck = await client.chat_completion_with_reasoning(
                system_prompt=_REASONING_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                history=history,
                follow_up=_RECHECK_FOLLOW_UP,
            )
            try:
                revised_groups, revised_confidence = _parse_groups(recheck)
                groups, confidence = revised_groups, revised_confidence
                history.append(assistant_message(recheck.raw_content, recheck.reasoning_details))
                reasoning_result = recheck
            except ValueError as exc:
                logger.warning("recheck turn unparseable (%s), keeping first reasoning verdict", exc)

        logger.info(
            "analyze tier=reasoning model=%s files=%d confidence=%.2f latency=%.2fs reasoning_details=%d turns=%d prompt_tokens=%s completion_tokens=%s total_latency=%.2fs",
            reasoning_result.model,
            len(files),
            confidence,
            reasoning_result.latency_seconds,
            len(reasoning_result.reasoning_details),
            len(history),
            reasoning_result.usage.get("prompt_tokens"),
            reasoning_result.usage.get("completion_tokens"),
            time.monotonic() - started,
        )
        return AnalyzeResponse(groups=groups, confidence=confidence, model_tier=ModelTier.reasoning)
    except (OpenRouterError, ValueError) as exc:
        logger.error("reasoning model escalation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Diff analysis is temporarily unavailable, please retry",
        ) from exc
