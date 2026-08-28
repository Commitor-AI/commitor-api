import logging
import re
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
    '"commit_message": "commit message", '
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
    "files in `files` and leave `partial_files` empty.\n"
    "\n"
    "Rules for commit_message:\n"
    "- Every message MUST describe what actually changed in THIS diff: the action "
    "(add / remove / fix / refactor / configure …), the concrete target "
    "(module, component, function, endpoint, config key), and the visible effect "
    "or purpose when the diff makes it clear.\n"
    "- Ground it in real names from the diff: file paths, types, functions, "
    "routes, flags. A reader who hasn't seen the diff should know what moved.\n"
    "- Use imperative mood with a fitting conventional-commit prefix "
    "(feat/fix/refactor/docs/test/chore/style/perf).\n"
    "- Keep the subject line under 72 characters but information-dense; put "
    "secondary detail in a short body line separated by \": \" when needed.\n"
    "- FORBIDDEN: content-free messages that could describe any diff, e.g. "
    "\"update app\", \"fixed bug\", \"bug fixes\", \"minor changes\", \"changes\", "
    "\"improvements\", \"wip\", \"refactor code\", \"update files\", \"cleanup\". "
    "If you cannot name WHAT changed, re-read the diff before answering."
)

_FAST_SYSTEM_PROMPT = (
    "You analyze git diffs and decide whether the changes form one logical commit or "
    "should be split into multiple commits. Group only changes that are clearly related. "
    "You may split a single file's hunks across groups when parts of it belong to "
    "different logical changes. Commit messages must be specific to the diff — see the "
    "message rules below.\n" + _JSON_INSTRUCTION
)

_REASONING_SYSTEM_PROMPT = (
    "You are a meticulous reviewer deciding how a git diff should be split into commits. "
    "Think carefully through each hunk: what code path it belongs to, which other hunks it depends on, "
    "and whether the changes could land independently without breaking the build. "
    "Only group changes when there is a concrete dependency or shared purpose; when in doubt, split. "
    "Weigh file paths, import changes, shared types, and call sites as evidence of coupling. "
    "When two hunks of the SAME file belong to different logical changes, assign them to "
    "different groups via partial_files instead of lumping them together. Commit messages must "
    "be specific to the diff — see the message rules below.\n" + _JSON_INSTRUCTION
)

_RECHECK_FOLLOW_UP = (
    "Re-examine your proposed grouping. For each group you kept together, confirm the changes "
    "could not be committed independently. If any group mixes unrelated concerns, revise the plan. "
    "Respond with the same JSON format."
)

_MESSAGE_QUALITY_FOLLOW_UP = (
    "At least one commit_message is too vague to describe this diff (e.g. it says only "
    "\"update\" or \"fix bug\" without naming what changed). Rewrite every commit_message so each "
    "one names the concrete action and target drawn from the diff — file paths, functions, "
    "components. Keep the same grouping. Respond with the same JSON format."
)

_BRANCH_SYSTEM_PROMPT = (
    "You suggest a concise, descriptive git branch name for the changes in a diff. "
    "Read the diff and produce a short kebab-case branch name: lowercase words joined by "
    "hyphens, no spaces, no backticks, no explanation. Prefer a conventional-commit-style "
    "prefix when it fits the primary change (feat-, fix-, refactor-, chore-, docs-, test-, "
    "perf-, style-, ci-, build-). Keep it under 40 characters and focused on the single main "
    "intent of the change. Respond with a single JSON object and nothing else:\n"
    '{"branch_name": "kebab-case-name"}'
)

_BRANCH_CHARS_RE = re.compile(r"[^a-z0-9/-]+")


def _clean_branch_name(raw: object) -> str | None:
    """Coerce a model branch name into a safe, git-friendly slug."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower().strip("`").strip()
    s = _BRANCH_CHARS_RE.sub("-", s)
    s = re.sub(r"[-/]{2,}", "-", s).strip("-/")
    if not s or len(s) > 60:
        return None
    return s[:60]

# A subject made purely of these verbs/filler words says nothing about
# the diff. Matched against the subject with any conventional-commit
# prefix stripped, lowercased.
_GENERIC_PHRASE_RE = re.compile(
    r"(?:(?:minor|small|misc|some|various|quick|little)\s+)?"
    r"(?:"
    r"update[sd]?|change[sd]?|fix(?:ed|es)?|bug ?fix(?:es)?|hotfix(?:es)?|patch(?:es)?|"
    r"wip|work in progress|improve(?:ment)?s?|refactor(?:ed)?|clean ?up|tweak(?:s|ed)?|"
    r"adjustments?"
    r")"
)

# Words that add no specificity after a generic verb ("update the app",
# "fix some bugs"). Anything left over means the message names something.
_GENERIC_FILLER = frozenset({
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "with",
    "some", "various", "small", "minor", "little", "few", "more",
    "app", "apps", "code", "codebase", "bug", "bugs", "stuff", "thing",
    "things", "file", "files", "project", "repo", "repos", "repository",
    "it", "this", "that", "these", "those", "my", "our",
})

_CONVENTIONAL_PREFIX_RE = re.compile(
    r"^(?:build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(?:\([^)]*\))?:\s*",
    re.IGNORECASE,
)


def _subject_of(message: str) -> str:
    """Subject line with an optional conventional prefix stripped."""
    subject = message.strip().splitlines()[0] if message.strip() else ""
    return _CONVENTIONAL_PREFIX_RE.sub("", subject).strip().lower()


def _vacuous_subject(subject: str) -> bool:
    if not subject:
        return True
    # Pure generic verb phrase: "update", "bug fixes", "minor changes".
    if _GENERIC_PHRASE_RE.fullmatch(subject):
        return True
    # Verb + filler only: "update the app", "fix some bugs". Strip the
    # leading verb, then see whether any concrete word survives.
    m = _GENERIC_PHRASE_RE.match(subject)
    remainder = subject[m.end() :] if m else subject
    words = re.split(r"[\s\-_,.;:]+", remainder)
    words = [w for w in words if w]
    return bool(words) and all(w in _GENERIC_FILLER for w in words)


def _generic_messages(groups: list[ChangeGroup]) -> list[str]:
    """Messages that say nothing about what changed.

    Vacuous = built from generic verbs and filler ("update app", "fixed
    bug", "minor changes") or a handful of tiny words with no concrete
    noun ("wip stuff").
    """
    flagged = []
    for group in groups:
        subject = _subject_of(group.commit_message)
        words = subject.replace("-", " ").replace("_", " ").split()
        if not any(len(w) > 3 for w in words) or _vacuous_subject(subject):
            flagged.append(group.commit_message)
    return flagged


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


def _local_commit_message(diff: str, top: str) -> str:
    """Deterministic fallback message naming the actual files touched.

    The local tier can't understand intent, but "update 3 files in src"
    is strictly better than a bare "update src": each file gets an
    add/remove/update verb derived from the diff headers.
    """
    actions: OrderedDict[str, str] = OrderedDict()
    old_path: str | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            old_path = None
        elif line.startswith("--- "):
            old_path = line[4:].strip()
        elif line.startswith("+++ b/"):
            new_path = line[6:].strip()
            base = new_path.rsplit("/", 1)[-1] if new_path else "file"
            old_is_dev_null = old_path in (None, "", "/dev/null")
            action = f"add {base}" if old_is_dev_null else f"update {base}"
            actions.setdefault(f"{new_path}:{action}", action)
        elif line.startswith("+++ /dev/null"):
            name = (old_path or "").removeprefix("a/").rsplit("/", 1)[-1] or "file"
            actions.setdefault(f"del:{name}", f"remove {name}")
    if not actions:
        return f"chore({top}): update {top} files"
    listed = list(actions.values())
    summary = "; ".join(listed[:5])
    if len(listed) > 5:
        summary += f"; +{len(listed) - 5} more"
    return f"chore({top}): {summary}"


def _group_by_top_level(files: list[str], diff: str) -> list[ChangeGroup]:
    buckets: OrderedDict[str, list[str]] = OrderedDict()
    for path in files:
        top = path.split("/", 1)[0] if "/" in path else "(root)"
        buckets.setdefault(top, []).append(path)
    return [
        ChangeGroup(
            files=paths,
            commit_message=_local_commit_message(diff, top),
            rationale=f"All changed files live under '{top}'",
        )
        for top, paths in buckets.items()
    ]


def _is_large(diff: str, files: list[str], settings: Settings) -> bool:
    return len(files) > settings.analyze_escalation_files or _diff_line_count(diff) > settings.analyze_escalation_diff_lines


def _heuristic_pass(payload: AnalyzeRequest, settings: Settings) -> tuple[AnalyzeResponse | None, list[str]]:
    # Commit requests always pay for a real message; the deterministic
    # tier is only good enough for scan's preview line.
    if payload.mode == "commit":
        return None, _extract_files(payload.diff)
    files = _extract_files(payload.diff)
    if not files or _is_large(payload.diff, files, settings):
        return None, files
    groups = _group_by_top_level(files, payload.diff)
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


async def _suggest_branch_name(diff: str, settings: Settings) -> str | None:
    """One focused fast-model call that returns a kebab-case branch name."""
    client = get_openrouter_client()
    user_prompt = diff if len(diff) <= 200_000 else diff[:200_000]
    try:
        result = await client.chat_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": _BRANCH_SYSTEM_PROMPT},
                {"role": "user", "content": "Diff:\n" + user_prompt},
            ],
        )
        raw = result.content.get("branch_name") if isinstance(result.content, dict) else None
        if not raw:
            raw = result.raw_content
        return _clean_branch_name(raw)
    except (OpenRouterError, ValueError) as exc:
        logger.warning("branch name suggestion failed (%s)", exc)
        return None


async def _rewrite_generic_messages(
    client: Any,
    model: str,
    groups: list[ChangeGroup],
    *,
    base_messages: list[dict[str, Any]],
) -> list[ChangeGroup]:
    """One bounded retry when the fast model returns vacuous messages."""
    vague = _generic_messages(groups)
    if not vague:
        return groups
    logger.warning("vague commit message(s) %s — requesting rewrite", vague)
    messages = [*base_messages, {"role": "user", "content": _MESSAGE_QUALITY_FOLLOW_UP}]
    try:
        retry = await client.chat_completion(model=model, messages=messages)
        retry_groups, _ = _parse_groups(retry)
        if not _generic_messages(retry_groups):
            return retry_groups
        logger.warning("rewritten messages still vague; keeping original grouping")
    except (OpenRouterError, ValueError) as exc:
        logger.warning("commit-message rewrite pass failed (%s); keeping original", exc)
    return groups


async def analyze_diff(payload: AnalyzeRequest, settings: Settings | None = None) -> AnalyzeResponse:
    settings = settings or get_settings()
    started = time.monotonic()

    # "branch" mode only needs a single focused call that returns a
    # suggested branch name — no grouping, no escalation chain.
    if payload.mode == "branch":
        name = await _suggest_branch_name(payload.diff, settings)
        return AnalyzeResponse(
            groups=[],
            confidence=1.0,
            model_tier=ModelTier.fast,
            branch_name=name,
        )

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
            groups = await _rewrite_generic_messages(
                client,
                settings.openrouter_model,
                groups,
                base_messages=[
                    {"role": "system", "content": _FAST_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    assistant_message(fast_result.raw_content, fast_result.reasoning_details),
                ],
            )
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

        recheck = None
        if confidence < 0.9:
            if time.monotonic() - started < settings.analyze_chain_deadline_seconds:
                recheck = await client.chat_completion_with_reasoning(
                    system_prompt=_REASONING_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    history=history,
                    follow_up=_RECHECK_FOLLOW_UP,
                )
            else:
                logger.info(
                    "skipping reasoning recheck turn (chain deadline %.0fs exceeded at %.1fs)",
                    settings.analyze_chain_deadline_seconds,
                    time.monotonic() - started,
                )
        if recheck is not None:
            try:
                revised_groups, revised_confidence = _parse_groups(recheck)
                groups, confidence = revised_groups, revised_confidence
                history.append(assistant_message(recheck.raw_content, recheck.reasoning_details))
                reasoning_result = recheck
            except ValueError as exc:
                logger.warning("recheck turn unparseable (%s), keeping first reasoning verdict", exc)

        # Same bounded quality net for the reasoning tier: one extra
        # turn asking for concrete messages; never fails the request.
        if _generic_messages(groups) and time.monotonic() - started < settings.analyze_chain_deadline_seconds:
            logger.warning(
                "reasoning model returned vague commit message(s) %s — requesting rewrite",
                _generic_messages(groups),
            )
            try:
                fix = await client.chat_completion_with_reasoning(
                    system_prompt=_REASONING_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    history=history,
                    follow_up=_MESSAGE_QUALITY_FOLLOW_UP,
                )
                fixed_groups, fixed_confidence = _parse_groups(fix)
                if not _generic_messages(fixed_groups):
                    groups, confidence = fixed_groups, fixed_confidence
                    history.append(assistant_message(fix.raw_content, fix.reasoning_details))
                    reasoning_result = fix
                else:
                    logger.warning("rewritten reasoning messages still vague; keeping verdict")
            except (OpenRouterError, ValueError) as exc:
                logger.warning("commit-message rewrite turn failed (%s); keeping verdict", exc)

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
