import asyncio
import logging
from typing import Any

from app.config import get_settings
from app.schemas.review import (
    HunkInput,
    HunkReviewResult,
    ReviewRequest,
    ReviewResponse,
)
from app.services.openrouter import (
    OpenRouterError,
    get_openrouter_client,
    parse_json_object,
)

logger = logging.getLogger(__name__)

_REVIEW_SYSTEM_PROMPT = (
    "You are a code-review expert. Review the following diff hunk for actual "
    "correctness issues: logic bugs, unhandled errors, null/None dereference, "
    "race conditions, off-by-one errors, missing edge cases, resource leaks, "
    "or broken invariants.\n\n"
    "Do NOT flag style, naming, formatting, or subjective preferences.\n\n"
    "Respond with a single JSON object and nothing else:\n"
    '{"has_issue": bool, "severity": "info"|"warning"|"error", '
    '"explanation": "what is wrong and why", '
    '"suggested_fix": "replacement code or null if no safe one-line fix"}\n\n'
    "Rules:\n"
    "- has_issue: true only when there is a concrete correctness problem.\n"
    "- severity: \"error\" for bugs that will break at runtime or corrupt data; "
    "\"warning\" for likely bugs or missing edge cases; \"info\" for correctness "
    "nits that probably won't cause a failure but are worth noting.\n"
    "- suggested_fix: provide a concrete code suggestion only when the fix is "
    "a small, safe replacement for the hunk. Set to null for semantic or logic "
    "issues that need broader refactoring.\n"
    "- explanation: be specific — name the variable, function, or condition "
    "that is wrong.\n"
)

_CONCURRENCY_LIMIT = 5

# Fields the LLM must return (excluding file_path / hunk_index / start_line
# which we overlay from the request, not from the model).
_LLM_FIELDS = {"has_issue", "severity", "explanation", "suggested_fix"}
_SEVERITY_VALUES = {"info", "warning", "error"}


def _parse_review_json(raw: str) -> dict[str, Any] | None:
    """Best-effort parse of the LLM's JSON response.

    Returns the parsed dict on success, ``None`` on any failure so the
    caller can skip the hunk gracefully.
    """
    try:
        obj = parse_json_object(raw)
    except OpenRouterError:
        logger.warning("LLM review response is not valid JSON: %s", raw[:200])
        return None
    if not _LLM_FIELDS <= obj.keys():
        logger.warning("LLM review response missing fields: %s", raw[:200])
        return None
    if obj["severity"] not in _SEVERITY_VALUES:
        logger.warning("LLM review severity %r not in %s", obj["severity"], _SEVERITY_VALUES)
        return None
    return obj


async def _review_single_hunk(
    hunk: HunkInput,
    *,
    model: str,
    semaphore: asyncio.Semaphore,
) -> HunkReviewResult | None:
    """Review one hunk. Returns ``None`` on parse failure or has_issue=false."""
    user_prompt = (
        f"File: {hunk.file_path}\n"
        f"Hunk index: {hunk.hunk_index}\n\n"
        f"```diff\n{hunk.patch_text}\n```"
    )
    async with semaphore:
        try:
            client = get_openrouter_client()
            result = await client.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=30.0,
            )
        except (OpenRouterError, Exception) as exc:
            logger.warning("LLM review failed for %s#%d: %s", hunk.file_path, hunk.hunk_index, exc)
            return None

    parsed = _parse_review_json(result.raw_content)
    if parsed is None:
        return None

    if not parsed["has_issue"]:
        return None

    return HunkReviewResult(
        file_path=hunk.file_path,
        hunk_index=hunk.hunk_index,
        start_line=hunk.start_line,
        has_issue=True,
        severity=parsed["severity"],
        explanation=parsed["explanation"],
        suggested_fix=parsed.get("suggested_fix"),
    )


async def review_hunks(request: ReviewRequest) -> ReviewResponse:
    """Review all hunks in a request concurrently, returning only flagged ones."""
    settings = get_settings()
    max_hunks = settings.max_hunks_per_review

    truncated = len(request.hunks) > max_hunks
    hunks = request.hunks[:max_hunks]

    semaphore = asyncio.Semaphore(_CONCURRENCY_LIMIT)
    tasks = [
        _review_single_hunk(
            hunk,
            model=settings.openrouter_model,
            semaphore=semaphore,
        )
        for hunk in hunks
    ]

    results_raw = await asyncio.gather(*tasks)
    results = [r for r in results_raw if r is not None]

    return ReviewResponse(results=results, truncated=truncated)
