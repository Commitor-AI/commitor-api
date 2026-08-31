import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.config import Settings
from app.schemas.review import HunkInput, ReviewRequest
from app.services.pr_review import _parse_review_json, review_hunks


# ---------------------------------------------------------------------------
# _parse_review_json
# ---------------------------------------------------------------------------

class TestParseReviewJson:
    def test_valid_full_object(self):
        obj = {
            "has_issue": True,
            "severity": "error",
            "explanation": "Missing null check on line 5",
            "suggested_fix": "if result is None: return",
        }
        assert _parse_review_json(json.dumps(obj)) == obj

    def test_valid_minimal_object(self):
        obj = {
            "has_issue": False,
            "severity": "info",
            "explanation": "Looks fine",
            "suggested_fix": None,
        }
        assert _parse_review_json(json.dumps(obj)) == obj

    def test_valid_with_json_fences(self):
        obj = {
            "has_issue": True,
            "severity": "warning",
            "explanation": "Off-by-one",
            "suggested_fix": None,
        }
        raw = f"```json\n{json.dumps(obj)}\n```"
        assert _parse_review_json(raw) == obj

    def test_missing_field_returns_none(self):
        obj = {"has_issue": True, "severity": "error"}  # missing explanation, suggested_fix
        assert _parse_review_json(json.dumps(obj)) is None

    def test_invalid_severity_returns_none(self):
        obj = {
            "has_issue": True,
            "severity": "critical",
            "explanation": "bad",
            "suggested_fix": None,
        }
        assert _parse_review_json(json.dumps(obj)) is None

    def test_non_json_returns_none(self):
        assert _parse_review_json("not json at all") is None

    def test_array_instead_of_object_returns_none(self):
        assert _parse_review_json("[1, 2, 3]") is None

    def test_empty_string_returns_none(self):
        assert _parse_review_json("") is None


# ---------------------------------------------------------------------------
# review_hunks — truncation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_truncation_when_hunks_exceed_limit(monkeypatch):
    """Hunks beyond max_hunks_per_review are dropped and truncated=True."""
    settings = Settings(
        database_url="sqlite+aiosqlite:///./test.db",
        jwt_secret="test",
        openrouter_api_key="sk-test",
        max_hunks_per_review=3,
    )
    monkeypatch.setattr("app.services.pr_review.get_settings", lambda: settings)

    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value=_noop_result(),
    )
    monkeypatch.setattr("app.services.pr_review.get_openrouter_client", lambda: mock_client)

    hunks = [
        HunkInput(file_path=f"f{i}.py", hunk_index=0, patch_text="patch", start_line=1)
        for i in range(10)
    ]
    request = ReviewRequest(repo_full_name="r/p", pr_number=1, hunks=hunks)

    response = await review_hunks(request)

    assert response.truncated is True
    assert len(response.results) <= 3  # at most the capped count flagged
    assert mock_client.chat_completion.call_count <= 3


# ---------------------------------------------------------------------------
# review_hunks — malformed LLM JSON for one hunk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_json_does_not_fail_batch(monkeypatch):
    """If one hunk's LLM response is unparseable, others still succeed."""
    settings = Settings(
        database_url="sqlite+aiosqlite:///./test.db",
        jwt_secret="test",
        openrouter_api_key="sk-test",
        max_hunks_per_review=10,
    )
    monkeypatch.setattr("app.services.pr_review.get_settings", lambda: settings)

    good_result = _make_result({
        "has_issue": True,
        "severity": "error",
        "explanation": "Bug here",
        "suggested_fix": None,
    })
    bad_result = _make_result("totally not json")

    call_count = 0

    async def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return bad_result  # first hunk returns garbage
        return good_result    # second hunk is fine

    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(side_effect=_side_effect)
    monkeypatch.setattr("app.services.pr_review.get_openrouter_client", lambda: mock_client)

    hunks = [
        HunkInput(file_path="a.py", hunk_index=0, patch_text="patch", start_line=1),
        HunkInput(file_path="b.py", hunk_index=0, patch_text="patch", start_line=10),
    ]
    request = ReviewRequest(repo_full_name="r/p", pr_number=1, hunks=hunks)

    response = await review_hunks(request)

    assert response.truncated is False
    # The malformed hunk is skipped; only the good one appears
    assert len(response.results) == 1
    assert response.results[0].file_path == "b.py"


# ---------------------------------------------------------------------------
# review_hunks — only has_issue=true in results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_only_has_issue_true_in_results(monkeypatch):
    """Hunks with has_issue=false or parse failures are filtered out."""
    settings = Settings(
        database_url="sqlite+aiosqlite:///./test.db",
        jwt_secret="test",
        openrouter_api_key="sk-test",
        max_hunks_per_review=10,
    )
    monkeypatch.setattr("app.services.pr_review.get_settings", lambda: settings)

    results_sequence = [
        _make_result({"has_issue": False, "severity": "info", "explanation": "ok", "suggested_fix": None}),
        _make_result({"has_issue": True, "severity": "warning", "explanation": "Watch out", "suggested_fix": None}),
        _make_result({"has_issue": False, "severity": "info", "explanation": "fine", "suggested_fix": None}),
    ]
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(side_effect=results_sequence)
    monkeypatch.setattr("app.services.pr_review.get_openrouter_client", lambda: mock_client)

    hunks = [
        HunkInput(file_path=f"f{i}.py", hunk_index=0, patch_text="patch", start_line=1)
        for i in range(3)
    ]
    request = ReviewRequest(repo_full_name="r/p", pr_number=1, hunks=hunks)

    response = await review_hunks(request)

    assert response.truncated is False
    assert len(response.results) == 1
    assert response.results[0].has_issue is True
    assert response.results[0].explanation == "Watch out"


# ---------------------------------------------------------------------------
# review_hunks — suggestions passed through
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_suggested_fix_passed_through(monkeypatch):
    settings = Settings(
        database_url="sqlite+aiosqlite:///./test.db",
        jwt_secret="test",
        openrouter_api_key="sk-test",
        max_hunks_per_review=10,
    )
    monkeypatch.setattr("app.services.pr_review.get_settings", lambda: settings)

    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(return_value=_make_result({
        "has_issue": True,
        "severity": "error",
        "explanation": "Missing null guard",
        "suggested_fix": "if x is None: return 0",
    }))
    monkeypatch.setattr("app.services.pr_review.get_openrouter_client", lambda: mock_client)

    hunks = [HunkInput(file_path="src/a.py", hunk_index=0, patch_text="patch", start_line=5)]
    request = ReviewRequest(repo_full_name="r/p", pr_number=1, hunks=hunks)

    response = await review_hunks(request)

    assert len(response.results) == 1
    assert response.results[0].suggested_fix == "if x is None: return 0"
    assert response.results[0].file_path == "src/a.py"
    assert response.results[0].start_line == 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(obj) -> MagicMock:
    result = MagicMock()
    if isinstance(obj, dict):
        result.raw_content = json.dumps(obj)
    else:
        result.raw_content = obj
    result.model = "test/model"
    result.latency_seconds = 0.1
    result.usage = {}
    result.reasoning_details = []
    return result


def _noop_result() -> MagicMock:
    return _make_result({
        "has_issue": False,
        "severity": "info",
        "explanation": "ok",
        "suggested_fix": None,
    })
