import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from app.config import Settings


@pytest.fixture
def settings_override(monkeypatch):
    """Provide a Settings instance with test defaults."""
    return Settings(
        database_url="sqlite+aiosqlite:///./test.db",
        jwt_secret="test-secret",
        openrouter_api_key="sk-test",
        openrouter_model="test/model",
        openrouter_reasoning_model="test/model",
        max_hunks_per_review=5,
    )


def _make_chat_result(raw_content: str) -> MagicMock:
    result = MagicMock()
    result.raw_content = raw_content
    result.model = "test/model"
    result.latency_seconds = 0.1
    result.usage = {}
    result.reasoning_details = []
    return result


def _make_chat_result_from_dict(obj: dict) -> MagicMock:
    import json
    return _make_chat_result(json.dumps(obj))


@pytest_asyncio.fixture
async def mock_openrouter(monkeypatch):
    """Patch get_openrouter_client to return a controllable mock."""
    mock_client = AsyncMock()
    monkeypatch.setattr(
        "app.services.pr_review.get_openrouter_client",
        lambda: mock_client,
    )
    return mock_client


@pytest.fixture
def make_review_request():
    """Factory for ReviewRequest payloads with minimal boilerplate."""

    def _make(hunks, repo_full_name="test/repo", pr_number=1):
        from app.schemas.review import HunkInput, ReviewRequest

        if hunks is None:
            hunks = [
                HunkInput(
                    file_path="src/foo.py",
                    hunk_index=0,
                    patch_text="@@ -1,3 +1,4 @@\n+import os\n import sys\n+import json\n import re",
                    start_line=1,
                ),
            ]
        return ReviewRequest(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            hunks=hunks,
        )

    return _make
