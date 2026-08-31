from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HunkInput(BaseModel):
    file_path: str
    hunk_index: int
    patch_text: str
    start_line: int


class ReviewRequest(BaseModel):
    repo_full_name: str
    pr_number: int
    hunks: list[HunkInput] = Field(min_length=1)


class HunkReviewResult(BaseModel):
    file_path: str
    hunk_index: int
    start_line: int
    has_issue: bool
    severity: Literal["info", "warning", "error"]
    explanation: str
    suggested_fix: str | None = None


class ReviewResponse(BaseModel):
    results: list[HunkReviewResult]
    truncated: bool
