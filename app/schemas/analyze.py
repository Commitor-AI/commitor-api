from enum import Enum

from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    local = "local"
    fast = "fast"
    reasoning = "reasoning"


class AnalyzeRequest(BaseModel):
    diff: str = Field(min_length=1, max_length=200_000)
    context: str | None = Field(default=None, max_length=4_000)
    # "commit" requests always get model-generated messages; the free
    # deterministic local tier is reserved for scan previews. "branch"
    # returns only an AI-suggested branch name for the diff.
    mode: str = Field(default="scan", pattern="^(scan|commit|branch)$")


class PartialFile(BaseModel):
    """A file whose changes are split across commits at hunk level.

    `hunks` are 1-based indices into that path's hunk sequence, in the
    order the hunks appear in the analyzed diff.
    """

    path: str
    hunks: list[int] = Field(min_length=1)


class ChangeGroup(BaseModel):
    files: list[str]
    commit_message: str
    rationale: str
    # Files (or parts of files) assigned to this group beyond whole-file
    # membership. Empty for purely whole-file groups; the local
    # heuristic pass never sets it.
    partial_files: list[PartialFile] = []


class AnalyzeResponse(BaseModel):
    groups: list[ChangeGroup]
    confidence: float = Field(ge=0.0, le=1.0)
    model_tier: ModelTier
    # AI-suggested kebab-case branch name (populated for mode="branch",
    # absent otherwise). CLI uses it to pre-fill `commitor commit -b`.
    branch_name: str | None = None
