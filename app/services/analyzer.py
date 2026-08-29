import logging
import re
import time
from collections import OrderedDict
from typing import Any

from fastapi import HTTPException, status

from app.config import Settings, get_settings
from app.schemas.analyze import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChangeGroup,
    ModelTier,
    PartialFile,
)
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
    "Rules for paths:\n"
    "- Every path in `files` and `partial_files` MUST be the EXACT full path as it "
    "appears in the diff (the text after `diff --git a/… b/`), including any "
    "`crates/`, `src/`, or directory prefixes. Never shorten a path to its basename "
    "or a repo-relative form.\n"
    "- A path not present verbatim in the diff will be rejected; double-check the "
    "prefixes against the diff before answering.\n"
    "\n"
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


def _categorize(path: str) -> str:
    """Mirror the CLI's file categorization used for conventional typing."""
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    if (
        "/tests/" in lower
        or "/test/" in lower
        or "__tests__" in lower
        or name.startswith("test_")
        or name.endswith("_test.rs")
        or name.endswith("_test.py")
        or name.endswith(".test.ts")
        or name.endswith(".test.js")
        or name.endswith("_spec.rs")
        or name.endswith("_spec.py")
    ):
        return "test"
    if (
        name.endswith(".md")
        or name.endswith(".rst")
        or "/docs/" in lower
        or name.startswith("readme")
        or name.startswith("changelog")
        or name.startswith("license")
    ):
        return "docs"
    build_names = {
        "cargo.toml",
        "cargo.lock",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "dockerfile",
        "makefile",
        "justfile",
        "build.rs",
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "docker-compose.yml",
        "composer.json",
        "gemfile",
        ".gitignore",
    }
    if (
        name in build_names
        or "/.github/" in lower
        or "/.gitlab/" in lower
        or lower.endswith(".tf")
        or lower.endswith(".toml")
        or lower.endswith(".yml")
        or lower.endswith(".yaml")
    ):
        return "build"
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    source_exts = {
        "rs",
        "py",
        "js",
        "ts",
        "jsx",
        "tsx",
        "go",
        "java",
        "c",
        "h",
        "cpp",
        "hpp",
        "cc",
        "rb",
        "php",
        "swift",
        "kt",
        "scala",
        "sh",
        "sql",
        "html",
        "css",
        "scss",
        "sass",
        "vue",
        "elm",
        "ex",
        "exs",
        "clj",
        "lua",
        "dart",
    }
    if ext in source_exts:
        return "source"
    return "other"


def _parse_file_changes(diff: str) -> list[dict]:
    """Reconstruct per-file change metadata from the patch (mirror of the CLI)."""
    changes: list[dict] = []
    current: dict | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                changes.append(current)
            rest = line[len("diff --git ") :]
            path = rest.split(" b/")[-1]
            current = {"path": path, "kind": "modified", "added": 0, "removed": 0}
        elif line.startswith("new file mode"):
            if current is not None:
                current["kind"] = "added"
        elif line.startswith("deleted file mode"):
            if current is not None:
                current["kind"] = "deleted"
        elif line.startswith("+") and not line.startswith("+++"):
            if current is not None:
                current["added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            if current is not None:
                current["removed"] += 1
    if current is not None:
        changes.append(current)
    return changes


def _classify_changes(changes: list[dict]) -> str:
    """Conventional-Commits type, derived from the diff (mirror of the CLI)."""
    cats = [_categorize(c["path"]) for c in changes]
    if changes and all(c == "test" for c in cats):
        return "test"
    if changes and all(c == "docs" for c in cats):
        return "docs"
    if changes and all(c == "build" for c in cats):
        return "build"

    def code_like(cat: str) -> bool:
        return cat in ("source", "other")

    # New functionality: a new file, or code added to existing files.
    if any(
        c["kind"] in ("added", "modified")
        and code_like(_categorize(c["path"]))
        and c["added"] > 0
        and c["removed"] == 0
        for c in changes
    ):
        return "feat"

    added = sum(c["added"] for c in changes)
    removed = sum(c["removed"] for c in changes)
    if len(changes) > 1 and removed > added and removed > 0:
        return "refactor"
    if removed > 0 or any(c["kind"] == "deleted" for c in changes):
        return "fix"
    return "refactor"


def _subject_for(changes: list[dict]) -> str:
    def code_like(cat: str) -> bool:
        return cat in ("source", "other")

    primary = next(
        (c for c in changes if c["kind"] == "added" and code_like(_categorize(c["path"]))),
        None,
    ) or next((c for c in changes if code_like(_categorize(c["path"]))), None) or (
        changes[0] if changes else None
    )
    if not primary:
        return "update working changes"
    verb = {"added": "add", "deleted": "remove", "modified": "update"}[primary["kind"]]
    filename = primary["path"].rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0] or filename
    n = len(changes)
    if n == 1:
        return f"{verb} {stem}"
    return f"{verb} {stem} and {n - 1} more"


def _conventional_local_message(diff: str, top: str, paths: list[str]) -> str:
    """Deterministic conventional-commit message for the local tier.

    Mirrors the CLI's offline generator so the local fallback reads like
    `feat(auth): add login` instead of a flat `chore(...): add x; update y`.
    """
    all_changes = _parse_file_changes(diff)
    path_set = set(paths)
    changes = [c for c in all_changes if c["path"] in path_set]
    if not changes:
        return f"chore: update {top} files"
    ctype = _classify_changes(changes)
    subject = _subject_for(changes)
    if top == "(root)":
        return f"{ctype}: {subject}"
    return f"{ctype}({top}): {subject}"


def _parent_dir(path: str) -> str | None:
    """Parent directory of a path, used as the per-file Conventional-Commits
    scope (`None` for files at the repo root)."""
    parts = path.split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else None


def _scope_for(path: str) -> str | None:
    """Conventional-Commits scope with generic source roots stripped, so a
    path like `crates/cli/src/auth.rs` yields `cli/auth` (not
    `crates/cli/src/auth`) and a crate's top-level source file like
    `crates/cli/src/main.rs` yields `cli` (not the uninformative `src`).
    Mirrors the CLI's `scope_for`."""
    parent = _parent_dir(path)
    if not parent:
        return None
    return _trim_source_root(parent) or None


def _trim_source_root(directory: str) -> str:
    if directory.startswith("crates/"):
        rest = directory[len("crates/"):]
        crate, _, after = rest.partition("/")
        if after == "src":
            return crate
        if after.startswith("src/"):
            inner = after[len("src/"):]
            return f"{crate}/{inner}" if inner else crate
        if after:
            return f"{crate}/{after}"
        return crate
    for root in ("src/", "lib/", "app/", "include/", "tests/", "test/"):
        if directory.startswith(root):
            return directory[len(root):]
    return directory


def _parse_file_hunks(diff: str) -> list[dict]:
    """Per-file hunk breakdown with added/removed line counts, in diff order.

    Used by the local tier to split a file's add-only hunks (a feature)
    from its modified hunks (a fix) at hunk granularity.
    """
    files: list[dict] = []
    current: dict | None = None
    hunk: dict | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                files.append(current)
            path = line[len("diff --git ") :].split(" b/")[-1]
            current = {"path": path, "kind": "modified", "hunks": []}
            hunk = None
        elif line.startswith("new file mode"):
            if current is not None:
                current["kind"] = "added"
        elif line.startswith("deleted file mode"):
            if current is not None:
                current["kind"] = "deleted"
        elif line.startswith("@@"):
            if current is None:
                continue
            hunk = {"index": len(current["hunks"]) + 1, "added": 0, "removed": 0}
            current["hunks"].append(hunk)
        elif line.startswith("+") and not line.startswith("+++"):
            if hunk is not None:
                hunk["added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            if hunk is not None:
                hunk["removed"] += 1
    if current is not None:
        files.append(current)
    return files


def _file_segments(file_change: dict) -> list[tuple[str, bool, list[int] | None]]:
    """One or more `(type, is_whole, hunk_ids)` segments for a file.

    A modified file whose hunks split into add-only (feature) and
    modified/deleted (fix) parts yields two hunk-level segments; every
    other file stays whole.
    """
    hunks = file_change["hunks"]
    kind = file_change["kind"]
    if kind == "added":
        return [("feat", True, None)]
    if kind == "deleted":
        return [("fix", True, None)]
    add_only = [h["index"] for h in hunks if h["added"] > 0 and h["removed"] == 0]
    others = [h["index"] for h in hunks if not (h["added"] > 0 and h["removed"] == 0)]
    if add_only and others:
        return [("feat", False, add_only), ("fix", False, others)]
    if add_only:
        return [("feat", True, None)]
    if others:
        return [("fix", True, None)]
    return [("chore", True, None)]


def _subject_for_paths(paths: list[str], ctype: str) -> str:
    """Imperative subject derived from file stems (mirror of the CLI)."""
    if not paths:
        return "update working changes"

    def code_like(cat: str) -> bool:
        return cat in ("source", "other")

    primary = next((p for p in paths if code_like(_categorize(p))), paths[0])
    verb = {
        "feat": "add",
        "fix": "update",
        "refactor": "refactor",
        "docs": "update",
        "test": "add",
        "build": "update",
        "chore": "update",
    }.get(ctype, "update")
    filename = primary.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0] or filename
    n = len(paths)
    if n == 1:
        return f"{verb} {stem}"
    return f"{verb} {stem} and {n - 1} more"


def _split_by_type_scope(diff: str, files: list[str]) -> list[ChangeGroup]:
    """Split a diff into one commit per `(type, scope)`, at hunk granularity
    where a single file mixes feature additions with fixes. Mirrors the
    CLI's `offline_groups` and extends it with hunk-level splitting.

    - distinct features, distinct fixes, and the remainder each get their
      own commit,
    - a modified file with both add-only hunks and modified hunks is split
      so the additions commit as `feat` and the edits as `fix`.
    """
    parsed = _parse_file_hunks(diff)
    path_set = set(files)
    parsed = [f for f in parsed if f["path"] in path_set]
    if not parsed:
        return [
            ChangeGroup(
                files=list(files),
                commit_message=_conventional_local_message(diff, "(root)", list(files)),
                rationale="All changed files",
            )
        ]

    buckets: dict[tuple[str, str | None], dict] = {}
    order: list[tuple[str, str | None]] = []
    for f in parsed:
        scope = _scope_for(f["path"])
        for ctype, is_whole, hunk_ids in _file_segments(f):
            key = (ctype, scope)
            if key not in buckets:
                buckets[key] = {"files": [], "partial": []}
                order.append(key)
            if is_whole:
                if f["path"] not in buckets[key]["files"]:
                    buckets[key]["files"].append(f["path"])
            else:
                buckets[key]["partial"].append({"path": f["path"], "hunks": hunk_ids})

    groups: list[ChangeGroup] = []
    for key in order:
        ctype, scope = key
        bucket = buckets[key]
        top = scope if scope else "(root)"
        subject_paths = list(bucket["files"]) + [p["path"] for p in bucket["partial"]]
        if bucket["partial"]:
            subject = _subject_for_paths(subject_paths, ctype)
            prefix = f"{ctype}({scope})" if scope else ctype
            message = f"{prefix}: {subject}"
        else:
            message = _conventional_local_message(diff, top, bucket["files"])
        partial_files = [PartialFile(path=p["path"], hunks=p["hunks"]) for p in bucket["partial"]]
        rationale = f"Changes classified as {ctype}" + (f" under '{scope}'" if scope else "")
        groups.append(
            ChangeGroup(
                files=bucket["files"],
                commit_message=message,
                rationale=rationale,
                partial_files=partial_files,
            )
        )
    return groups


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
    groups = _split_by_type_scope(payload.diff, files)
    response = AnalyzeResponse(groups=groups, confidence=0.8, model_tier=ModelTier.local)
    return response, files


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
