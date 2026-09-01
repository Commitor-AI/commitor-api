import pytest
from app.schemas.analyze import ChangeGroup, PartialFile
from app.services.analyzer import _validate_partition


def test_valid_partition_all_whole_files():
    groups = [
        ChangeGroup(
            files=["src/services/audio.rs", "src/services/network.rs"],
            commit_message="feat(services): update audio and network",
            rationale="Services changes",
        ),
        ChangeGroup(
            files=["src/widgets/audio.rs", "src/widgets/network.rs"],
            commit_message="feat(widgets): update widgets",
            rationale="Widgets changes",
        ),
    ]
    files = [
        "src/services/audio.rs",
        "src/services/network.rs",
        "src/widgets/audio.rs",
        "src/widgets/network.rs",
    ]
    err = _validate_partition(groups, files)
    assert err is None


def test_valid_partition_with_partial_files():
    groups = [
        ChangeGroup(
            files=["src/app.rs"],
            commit_message="feat(app): add feature",
            rationale="App update",
            partial_files=[PartialFile(path="src/widgets/mod.rs", hunks=[1])],
        ),
        ChangeGroup(
            files=[],
            commit_message="fix(widgets): fix widget mod",
            rationale="Widget fix",
            partial_files=[PartialFile(path="src/widgets/mod.rs", hunks=[2])],
        ),
    ]
    files = ["src/app.rs", "src/widgets/mod.rs"]
    err = _validate_partition(groups, files)
    assert err is None


def test_duplicate_whole_file_assignment():
    groups = [
        ChangeGroup(
            files=["src/widgets/audio.rs"],
            commit_message="feat: audio 1",
            rationale="first claim",
        ),
        ChangeGroup(
            files=["src/widgets/audio.rs"],
            commit_message="feat: audio 2",
            rationale="second claim",
        ),
    ]
    err = _validate_partition(groups, ["src/widgets/audio.rs"])
    assert err is not None
    assert "'src/widgets/audio.rs' is assigned whole to more than one group" in err


def test_mixed_whole_and_partial_assignment():
    groups = [
        ChangeGroup(
            files=["src/widgets/mod.rs"],
            commit_message="feat: whole",
            rationale="claimed whole",
        ),
        ChangeGroup(
            files=[],
            commit_message="fix: partial",
            rationale="claimed partial",
            partial_files=[PartialFile(path="src/widgets/mod.rs", hunks=[1])],
        ),
    ]
    err = _validate_partition(groups, ["src/widgets/mod.rs"])
    assert err is not None
    assert "'src/widgets/mod.rs' is claimed both whole and partially" in err


def test_duplicate_hunk_assignment():
    groups = [
        ChangeGroup(
            files=[],
            commit_message="feat: part 1",
            rationale="hunk 1",
            partial_files=[PartialFile(path="src/events.rs", hunks=[1, 2])],
        ),
        ChangeGroup(
            files=[],
            commit_message="fix: part 2",
            rationale="hunk 2 again",
            partial_files=[PartialFile(path="src/events.rs", hunks=[2, 3])],
        ),
    ]
    err = _validate_partition(groups, ["src/events.rs"])
    assert err is not None
    assert "'src/events.rs' hunk 2 is assigned more than once" in err


def test_missing_file_omitted():
    groups = [
        ChangeGroup(
            files=["src/app.rs"],
            commit_message="feat: app only",
            rationale="app",
        ),
    ]
    err = _validate_partition(groups, ["src/app.rs", "src/events.rs"])
    assert err is not None
    assert "'src/events.rs' is not included in any group" in err


def test_empty_group():
    groups = [
        ChangeGroup(
            files=[],
            commit_message="empty",
            rationale="none",
            partial_files=[],
        ),
    ]
    err = _validate_partition(groups, ["src/app.rs"])
    assert err is not None
    assert "has no files or hunks assigned" in err
