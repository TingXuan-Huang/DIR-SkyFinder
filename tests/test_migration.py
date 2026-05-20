"""Migration script tests: flat layout -> nested layout, with safety check for in-flight jobs."""
import json
from pathlib import Path

from skyfinder.training.migrate import plan_migration, run_migration


def test_plan_empty_dir(tmp_path):
    moves, blockers = plan_migration(tmp_path)
    assert moves == []
    assert blockers == []


def test_plan_flat_layout(tmp_path):
    # Build a fake flat layout.
    (tmp_path / "baseline_resnet50_fold0.pt").write_bytes(b"fake-pt")
    (tmp_path / "baseline_resnet50_fold0.json").write_text(json.dumps({"run_name": "baseline_resnet50_fold0"}))
    (tmp_path / "lds_resnet50_fold0.pt").write_bytes(b"fake-pt")
    moves, blockers = plan_migration(tmp_path)
    assert len(moves) == 3
    assert blockers == []
    # Every move goes into a per-experiment subfolder.
    for src, dst in moves:
        assert dst.parent.name in {"baseline_resnet50", "lds_resnet50"}


def test_plan_aborts_on_last_pt(tmp_path):
    """In-flight job indicator -> blocker, no moves."""
    (tmp_path / "baseline_resnet50_fold0_last.pt").write_bytes(b"fake-pt")
    (tmp_path / "baseline_resnet50_fold0.pt").write_bytes(b"fake-pt")
    moves, blockers = plan_migration(tmp_path)
    assert len(blockers) == 1
    assert blockers[0].name == "baseline_resnet50_fold0_last.pt"


def test_run_migration_dry_run(tmp_path):
    (tmp_path / "baseline_resnet50_fold0.pt").write_bytes(b"fake-pt")
    rc = run_migration(tmp_path, dry_run=True)
    assert rc == 0
    # Dry run -> file still at flat location.
    assert (tmp_path / "baseline_resnet50_fold0.pt").exists()
    assert not (tmp_path / "baseline_resnet50" / "baseline_resnet50_fold0.pt").exists()


def test_run_migration_executes(tmp_path):
    (tmp_path / "baseline_resnet50_fold0.pt").write_bytes(b"fake-pt")
    (tmp_path / "baseline_resnet50_fold0.json").write_text("{}")
    rc = run_migration(tmp_path, dry_run=False)
    assert rc == 0
    # Files moved into nested layout.
    assert (tmp_path / "baseline_resnet50" / "baseline_resnet50_fold0.pt").exists()
    assert (tmp_path / "baseline_resnet50" / "baseline_resnet50_fold0.json").exists()
    # Originals gone.
    assert not (tmp_path / "baseline_resnet50_fold0.pt").exists()


def test_run_migration_blocked_by_last_pt(tmp_path):
    (tmp_path / "baseline_resnet50_fold0_last.pt").write_bytes(b"fake-pt")
    rc = run_migration(tmp_path, dry_run=False)
    assert rc == 2


def test_run_migration_already_nested(tmp_path):
    """If nothing's at flat root, exit code 3 ("nothing to do")."""
    sub = tmp_path / "baseline_resnet50"
    sub.mkdir()
    (sub / "baseline_resnet50_fold0.pt").write_bytes(b"fake-pt")
    rc = run_migration(tmp_path, dry_run=False)
    assert rc == 3
