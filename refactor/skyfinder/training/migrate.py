"""One-shot migration: flat results layout -> nested per-experiment subfolders.

Before May 2026 the trainer wrote `results/<run>.pt` and `results/<run>.json` at
the flat root. After May 2026 everything goes into `results/<subdir>/<run>.{pt,json}`
where `<subdir>` is the run name minus `_fold{N}` (and `_ep{N}`).

For backwards compat we kept a dual-path lookup (`_resolve_load_path`) that
checked both. That was removed in the May 2026 restructure — now this script
migrates any leftover flat-layout files into the nested layout. Idempotent.

Usage:
    skyfinder data-prep --migrate-results --dry-run --root results/
    skyfinder data-prep --migrate-results --root results/
"""
from __future__ import annotations

import sys
from pathlib import Path

from .checkpoint import subdir_for


def _is_flat_artifact(path: Path) -> bool:
    """True if `path` is a flat-layout artifact (file at root of results_dir)."""
    if not path.is_file():
        return False
    return path.suffix in {".pt", ".json"}


def plan_migration(results_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Return (planned_moves, in_flight_blockers).

    `planned_moves`: list of (src, dst) pairs to move (flat -> nested).
    `in_flight_blockers`: `_last.pt` files at the flat root, indicating a
        currently-running job. Migration aborts if any are found.
    """
    moves: list[tuple[Path, Path]] = []
    blockers: list[Path] = []
    if not results_dir.exists():
        return moves, blockers

    for child in sorted(results_dir.iterdir()):
        if not _is_flat_artifact(child):
            continue  # already nested or non-artifact
        stem = child.stem
        if stem.endswith("_last"):
            blockers.append(child)
            continue
        sub = subdir_for(stem)
        if sub == stem and "_fold" not in stem:
            # No `_fold{N}` suffix — leave file at root; new layout puts it in
            # `<stem>/<stem>.<suffix>` per `subdir_for`. Move anyway so the rule
            # is consistent.
            pass
        dst = results_dir / sub / child.name
        if dst == child or dst.exists():
            continue
        moves.append((child, dst))
    return moves, blockers


def run_migration(results_dir: Path, dry_run: bool = True) -> int:
    """Print and execute the migration. Returns exit code (0=ok, 2=blocked, 3=nothing-to-do)."""
    moves, blockers = plan_migration(results_dir)

    if blockers:
        print("[abort] In-flight job detected — refusing to migrate.", file=sys.stderr)
        for b in blockers:
            print(f"  {b}", file=sys.stderr)
        print("Wait for jobs to finish (or remove `_last.pt`) then re-run.", file=sys.stderr)
        return 2

    if not moves:
        print(f"[ok] {results_dir} is already nested — nothing to migrate.")
        return 3

    print(f"[plan] {len(moves)} file(s) to move:")
    for src, dst in moves:
        print(f"  {src.name}  ->  {dst.parent.name}/{dst.name}")

    if dry_run:
        print(f"[dry-run] No changes made. Re-run without --dry-run to execute.")
        return 0

    n_done = 0
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        n_done += 1
    print(f"[done] Moved {n_done} file(s).")
    return 0
