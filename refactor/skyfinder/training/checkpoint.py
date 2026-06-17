"""Save and load model weights, full training state, and results JSON.

Nested layout (single source of truth, no fallback):
    <results_dir>/<subdir>/<run_name>.{json,pt}
    <results_dir>/<subdir>/<run_name>_last.pt     # full training state (deleted on success)
    <results_dir>/<subdir>/<run_name>_ep{N}.pt    # per-epoch snapshots

`<subdir>` is derived from `run_name` by stripping `_fold{N}` (and `_ep{N}`).
The old dual-path fallback (flat root + nested) was removed in the May 2026 refactor.
For flat-layout artifacts produced by older code, run `skyfinder data-prep --migrate-results`.

Function naming reflects what's saved:
    save_model_weights / load_model_weights — state_dict only
    save_training_state / load_training_state — model + optimizer + scheduler + RNG + history
    save_results / load_results — JSON: config + history + per-bin MAE + raw val preds
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import RESULTS_DIR


def subdir_for(run_name: str) -> str:
    """Map a run_name to its per-experiment subfolder.

      'baseline_resnet50_fold0'        -> 'baseline_resnet50'
      'baseline_resnet50_fold0_ep5'    -> 'baseline_resnet50'
      'smoke_hyak'                     -> 'smoke_hyak'      (no fold suffix -> name as-is)
    """
    return run_name.split("_fold")[0]


def _resolved_path(run_name: str, suffix: str, results_dir: Path) -> Path:
    """Return the canonical nested path for `<run_name><suffix>`."""
    return results_dir / subdir_for(run_name) / f"{run_name}{suffix}"


def find_artifact(run_name: str, suffix: str, results_dir: Path) -> Path | None:
    """Return the nested-layout path for `<run_name><suffix>` if it exists.

    Replaces the old dual-path `_resolve_load_path` (May 2026 refactor). For
    flat-layout artifacts from before that refactor, run `skyfinder data-prep
    --migrate-results` first.
    """
    path = _resolved_path(run_name, suffix, results_dir)
    return path if path.exists() else None


# ============================================================
# Results JSON (config + history + metrics + val preds)
# ============================================================

def save_results(results: dict, results_dir: Path | None = None) -> Path:
    base = results_dir if results_dir is not None else RESULTS_DIR
    out_dir = base / subdir_for(results["run_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{results['run_name']}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"[saved] {path}")
    return path


def load_results(path) -> dict:
    return json.loads(Path(path).read_text())


# ============================================================
# Model weights only (state_dict)
# ============================================================

def save_model_weights(state_dict, run_name: str, results_dir: Path | None = None) -> Path:
    base = results_dir if results_dir is not None else RESULTS_DIR
    out_dir = base / subdir_for(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}.pt"
    torch.save(state_dict, path)
    print(f"[saved] {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_model_weights(run_name_or_path, results_dir: Path | None = None, map_location="cpu"):
    """Load a saved state_dict by run_name or full path.

    Example:
        from skyfinder.training import model, checkpoint
        sd = checkpoint.load_model_weights("baseline_resnet50_fold0")
        net = model.build_model("resnet50")
        net.load_state_dict(sd)
    """
    p = Path(run_name_or_path)
    if p.exists():
        return torch.load(p, map_location=map_location, weights_only=True)
    base = results_dir if results_dir is not None else RESULTS_DIR
    name = str(run_name_or_path)
    stem = name[:-3] if name.endswith(".pt") else name
    path = _resolved_path(stem, ".pt", base)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint for {run_name_or_path!r} under {base}")
    return torch.load(path, map_location=map_location, weights_only=True)


# ============================================================
# Full training state (model + optimizer + scheduler + RNG + history)
# Atomic via .part-then-rename so Hyak preemption never leaves a half-written file.
# ============================================================

def save_training_state(state: dict, run_name: str, results_dir: Path | None = None) -> Path:
    base = results_dir if results_dir is not None else RESULTS_DIR
    out_dir = base / subdir_for(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{run_name}_last.pt"
    tmp = final.with_name(final.name + ".part")
    torch.save(state, tmp)
    tmp.rename(final)
    return final


def load_training_state(run_name: str, results_dir: Path | None = None) -> dict | None:
    """Load full training state. Returns None if no `_last.pt` exists."""
    base = results_dir if results_dir is not None else RESULTS_DIR
    path = _resolved_path(run_name, "_last.pt", base)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)
