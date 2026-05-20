"""Test-set inference for every saved checkpoint.

Why this exists: `run_baseline` only evaluates on val (in-distribution held-out
frames from the train cameras). The honest comparison is on the LOCO test
cameras held out at the fold level. This module loads each saved .pt and runs
it through `splits[fold]["test"]`, writing one consolidated JSON.

Output schema (`<results_dir>/test_inference.json`):

    {
      "runs": {
        "<run_name>": {
            "model": "resnet50",
            "fold": 0,
            "config_kind": "baseline" | "lds" | "fds" | "lds_fds",
            "n_test": int,
            "test_cams": [...],
            "test": {"overall": float, "many": float, "medium": float, "few": float},
            "test_preds": [...],   # only when with_preds=True (default)
            "test_ys":    [...]
        },
        ...
      },
      "generated_at": "...",
      "n_runs": int,
      "skipped": {"<run_name>": "reason", ...}
    }

CLI: `skyfinder inference [--exclude PATTERN] [--include-vit] [--no-preds]`
"""
from __future__ import annotations

import datetime as _dt
import fnmatch
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from skyfinder.analysis.skymask_inference import _build_model_from_results
from skyfinder.training.dataloader import EVAL_TF, SkyFinderDataset
from skyfinder.training.engine import get_device, per_bin_mae

# Default: skip ViT (per project decision) and the smoke-test runs.
DEFAULT_SKIP_PATTERNS = ("*_vit_*", "save_test", "smoke_*")


def _list_run_jsons(result_dirs: list[Path], pattern: str | None,
                    exclude: str | None, include_vit: bool) -> list[Path]:
    """Return all `<run>_fold<k>.json` files that have a matching `.pt` checkpoint."""
    paths: list[Path] = []
    for d in result_dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*_fold*.json")):
            if not (p.parent / f"{p.stem}.pt").exists():
                continue
            if pattern and not fnmatch.fnmatch(p.stem, pattern):
                continue
            if exclude and fnmatch.fnmatch(p.stem, exclude):
                continue
            if not include_vit and any(fnmatch.fnmatch(p.stem, pat) for pat in DEFAULT_SKIP_PATTERNS):
                continue
            paths.append(p)
    return paths


def _config_kind(cfg: dict) -> str:
    has_lds = bool(cfg.get("use_lds", False))
    has_fds = bool(cfg.get("use_fds", False))
    if has_lds and has_fds: return "lds_fds"
    if has_lds: return "lds"
    if has_fds: return "fds"
    return "baseline"


def _eval_test_one(config: dict, results_path: Path, device: str,
                   batch_size: int, num_workers: int) -> dict | str:
    """Run test inference for one checkpoint. Returns row dict or skip-reason string."""
    labels_path = Path(config["labels_path"])
    splits_path = Path(config["splits_path"])
    img_dir     = Path(config["img_dir"])

    d = json.loads(results_path.read_text())
    cfg = d.get("config", {}) or {}
    fold = cfg.get("fold")
    if fold is None:
        return "no fold in config"

    splits = json.loads(splits_path.read_text())
    if fold >= len(splits):
        return f"fold {fold} out of range"
    fold_info = splits[fold]
    if not fold_info.get("test"):
        return f"fold {fold} has no test split"

    df = pd.read_csv(labels_path)
    test_df = df.iloc[fold_info["test"]].reset_index(drop=True)
    train_y = df.iloc[fold_info["train"]]["TempM"].to_numpy()

    try:
        model, _ = _build_model_from_results(config, results_path)
    except Exception as e:
        return f"model load failed: {e}"
    model.to(device).eval()

    loader = DataLoader(SkyFinderDataset(test_df, EVAL_TF, img_dir=img_dir),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    preds, ys = [], []
    with torch.no_grad():
        for x, y, _w in loader:
            out = model(x.to(device))
            out = out.squeeze(-1) if out.ndim > 1 else out
            preds.append(out.cpu().numpy())
            ys.append(y.numpy())
    preds = np.concatenate(preds)
    ys    = np.concatenate(ys)

    return {
        "model": cfg.get("model"),
        "fold": fold,
        "config_kind": _config_kind(cfg),
        "n_test": int(len(test_df)),
        "test_cams": fold_info.get("test_cams", []),
        "test": per_bin_mae(ys, preds, train_y),
        "_preds": preds.astype(np.float32),
        "_ys":    ys.astype(np.float32),
    }


def _save_atomic(out_path: Path, runs: dict, skipped: dict) -> None:
    out = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_runs": len(runs),
        "runs": runs,
        "skipped": skipped,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)


def run(config: dict, out_path: Path | str | None = None,
        result_dirs: list[Path] | None = None, pattern: str | None = None,
        exclude: str | None = None, include_vit: bool = False,
        with_preds: bool = True, skip_existing: bool = True,
        batch_size: int = 32, num_workers: int = 2) -> dict:
    """Walk every saved checkpoint, run test inference, write consolidated JSON."""
    out_path = Path(out_path) if out_path is not None else Path(config["test_inference_path"])
    result_dirs = [Path(d) for d in (result_dirs or config["aggregate_result_dirs"])]
    device = get_device()
    paths = _list_run_jsons(result_dirs, pattern, exclude, include_vit)

    runs: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    if skip_existing and out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            runs = prev.get("runs", {}) or {}
            skipped = prev.get("skipped", {}) or {}
            print(f"[resume] {out_path} has {len(runs)} runs + {len(skipped)} skipped already")
        except Exception as e:
            print(f"[resume] couldn't read {out_path}: {e}")

    todo = [p for p in paths if p.stem not in runs] if skip_existing else paths
    print(f"[plan] device={device}  {len(paths)} matched  {len(todo)} to run "
          f"({len(paths) - len(todo)} skipped via resume)")

    for i, p in enumerate(todo, 1):
        name = p.stem
        print(f"[{i:3d}/{len(todo)}] {name}")
        result = _eval_test_one(config, p, device, batch_size, num_workers)
        if isinstance(result, str):
            skipped[name] = result
            print(f"  -> SKIP ({result})")
            _save_atomic(out_path, runs, skipped)
            continue
        if with_preds:
            result["test_preds"] = result["_preds"].tolist()
            result["test_ys"]    = result["_ys"].tolist()
        result.pop("_preds"); result.pop("_ys")
        runs[name] = result
        t = result["test"]
        print(f"  overall={t['overall']:.3f}  many={t['many']:.3f}  "
              f"medium={t['medium']:.3f}  few={t['few']:.3f}")
        _save_atomic(out_path, runs, skipped)

    print(f"[saved] {out_path}  ({len(runs)} runs, {len(skipped)} skipped)")
    return {"runs": runs, "skipped": skipped}
