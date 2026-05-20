"""Extract embedding trajectory across training epochs.

Requires training to have been done with `Config.snapshot_every > 0` so that
`<run_name>_ep{N}.pt` files exist alongside the main `.pt`.

For each (run, epoch_snapshot, split): builds the model from the run's JSON,
loads weights from `<run>_ep{N}.pt`, extracts penultimate features, saves a
`.npz` under `<trajectory_dir>` (from config).

All paths come from `config`.
"""
from __future__ import annotations

import re
from pathlib import Path

from skyfinder.training.checkpoint import find_artifact
from skyfinder.analysis.extract_embeddings import extract_one


DEFAULT_RUNS = (
    "baseline_resnet50", "lds_resnet50", "fds_resnet50", "lds_fds_resnet50",
    "baseline_vit",      "lds_vit",      "fds_vit",      "lds_fds_vit",
)
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_TRAIN_SUBSAMPLE = 5000   # keep PCA/UMAP/k-NN tractable across many epochs


_EP_RE = re.compile(r"_ep(\d+)\.pt$")


def find_snapshot_epochs(config: dict, run_name: str, fold: int) -> list[int]:
    """Discover all `<run>_fold{fold}_ep{N}.pt` under `config["results_dir"]`. Returns sorted N's.

    rglob to handle both flat and per-experiment-subfolder layouts.
    """
    results_dir = Path(config["results_dir"])
    eps = []
    for p in results_dir.rglob(f"{run_name}_fold{fold}_ep*.pt"):
        m = _EP_RE.search(p.name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(set(eps))


def run_trajectory(config: dict, out_dir: Path | str | None = None, fold: int = 0,
                   runs: tuple[str, ...] = DEFAULT_RUNS,
                   splits: tuple[str, ...] = DEFAULT_SPLITS,
                   train_subsample: int | None = DEFAULT_TRAIN_SUBSAMPLE,
                   batch_size: int = 32, num_workers: int = 2) -> list[Path]:
    """For each (run, ep, split), extract features and save .npz. Resume-friendly:
    if the output file already exists, the inference is skipped.
    """
    out_dir = Path(out_dir) if out_dir is not None else Path(config["trajectory_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config["results_dir"])

    saved: list[Path] = []
    for run in runs:
        eps = find_snapshot_epochs(config, run, fold)
        if not eps:
            print(f"[skip] no snapshot .pt files matching {run}_fold{fold}_ep*.pt")
            continue
        print(f"[run] {run}  snapshots={eps}")
        for ep in eps:
            ckpt_override = find_artifact(f"{run}_fold{fold}_ep{ep}", ".pt", results_dir)
            if ckpt_override is None:
                print(f"  [skip] missing ckpt for ep={ep}"); continue
            for split in splits:
                target = out_dir / f"{run}_fold{fold}_ep{ep}_{split}.npz"
                if target.exists():
                    print(f"  [skip] {target.name} exists")
                    saved.append(target)
                    continue
                subsample = train_subsample if split == "train" else None
                p = extract_one(
                    config=config, run_name=run, fold=fold, out_dir=out_dir, split=split,
                    ckpt_override=ckpt_override, epoch_tag=ep, subsample=subsample,
                    batch_size=batch_size, num_workers=num_workers,
                )
                if p is not None:
                    saved.append(p)
    return saved
