"""Run SkyFinder baseline experiments from config.yaml.

This is the script version of notebooks/colab_baseline.ipynb for Hyak:
it binds paths, runs the smoke test, then runs ResNet-50 and ViT-B/16
over the configured folds.

Run from project root:
    python run.py --config config.yaml
    python run.py --config config.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import dir_skyfinder.baseline as baseline


RUN_KEYS = (
    # Core training
    "model",
    "fold",
    "epochs",
    "batch_size",
    "lr",
    "num_workers",
    "train_subset",
    "val_subset",
    "seed",
    "run_name",
    # LDS (loss-side reweighting)
    "use_lds",
    "lds_kernel",
    "lds_ks",
    "lds_sigma",
    "lds_reweight",
    # FDS (feature calibration)
    "use_fds",
    "fds_kernel",
    "fds_ks",
    "fds_sigma",
    "fds_momentum",
    "fds_start_smooth",
    # Shared (LDS+FDS bucketing)
    "bin_width",
    # Ablation hooks (see ablations/ablation.py)
    "freeze_backbone",
    "corruption",
)


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def bind_baseline_paths(cfg: dict) -> None:
    root = resolve_path(Path.cwd(), cfg.get("project_root", ".")).resolve()
    paths = cfg["paths"]

    baseline.PROJ = root
    baseline.DATA = root / "data"
    baseline.LABELS = resolve_path(root, paths["labels"])
    baseline.SPLITS = resolve_path(root, paths["splits"])
    baseline.IMG_DIR = resolve_path(root, paths["images"])
    baseline.RESULTS_DIR = resolve_path(root, paths["results"])
    baseline.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in [baseline.LABELS, baseline.SPLITS, baseline.IMG_DIR]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[paths]")
    print(f"  project: {baseline.PROJ}")
    print(f"  labels:  {baseline.LABELS}")
    print(f"  splits:  {baseline.SPLITS}")
    print(f"  images:  {baseline.IMG_DIR}")
    print(f"  results: {baseline.RESULTS_DIR}")


def split_count() -> int:
    return len(json.loads(baseline.SPLITS.read_text()))


def completed(run_name: str) -> bool:
    return (baseline.RESULTS_DIR / f"{run_name}.json").exists()


def run_one(spec: dict, save: bool, skip_existing: bool, dry_run: bool) -> None:
    kwargs = {k: spec[k] for k in RUN_KEYS if k in spec}
    run_name = kwargs["run_name"]

    if skip_existing and completed(run_name):
        print(f"[skip] {run_name} already has a results JSON")
        return

    print(f"[run] {run_name}")
    if dry_run:
        print(f"      {kwargs}")
        return
    baseline.run_baseline(save=save, **kwargs)


def expand_experiment(exp: dict) -> list[dict]:
    folds = exp["folds"]
    specs = []
    for fold in folds:
        spec = {k: exp[k] for k in RUN_KEYS if k in exp}
        spec["fold"] = fold
        spec["run_name"] = exp["run_name_template"].format(
            name=exp["name"],
            model=exp["model"],
            fold=fold,
            epochs=exp["epochs"],
        )
        specs.append(spec)
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--experiment", type=str, default=None,
                    help="Run only the experiment with this `name`. Used by SLURM array jobs.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bind_baseline_paths(cfg)
    n_splits = split_count()
    print(f"[env] device={baseline.get_device()}  splits={n_splits}")

    save = bool(cfg.get("save", True))
    skip_existing = bool(cfg.get("skip_existing", True))

    smoke = cfg.get("smoke_test", {})
    if smoke.get("enabled", False) and not args.skip_smoke:
        run_one(smoke, save=save, skip_existing=skip_existing, dry_run=args.dry_run)

    experiments = cfg["experiments"]
    if args.experiment is not None:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            raise ValueError(f"--experiment {args.experiment!r} not found in {args.config}")
        print(f"[filter] running only experiment: {args.experiment}")

    for exp in experiments:
        for spec in expand_experiment(exp):
            if spec["fold"] >= n_splits:
                raise ValueError(f"fold {spec['fold']} requested, but {baseline.SPLITS} has {n_splits} folds")
            run_one(spec, save=save, skip_existing=skip_existing, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
