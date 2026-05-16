"""Run SkyFinder DIR experiments by family or by individual experiment name.

Three invocation modes:
  1. By family (registry below maps family -> YAML + name-glob):
         python run.py --family linear_probe
         python run.py --family corrupt_random
         python run.py --family corrupt_random --experiment f1_rate25_baseline
  2. By YAML directly (back-compat with pre-registry callers):
         python run.py --config config.yaml
         python run.py --config ablations/config_ab.yaml --experiment d4_linprobe_resnet
  3. Discovery:
         python run.py --list                          # all families with descriptions
         python run.py --list label_corruption         # experiments in one family
         python run.py --count label_corruption        # number of experiments (for SLURM array)
         python run.py --list <family> --names-only    # one name per line (SLURM-friendly)

The training-loop entry point is `dir_skyfinder.trainer.run_baseline(cfg)`; this
script is the thin orchestrator on top of it.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import yaml

import dir_skyfinder.baseline as baseline


# ============================================================
# Registry: family -> (yaml config, name-glob pattern, description)
# ============================================================

FAMILIES: dict[str, dict] = {
    # ---------- Headline ----------
    "main": {
        "config": "config.yaml",
        "pattern": None,
        "desc": "Main sweep: baseline / LDS / FDS / LDS+FDS x 5 folds (ResNet + ViT)",
    },

    # ---------- DIR hyperparameter robustness (A1-A5) ----------
    "lds_sigma": {
        "config": "config_tune.yaml",
        "pattern": "tune_lds_sigma_*",
        "desc": "LDS sigma sweep (A1)",
    },
    "lds_reweight": {
        "config": "config_tune.yaml",
        "pattern": "tune_lds_reweight_*",
        "desc": "LDS reweight scheme (A2)",
    },
    "bin_width": {
        "config": "config_tune.yaml",
        "pattern": "tune_bucket_*",
        "desc": "Bucket width sweep (A3)",
    },
    "fds_momentum": {
        "config": "config_tune.yaml",
        "pattern": "tune_fds_momentum_*",
        "desc": "FDS momentum sweep (A4)",
    },
    "fds_start_smooth": {
        "config": "config_tune.yaml",
        "pattern": "tune_fds_start_smooth_*",
        "desc": "FDS start_smooth sweep (A5)",
    },
    "dir_hyperparams": {
        "config": "config_tune.yaml",
        "pattern": None,
        "desc": "All DIR hyperparams (A1-A5)",
    },

    # ---------- Diagnostic ----------
    "linear_probe": {
        "config": "ablations/config_ab.yaml",
        "pattern": "d4_*",
        "desc": "Linear probe vs full fine-tune (D4)",
    },
    "seed_variance": {
        "config": "ablations/config_ab.yaml",
        "pattern": "e1_*",
        "desc": "Seed variance on headline DIR config (E1)",
    },

    # ---------- Label corruption robustness (F1-F5) ----------
    "corrupt_random": {
        "config": "ablations/config_ab.yaml",
        "pattern": "f1_*",
        "desc": "Random label corruption + impute (F1)",
    },
    "corrupt_range_impute": {
        "config": "ablations/config_ab.yaml",
        "pattern": "f2_*_impute_*",
        "desc": "Range MNAR + impute (F2)",
    },
    "corrupt_range_drop": {
        "config": "ablations/config_ab.yaml",
        "pattern": "f2_*_drop_*",
        "desc": "Range MNAR + drop rows (F2)",
    },
    "corrupt_rare_bin": {
        "config": "ablations/config_ab.yaml",
        "pattern": "f3_*",
        "desc": "Rare-bin amplification (F3)",
    },
    "corrupt_noise": {
        "config": "ablations/config_ab.yaml",
        "pattern": "f5_*",
        "desc": "Gaussian noise on all labels (F5)",
    },
    "label_corruption": {
        "config": "ablations/config_ab.yaml",
        "pattern": "f[1-5]_*",
        "desc": "All label corruption (F1-F5)",
    },

    # ---------- Mechanism decomposition (LDS-only / FDS-only) ----------
    "corrupt_range_impute_decomp": {
        "config": "ablations/config_ab2.yaml",
        "pattern": "f2_*_impute_*",
        "desc": "F2 impute LDS/FDS singletons",
    },
    "corrupt_range_drop_decomp": {
        "config": "ablations/config_ab2.yaml",
        "pattern": "f2_*_drop_*",
        "desc": "F2 drop LDS/FDS singletons",
    },
    "corrupt_noise_decomp": {
        "config": "ablations/config_ab2.yaml",
        "pattern": "f5_*",
        "desc": "F5 LDS/FDS singletons",
    },
    "mechanism_decomp": {
        "config": "ablations/config_ab2.yaml",
        "pattern": None,
        "desc": "All F2+F5 LDS/FDS mechanism splits",
    },
}


RUN_KEYS = (
    "model", "fold", "epochs", "batch_size", "lr", "num_workers",
    "train_subset", "val_subset", "seed", "run_name",
    "use_lds", "lds_kernel", "lds_ks", "lds_sigma", "lds_reweight",
    "use_fds", "fds_kernel", "fds_ks", "fds_sigma", "fds_momentum", "fds_start_smooth",
    "bin_width",
    "freeze_backbone", "corruption",       # F-family + D4 hooks
    "snapshot_every",                      # trajectory hook
)


# ============================================================
# YAML + path utilities
# ============================================================

def load_yaml(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def bind_baseline_paths(cfg: dict) -> None:
    """Mutate `dir_skyfinder.baseline` module-level paths from the YAML's `paths:` block."""
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


# ============================================================
# Experiment expansion
# ============================================================

def filter_experiments(experiments: list[dict], pattern: str | None,
                       experiment_name: str | None) -> list[dict]:
    out = list(experiments)
    if pattern is not None:
        out = [e for e in out if fnmatch.fnmatch(e["name"], pattern)]
    if experiment_name is not None:
        out = [e for e in out if e["name"] == experiment_name]
    return out


def expand_experiment(exp: dict) -> list[dict]:
    """One experiment block + N folds -> N specs (one per fold), each with a run_name."""
    specs = []
    for fold in exp["folds"]:
        spec = {k: exp[k] for k in RUN_KEYS if k in exp}
        spec["fold"] = fold
        spec["run_name"] = exp["run_name_template"].format(
            name=exp["name"], model=exp["model"], fold=fold, epochs=exp["epochs"],
        )
        specs.append(spec)
    return specs


def run_one(spec: dict, save: bool, skip_existing: bool, dry_run: bool) -> None:
    from dir_skyfinder.trainer import run_baseline
    kwargs = {k: spec[k] for k in RUN_KEYS if k in spec}
    run_name = kwargs["run_name"]

    if skip_existing and completed(run_name):
        print(f"[skip] {run_name} already has a results JSON")
        return
    print(f"[run] {run_name}")
    if dry_run:
        print(f"      {kwargs}")
        return
    run_baseline(save=save, **kwargs)


# ============================================================
# Discovery (`--list`, `--count`, `--names-only`)
# ============================================================

def _enumerate_family(family: str) -> tuple[Path, list[str]]:
    """Returns (yaml_path, list_of_experiment_names) for one family."""
    entry = FAMILIES[family]
    yaml_path = Path(entry["config"])
    if not yaml_path.exists():
        return yaml_path, []
    data = load_yaml(yaml_path) or {}
    names = [e["name"] for e in data.get("experiments", [])]
    if entry.get("pattern"):
        names = [n for n in names if fnmatch.fnmatch(n, entry["pattern"])]
    return yaml_path, names


def cmd_list(family: str | None, names_only: bool) -> None:
    if family is None:
        # All families.
        if names_only:
            for fam in FAMILIES:
                print(fam)
            return
        width = max(len(f) for f in FAMILIES)
        print(f"{'family'.ljust(width)}  count  description")
        print("-" * (width + 25))
        for fam, entry in FAMILIES.items():
            _, names = _enumerate_family(fam)
            print(f"{fam.ljust(width)}  {len(names):>5d}  {entry['desc']}")
        return

    if family not in FAMILIES:
        print(f"unknown family: {family!r}", file=sys.stderr)
        print(f"known families: {', '.join(FAMILIES)}", file=sys.stderr)
        sys.exit(2)

    yaml_path, names = _enumerate_family(family)
    if names_only:
        for n in names:
            print(n)
        return
    print(f"# family={family}  config={yaml_path}  ({len(names)} experiments)")
    print(f"# {FAMILIES[family]['desc']}")
    for n in names:
        print(n)


def cmd_count(family: str) -> None:
    if family not in FAMILIES:
        print(f"unknown family: {family!r}", file=sys.stderr); sys.exit(2)
    _, names = _enumerate_family(family)
    print(len(names))


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", type=str, default=None,
                    help="Run a family from the registry. See `--list`.")
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config (overrides --family's default). Required if --family not given.")
    ap.add_argument("--experiment", type=str, default=None,
                    help="Run only this experiment name (filters inside family/config).")
    ap.add_argument("--list", nargs="?", const="__ALL__", default=None,
                    help="List all families, or experiments inside one family.")
    ap.add_argument("--count", type=str, default=None,
                    help="Print the number of experiments in a family (for SLURM --array).")
    ap.add_argument("--names-only", action="store_true",
                    help="With --list: print one name per line (SLURM-friendly).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    # --- Discovery modes (don't load data) ---
    if args.list is not None:
        cmd_list(None if args.list == "__ALL__" else args.list, args.names_only)
        return
    if args.count is not None:
        cmd_count(args.count); return

    # --- Resolve config + filter ---
    pattern = None
    config_path = args.config
    if args.family is not None:
        if args.family not in FAMILIES:
            print(f"unknown family: {args.family!r}", file=sys.stderr)
            print(f"known families: {', '.join(FAMILIES)}", file=sys.stderr)
            sys.exit(2)
        entry = FAMILIES[args.family]
        config_path = config_path or Path(entry["config"])
        pattern = entry.get("pattern")
    if config_path is None:
        print("either --family or --config is required", file=sys.stderr); sys.exit(2)

    cfg = load_yaml(config_path)
    bind_baseline_paths(cfg)
    n_splits = split_count()
    print(f"[env] device={baseline.get_device()}  splits={n_splits}")

    save = bool(cfg.get("save", True))
    skip_existing = bool(cfg.get("skip_existing", True))

    smoke = cfg.get("smoke_test", {})
    if smoke.get("enabled", False) and not args.skip_smoke:
        run_one(smoke, save=save, skip_existing=skip_existing, dry_run=args.dry_run)

    experiments = filter_experiments(cfg["experiments"], pattern, args.experiment)
    if args.family or args.experiment:
        print(f"[filter] family={args.family} experiment={args.experiment} "
              f"-> {len(experiments)} match(es)")
    if not experiments:
        msg = f"no experiments match family={args.family!r} experiment={args.experiment!r}"
        raise ValueError(msg)

    for exp in experiments:
        for spec in expand_experiment(exp):
            if spec["fold"] >= n_splits:
                raise ValueError(f"fold {spec['fold']} requested, but {baseline.SPLITS} has {n_splits} folds")
            run_one(spec, save=save, skip_existing=skip_existing, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
