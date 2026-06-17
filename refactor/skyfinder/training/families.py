"""Experiment-family registry + YAML loading + expansion helpers.

A "family" is a named subset of experiments across one or more YAML configs.
The CLI's `skyfinder train --family X` looks up X here, loads the YAML, and
runs the matching experiments.

To add a new family: append one line to `FAMILIES` and (if needed) a new YAML
under `configs/`. No SLURM or CLI changes required.
"""
from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path

import yaml

from . import config as cfg_module
from .checkpoint import subdir_for

# ============================================================
# Registry: family -> (YAML config, name-glob pattern, description)
# ============================================================

FAMILIES: dict[str, dict] = {
    # ---------- Headline ----------
    "main": {
        "config": "configs/main.yaml",
        "pattern": None,
        "desc": "Main sweep: baseline / LDS / FDS / LDS+FDS x 5 folds (ResNet + ViT)",
    },

    # ---------- DIR hyperparameter robustness (A1-A5) ----------
    "lds_sigma": {
        "config": "configs/dir_hyperparams.yaml",
        "pattern": "tune_lds_sigma_*",
        "desc": "LDS sigma sweep (A1)",
    },
    "lds_reweight": {
        "config": "configs/dir_hyperparams.yaml",
        "pattern": "tune_lds_reweight_*",
        "desc": "LDS reweight scheme (A2)",
    },
    "bin_width": {
        "config": "configs/dir_hyperparams.yaml",
        "pattern": "tune_bucket_*",
        "desc": "Bucket width sweep (A3)",
    },
    "fds_momentum": {
        "config": "configs/dir_hyperparams.yaml",
        "pattern": "tune_fds_momentum_*",
        "desc": "FDS momentum sweep (A4)",
    },
    "fds_start_smooth": {
        "config": "configs/dir_hyperparams.yaml",
        "pattern": "tune_fds_start_smooth_*",
        "desc": "FDS start_smooth sweep (A5)",
    },
    "dir_hyperparams": {
        "config": "configs/dir_hyperparams.yaml",
        "pattern": None,
        "desc": "All DIR hyperparams (A1-A5)",
    },

    # ---------- Diagnostic ----------
    "linear_probe": {
        "config": "configs/ablations.yaml",
        "pattern": "d4_*",
        "desc": "Linear probe vs full fine-tune (D4)",
    },
    "seed_variance": {
        "config": "configs/ablations.yaml",
        "pattern": "e1_*",
        "desc": "Seed variance on headline DIR config (E1)",
    },

    # ---------- Label corruption robustness (F1-F5) ----------
    "corrupt_random": {
        "config": "configs/ablations.yaml",
        "pattern": "f1_*",
        "desc": "Random label corruption + impute (F1)",
    },
    "corrupt_range_impute": {
        "config": "configs/ablations.yaml",
        "pattern": "f2_*_impute_*",
        "desc": "Range MNAR + impute (F2)",
    },
    "corrupt_range_drop": {
        "config": "configs/ablations.yaml",
        "pattern": "f2_*_drop_*",
        "desc": "Range MNAR + drop rows (F2)",
    },
    "corrupt_rare_bin": {
        "config": "configs/ablations.yaml",
        "pattern": "f3_*",
        "desc": "Rare-bin amplification (F3)",
    },
    "corrupt_noise": {
        "config": "configs/ablations.yaml",
        "pattern": "f5_*",
        "desc": "Gaussian noise on all labels (F5)",
    },
    "label_corruption": {
        "config": "configs/ablations.yaml",
        "pattern": "f[1-5]_*",
        "desc": "All label corruption (F1-F5)",
    },

    # ---------- Mechanism decomposition (LDS-only / FDS-only) ----------
    "corrupt_range_impute_decomp": {
        "config": "configs/ablations_decomp.yaml",
        "pattern": "f2_*_impute_*",
        "desc": "F2 impute LDS/FDS singletons",
    },
    "corrupt_range_drop_decomp": {
        "config": "configs/ablations_decomp.yaml",
        "pattern": "f2_*_drop_*",
        "desc": "F2 drop LDS/FDS singletons",
    },
    "corrupt_noise_decomp": {
        "config": "configs/ablations_decomp.yaml",
        "pattern": "f5_*",
        "desc": "F5 LDS/FDS singletons",
    },
    "mechanism_decomp": {
        "config": "configs/ablations_decomp.yaml",
        "pattern": None,
        "desc": "All F2+F5 LDS/FDS mechanism splits",
    },
}


# Keys read from a YAML experiment block and passed as kwargs to `run_baseline`.
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
# YAML loading + path binding
# ============================================================

def load_yaml(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def bind_paths(yaml_cfg: dict) -> None:
    """Mutate `skyfinder.training.config` module-level paths from the YAML's `paths:` block."""
    root = resolve_path(Path.cwd(), yaml_cfg.get("project_root", ".")).resolve()
    paths = yaml_cfg["paths"]

    cfg_module.PROJ = root
    cfg_module.DATA = root / "data"
    cfg_module.LABELS = resolve_path(root, paths["labels"])
    cfg_module.SPLITS = resolve_path(root, paths["splits"])
    cfg_module.IMG_DIR = resolve_path(root, paths["images"])
    cfg_module.RESULTS_DIR = resolve_path(root, paths["results"])
    cfg_module.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in [cfg_module.LABELS, cfg_module.SPLITS, cfg_module.IMG_DIR]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[paths]")
    print(f"  project: {cfg_module.PROJ}")
    print(f"  labels:  {cfg_module.LABELS}")
    print(f"  splits:  {cfg_module.SPLITS}")
    print(f"  images:  {cfg_module.IMG_DIR}")
    print(f"  results: {cfg_module.RESULTS_DIR}")


def split_count() -> int:
    return len(json.loads(cfg_module.SPLITS.read_text()))


def completed(run_name: str) -> bool:
    """True iff a results JSON exists for this run under the nested layout.

    (Dual-path lookup was removed in the May 2026 refactor; for old flat-layout
    runs, migrate first with `skyfinder data-prep --migrate-results`.)
    """
    path = cfg_module.RESULTS_DIR / subdir_for(run_name) / f"{run_name}.json"
    return path.exists()


# ============================================================
# Experiment expansion + filtering
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
    """One YAML experiment block + N folds -> N specs (one per fold) with `run_name`."""
    specs = []
    for fold in exp["folds"]:
        spec = {k: exp[k] for k in RUN_KEYS if k in exp}
        spec["fold"] = fold
        spec["run_name"] = exp["run_name_template"].format(
            name=exp["name"], model=exp["model"], fold=fold, epochs=exp["epochs"],
        )
        specs.append(spec)
    return specs


# ============================================================
# Discovery (used by `skyfinder train --list / --count`)
# ============================================================

def enumerate_family(family: str) -> tuple[Path, list[str]]:
    """Returns (yaml_path, list_of_experiment_names) for one family."""
    if family not in FAMILIES:
        raise KeyError(family)
    entry = FAMILIES[family]
    yaml_path = Path(entry["config"])
    if not yaml_path.exists():
        return yaml_path, []
    data = load_yaml(yaml_path) or {}
    names = [e["name"] for e in data.get("experiments", [])]
    if entry.get("pattern"):
        names = [n for n in names if fnmatch.fnmatch(n, entry["pattern"])]
    return yaml_path, names


def print_list(family: str | None, names_only: bool) -> None:
    if family is None:
        if names_only:
            for fam in FAMILIES:
                print(fam)
            return
        width = max(len(f) for f in FAMILIES)
        print(f"{'family'.ljust(width)}  count  description")
        print("-" * (width + 25))
        for fam, entry in FAMILIES.items():
            try:
                _, names = enumerate_family(fam)
            except KeyError:
                names = []
            print(f"{fam.ljust(width)}  {len(names):>5d}  {entry['desc']}")
        return

    if family not in FAMILIES:
        print(f"unknown family: {family!r}", file=sys.stderr)
        print(f"known families: {', '.join(FAMILIES)}", file=sys.stderr)
        sys.exit(2)

    yaml_path, names = enumerate_family(family)
    if names_only:
        for n in names:
            print(n)
        return
    print(f"# family={family}  config={yaml_path}  ({len(names)} experiments)")
    print(f"# {FAMILIES[family]['desc']}")
    for n in names:
        print(n)


def print_count(family: str) -> None:
    if family not in FAMILIES:
        print(f"unknown family: {family!r}", file=sys.stderr)
        sys.exit(2)
    _, names = enumerate_family(family)
    print(len(names))
