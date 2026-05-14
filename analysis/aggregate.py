"""Aggregate run JSONs into a single flat DataFrame.

Reads from `results/` (main sweep + tune) and `ablations/results/` (ablations).
Group + config_kind are inferred from the run `name` prefix; see `_classify`.

Schema of the returned DataFrame (one row per JSON):

  source            "results" | "ablations/results"
  name              run_name (matches the file stem)
  group             "main" | "A1" | ... | "F5" | "C1" | "C2" | "D1"
  model             "resnet50" | "vit_b_16"
  config_kind       "baseline" | "lds" | "fds" | "lds_fds"
  fold              int
  seed              int
  freeze_backbone   bool
  corruption_mode   str | None  (e.g., "random", "range", "rare_bin", "noise")
  corruption_param  float | None  (rate / range-midpoint / drop_prob / noise_std)
  lds_sigma         float | None
  fds_sigma         float | None
  bin_width         float | None
  fds_momentum      float | None
  fds_start_smooth  float | None
  lds_reweight      str | None
  val_overall, val_many, val_medium, val_few    (from final_val)
  test_overall, test_many, test_medium, test_few (from final_test if present)

C1, C2, D1 results have a `per_fold` array instead of a single final_val; one
row per fold gets emitted for those.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml


RESULT_DIRS = [Path("results"), Path("ablations/results")]
ABLATION_CONFIGS = [Path("ablations/config_ab.yaml"), Path("ablations/config_ab2.yaml")]


# ============================================================
# Group classification from run name
# ============================================================

_GROUP_RULES = [
    (re.compile(r"^tune_lds_sigma_"),         "A1"),
    (re.compile(r"^tune_lds_reweight_"),      "A2"),
    (re.compile(r"^tune_bucket_"),            "A3"),
    (re.compile(r"^tune_fds_momentum_"),      "A4"),
    (re.compile(r"^tune_fds_start_smooth_"),  "A5"),
    (re.compile(r"^d4_"),                     "D4"),
    (re.compile(r"^e1_"),                     "E1"),
    (re.compile(r"^f1_"),                     "F1"),
    (re.compile(r"^f2_.*_impute_"),           "F2_impute"),
    (re.compile(r"^f2_.*_drop_"),             "F2_drop"),
    (re.compile(r"^f3_"),                     "F3"),
    (re.compile(r"^f5_"),                     "F5"),
    (re.compile(r"^c1_"),                     "C1"),
    (re.compile(r"^c2_"),                     "C2"),
    (re.compile(r"^d1_"),                     "D1"),
    # Main sweep falls through (baseline/lds/fds/lds_fds with model suffix).
    (re.compile(r"^(baseline|lds|fds|lds_fds)_(resnet50|vit_b_16)"), "main"),
]


def _classify(name: str) -> str:
    for rx, group in _GROUP_RULES:
        if rx.match(name):
            return group
    return "unknown"


def _config_kind(cfg: dict) -> str:
    has_lds = bool(cfg.get("use_lds", False))
    has_fds = bool(cfg.get("use_fds", False))
    if has_lds and has_fds:
        return "lds_fds"
    if has_lds:
        return "lds"
    if has_fds:
        return "fds"
    return "baseline"


def _corruption_param(corruption: dict | None) -> tuple[str | None, float | None]:
    if not corruption:
        return None, None
    mode = corruption.get("mode")
    if mode == "random":
        return mode, float(corruption.get("rate", 0.0))
    if mode == "range":
        lo, hi = corruption.get("range", [0, 0])
        return mode, (float(lo) + float(hi)) / 2.0   # midpoint
    if mode == "rare_bin":
        return mode, float(corruption.get("drop_prob", 0.0))
    if mode == "noise":
        return mode, float(corruption.get("noise_std", 0.0))
    return mode, None


def _run_name_from_template(exp: dict, fold: int) -> str:
    return exp["run_name_template"].format(
        name=exp.get("name"),
        model=exp.get("model"),
        fold=fold,
        epochs=exp.get("epochs"),
    )


def _load_config_lookup(config_paths: list[Path] = ABLATION_CONFIGS) -> dict[str, dict]:
    """Map saved run_name -> experiment config from the ablation YAML files."""
    lookup: dict[str, dict] = {}
    for path in config_paths:
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text()) or {}
        for exp in data.get("experiments", []):
            if "run_name_template" not in exp:
                continue
            for fold in exp.get("folds", []):
                lookup[_run_name_from_template(exp, fold)] = exp
    return lookup


CONFIG_LOOKUP = _load_config_lookup()


def _fill_cfg_from_yaml(name: str, cfg: dict) -> dict:
    """JSON config wins, but YAML fills fields that older JSONs saved as null/missing."""
    fallback = CONFIG_LOOKUP.get(name)
    if not fallback:
        return cfg
    merged = fallback.copy()
    merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


# ============================================================
# JSON parsers
# ============================================================

def _parse_standard(path: Path, source: str) -> list[dict]:
    """Standard training-run JSON: one row per file."""
    d = json.loads(path.read_text())
    name = d.get("run_name", path.stem)
    cfg = _fill_cfg_from_yaml(name, d.get("config", {}) or {})
    val = d.get("final_val", {}) or {}
    test = d.get("final_test", {}) or {}   # not currently produced by baseline.py
    cmode, cparam = _corruption_param(cfg.get("corruption"))
    return [{
        "source": source,
        "name": name,
        "group": _classify(name),
        "model": cfg.get("model"),
        "config_kind": _config_kind(cfg),
        "fold": cfg.get("fold"),
        "seed": cfg.get("seed"),
        "freeze_backbone": bool(cfg.get("freeze_backbone", False)),
        "corruption_mode": cmode,
        "corruption_param": cparam,
        "lds_sigma": cfg.get("lds_sigma"),
        "fds_sigma": cfg.get("fds_sigma"),
        "bin_width": cfg.get("bin_width"),
        "fds_momentum": cfg.get("fds_momentum"),
        "fds_start_smooth": cfg.get("fds_start_smooth"),
        "lds_reweight": cfg.get("lds_reweight"),
        "val_overall": val.get("overall"),
        "val_many": val.get("many"),
        "val_medium": val.get("medium"),
        "val_few": val.get("few"),
        "test_overall": test.get("overall"),
        "test_many": test.get("many"),
        "test_medium": test.get("medium"),
        "test_few": test.get("few"),
    }]


def _parse_per_fold(path: Path, source: str, name_prefix: str) -> list[dict]:
    """C1 / C2 / D1: outer dict has `per_fold` list; emit one row per fold entry."""
    d = json.loads(path.read_text())
    rows = []
    for r in d.get("per_fold", []):
        val = r.get("val", {}) or {}
        test = r.get("test", {}) or {}
        # C1 / D1 may carry a sub-name (e.g. predictor kind or config_kind).
        sub = r.get("predictor") or r.get("config_kind") or ""
        full = f"{name_prefix}/{sub}" if sub else name_prefix
        rows.append({
            "source": source,
            "name": full,
            "group": _classify(name_prefix),
            "model": r.get("model"),
            "config_kind": r.get("config_kind"),
            "fold": r.get("fold"),
            "seed": None,
            "freeze_backbone": False,
            "corruption_mode": None,
            "corruption_param": None,
            "lds_sigma": None, "fds_sigma": None, "bin_width": None,
            "fds_momentum": None, "fds_start_smooth": None, "lds_reweight": None,
            "val_overall": val.get("overall"),
            "val_many":    val.get("many"),
            "val_medium":  val.get("medium"),
            "val_few":     val.get("few"),
            "test_overall": test.get("overall"),
            "test_many":    test.get("many"),
            "test_medium":  test.get("medium"),
            "test_few":     test.get("few"),
        })
    return rows


def _parse_one(path: Path) -> list[dict]:
    source = path.parent.name
    name = path.stem
    if name.startswith(("c1_", "c2_", "d1_")):
        return _parse_per_fold(path, source, name)
    if "per_fold" in path.read_text()[:200]:   # cheap sniff
        return _parse_per_fold(path, source, name)
    return _parse_standard(path, source)


# ============================================================
# Public API
# ============================================================

def build_dataframe(result_dirs: list[Path] | None = None) -> pd.DataFrame:
    """Glob all `*.json` in result_dirs, return one flat df."""
    dirs = result_dirs if result_dirs is not None else RESULT_DIRS
    rows = []
    for d in dirs:
        if not d.exists():
            continue
        for path in sorted(d.glob("*.json")):
            try:
                rows.extend(_parse_one(path))
            except Exception as e:
                print(f"[skip] {path}: {e}")
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, out: Path | str = "ablations/results/aggregate.csv") -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[saved] {out}  ({len(df)} rows)")
    return out


def main():
    df = build_dataframe()
    print(df.groupby("group").size().to_string())
    save_csv(df)


if __name__ == "__main__":
    main()
