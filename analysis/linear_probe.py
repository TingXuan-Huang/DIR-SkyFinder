"""D4 linear-probe analysis: compare frozen-backbone vs full-fine-tune on the
4 DIR configurations (baseline / LDS / FDS / LDS+FDS) at fold 0.

Reads the aggregate DataFrame (built by `analysis/aggregate.py` from the run JSONs
+ `test_inference.json`) and computes:
  - per-config delta = full_fine_tune_MAE - linear_probe_MAE
  - relative drop (%): what fraction of the full-FT performance the linear probe recovers

Output: `<results_dir>/d4_linprobe_summary.json` (alongside the run JSONs).
Figure: handled by `analysis/figures.fig_d4_linprobe(config, df)`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


KIND_ORDER = ("baseline", "lds", "fds", "lds_fds")
BIN_COLS = ("test_overall", "test_many", "test_medium", "test_few")


def _filter(df: pd.DataFrame, group: str, fold: int = 0) -> pd.DataFrame:
    return df[(df["group"] == group) & (df["model"] == "resnet50") & (df["fold"] == fold)]


def run_linear_probe_summary(config: dict, out_path: Path | str | None = None,
                             fold: int = 0) -> dict:
    """Compute D4 delta vs full FT per config_kind. Returns the summary dict."""
    from analysis.aggregate import build_dataframe
    df = build_dataframe(config)

    main = _filter(df, "main", fold)
    d4   = _filter(df, "D4", fold)

    if d4.empty:
        print("[skip] no D4 rows in aggregate — run linear probe experiments first")
        return {"per_kind": [], "n_main": int(len(main)), "n_d4": 0}

    per_kind: list[dict] = []
    for kind in KIND_ORDER:
        m = main[main["config_kind"] == kind]
        l = d4[d4["config_kind"]   == kind]
        if l.empty:
            continue
        row: dict = {"config_kind": kind}
        for col in BIN_COLS:
            mv = float(m[col].mean()) if not m.empty else float("nan")
            lv = float(l[col].mean())
            row[col] = {
                "full_ft":    mv,
                "linprobe":   lv,
                "delta":      lv - mv,                       # positive = linprobe worse
                "rel_drop":   (lv - mv) / mv if mv > 0 else float("nan"),
            }
        per_kind.append(row)
        if "test_overall" in row:
            print(f"[{kind:>10s}] full FT={row['test_overall']['full_ft']:.3f}  "
                  f"linprobe={row['test_overall']['linprobe']:.3f}  "
                  f"Δ={row['test_overall']['delta']:+.3f}")

    out = {
        "fold": fold,
        "n_main": int(len(main)),
        "n_d4": int(len(d4)),
        "per_kind": per_kind,
    }
    out_path = Path(out_path) if out_path is not None else \
        Path(config["ablation_results_dir"]) / "d4_linprobe_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[saved] {out_path}")
    return out
