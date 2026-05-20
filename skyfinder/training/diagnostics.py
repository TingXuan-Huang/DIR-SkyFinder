"""Utility functions for inspecting saved baseline runs.

Currently exposes `convergence_diagnostic`, which reads a results JSON
written by `baseline.run_baseline` and reports four convergence signals.

Usage:
    from skyfinder.training.diagnostics import convergence_diagnostic
    convergence_diagnostic("results/baseline_resnet50_fold0.json")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def convergence_diagnostic(results_path, verbose: bool = True) -> dict:
    """Read a saved run JSON and report whether training converged.

    Runs four checks on the per-epoch history:
      1. Train MAE drop in the last 5 pre-LR-tail epochs — is the model still learning?
      2. Val MAE plateau — last 5-epoch mean vs the 5 before that.
      3. Epochs since `best_epoch` — is "best" the last epoch (under-trained) or several back (safe)?
      4. Val/train ratio at the final epoch — is the model overfitting?

    Returns a dict with each signal's raw value, qualitative label, and an overall verdict.
    If `verbose`, prints a human-readable summary.

    Notes
    -----
    - Cosine LR schedule mechanically flattens train_mae in the final 1-2 epochs (lr ≈ 0).
      Signal 1 excludes the last 2 epochs from the "last 5" window for that reason.
    - Verdicts are heuristic — review the four individual signals if anything looks off.
    """
    data = json.loads(Path(results_path).read_text())
    history = data["history"]
    n = len(history)

    if n < 10:
        msg = f"[diagnostic] only {n} epochs in history — need ≥ 10 for a full check"
        if verbose:
            print(msg)
        return {"verdict": "history too short", "n_epochs": n}

    train = np.array([x["train_mae"] for x in history])
    val = np.array([x["val_mae"] for x in history])

    # --- Signal 1: train MAE drop in last 5 pre-LR-tail epochs ---
    tail_skip = 2
    seg = train[n - 5 - tail_skip : n - tail_skip]
    drops_per_epoch = -np.diff(seg) / seg[:-1] * 100  # +ve = still improving
    train_drop_pct = float(np.mean(drops_per_epoch))
    if abs(train_drop_pct) < 0.5:
        signal1 = "converged"
    elif train_drop_pct > 2:
        signal1 = "still learning"
    else:
        signal1 = "approaching"

    # --- Signal 2: val MAE plateau ---
    val_recent = float(np.mean(val[-5:]))
    val_prev = float(np.mean(val[-10:-5]))
    val_shift_pct = float((val_prev - val_recent) / val_prev * 100)
    signal2 = "plateau" if abs(val_shift_pct) < 3 else "still moving"

    # --- Signal 3: epochs since best ---
    best_epoch = int(data["best_epoch"])
    gap = n - 1 - best_epoch
    if gap >= 5:
        signal3 = "safely converged"
    elif gap == 0:
        signal3 = "best is last epoch — likely needs more"
    else:
        signal3 = "borderline"

    # --- Signal 4: overfit ratio ---
    ratio = float(val[-1] / train[-1])
    if ratio < 1.5:
        signal4 = "healthy"
    elif ratio > 2.0:
        signal4 = "overfitting"
    else:
        signal4 = "borderline"

    # --- Overall verdict (check overfitting before "needs more epochs",
    # because classic overfitting has train_mae still dropping while val
    # has been rising — the right action is FEWER epochs, not more) ---
    if signal4 == "overfitting" and signal3 == "safely converged":
        verdict = "overfitting (best checkpoint is fine; just trained too long)"
    elif signal1 == "still learning" or signal3 == "best is last epoch — likely needs more":
        verdict = "needs more epochs"
    elif signal1 == "converged" and signal2 == "plateau" and signal3 == "safely converged":
        verdict = "converged"
    else:
        verdict = "borderline — review individual signals"

    out = {
        "run_name": data.get("run_name"),
        "n_epochs": n,
        "best_epoch": best_epoch,
        "best_val_mae": data.get("best_val_mae"),
        "train_drop_pct_per_epoch": round(train_drop_pct, 3),
        "val_shift_pct": round(val_shift_pct, 3),
        "epochs_since_best": gap,
        "val_train_ratio": round(ratio, 3),
        "signal1_train": signal1,
        "signal2_val": signal2,
        "signal3_best": signal3,
        "signal4_overfit": signal4,
        "verdict": verdict,
    }

    if verbose:
        best_mae = data.get("best_val_mae")
        best_mae_str = f"{best_mae:.3f}" if best_mae is not None else "—"
        print(f"=== convergence diagnostic: {data.get('run_name')} ===")
        print(f"  epochs: {n}    best: {best_epoch}    best_val_mae: {best_mae_str}")
        print(f"  [1] train MAE drop, last 5 pre-tail eps:  {train_drop_pct:+6.2f}%/ep   → {signal1}")
        print(f"  [2] val plateau, last 5 vs prev 5 mean:   {val_shift_pct:+6.2f}%      → {signal2}")
        print(f"  [3] epochs since best epoch:              {gap:6d}         → {signal3}")
        print(f"  [4] val/train ratio at end:               {ratio:6.2f}         → {signal4}")
        print(f"  VERDICT: {verdict}")

    return out
