"""Training subpackage (refactor) — COMPLETE. All modules triaged in from the old package.

    config.py      — Config dataclass + path defaults            (EDIT: path fields)
    dataloader.py  — SkyFinderDataset, build_loaders(cfg)        (REWRITE: Config-threading)
    lds.py         — compute_lds_weights, weighted_l1_loss       (KEEP, verbatim)
    fds.py         — FDS module, calibrate_mean_var              (KEEP, verbatim)
    model.py       — build_model, FDSModel                       (KEEP, import-tweak)
    engine.py      — train_one_epoch, predict_split, per_bin_mae (KEEP, tweak; bin_w=1.0)
    checkpoint.py  — save/load weights/state/results             (KEEP, import-tweak)
    trainer.py     — run_baseline orchestration                  (KEEP, tweak + build_loaders(cfg))
    families.py    — experiment-family registry                  (KEEP, subagent)
    diagnostics.py — convergence check                           (KEEP, identical)
    migrate.py     — flat->nested results migration              (KEEP, subagent)

End-to-end smoke: refactor/smoke_train.py (run_baseline green: baseline + LDS+FDS).
"""
from __future__ import annotations
