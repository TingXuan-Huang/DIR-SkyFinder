# Restart workspace — 2026-05-24

**Objective.** Rebuild deep understanding of DIR-SkyFinder before continuing experiments.
The existing `skyfinder/` package and `results/` artifacts stay in place until per-module
triage decides otherwise.

**Mode.** `experiment-helper` Phase 1 → Phase 2 → Phase 4 (Phase 3 deferred until 1+2+4
are understood). `/self-study` gap-spotter throughout. Hybrid restart (option 1+2 from the
scope question): walk through each existing module Socratically; keep if well-designed and
explainable from recall, rewrite if not. Code only via `/experiment-helper-code`, one
function at a time.

**Current phase:** 4 (analysis). Re-interpreting the archived Round-1 sweep (`data/old_outputs/
server_results/`) at the corrected `bin_w=1.0`. Training-package refactor complete (11 modules,
end-to-end smoke green). Phase 1 §6 unknowns drafted; U3 resolved.

**Iteration:** v1.

**Files in this workspace:**
- `REFACTOR_PLAN.md` — restart plan; source of truth for *how* we work this iteration.
- `phase1-data.md` — Phase 1 data understanding (Q1.5 unknowns still open).
- `phase2-design.md` — Phase 2 framing + module triage table (§8 tracks keep/rewrite verdicts).
- `phase4-analysis.md` — created when Phase 4 begins.

**Refactored code (outside this workspace):** `refactor/skyfinder/` at repo root — the parallel
rebuild target that replaces `skyfinder/` once triage completes. Smoke tests live at `refactor/*.py`.

**Status log:**
- 2026-05-24: workspace created; `REFACTOR_PLAN.md` drafted; Phase 1 batch 1 of 5 questions
  awaiting user reply.
- 2026-05-24 (later): Phase 1 batch 1 answered. Q1.1-Q1.4 filled in `phase1-data.md`
  from code (`dataloader.py`, `config.py`) and figures (`temperature_hist.png`,
  `pairs_with_temp.png`). Q1.5 (three unknowns) reserved for user. Data-prep module
  triage deferred to batch 2.
- 2026-05-24 (Phase 2): framing locked as a DIR transfer-study vs IMDB-WIKI-DIR + AgeDB-DIR.
  `phase2-design.md` written with paper reference numbers (fetched from ar5iv arXiv:2102.09554),
  improvement-rate decision rule, 10-condition matrix, risks. F3 (surprise) deferred.
- 2026-05-24 (Phase 2, first code): 10 conditions confirmed = 8-cell + C1 + C2. Triaged
  `dataloader.py` → verdict DONE via Option 2 (Config-threading). Parallel refactor package
  created at repo-root `refactor/skyfinder/training/` (`config.py` + `dataloader.py` + minimal
  `__init__.py`s). `build_loaders(cfg)` smoke-tested green (`refactor/smoke_dataloader.py`).
  Promotion plan: `mv refactor/skyfinder ./skyfinder` once whole triage done.
- 2026-05-24 (Phase 2, core triage complete): triaged all 6 core training modules into refactor/
  — config(edit), dataloader(rewrite), lds(keep), fds(keep), model(keep import-tweak),
  engine(keep import-tweak). All silent-failure traps (LDS w, FDS labels/feats) VERIFIED wired by
  inspection. Each module smoke-tested from the refactor package.
- 2026-05-24 (**KEY FINDING**, U3 resolved): `measure_bins.py` shows the DIR few-shot bin is
  essentially empty on SkyFinder — test few ≈ 220 (0.27%) at 1 °C, 132 at 2 °C, some folds ZERO.
  The tail is too sparse to measure DIR's signature few-shot improvement. Decisions: eval
  `bin_w → 1.0 °C` (user-confirmed, refactor only); reframe headline metric to lead with Overall +
  Medium-shot (few = underpowered, report with n + caveat). Partial answer to F3.
- 2026-05-24 (**MILESTONE** — training package fully refactored): copied checkpoint.py + trainer.py
  (with the deferred `build_loaders(cfg)` collapse) into refactor/; a background subagent did the 3
  peripheral support modules (families/diagnostics/migrate). End-to-end `run_baseline` smoke GREEN
  (`refactor/smoke_train.py`): baseline (16.424) ≠ LDS+FDS (16.037) on identical tiny data → DIR
  interventions empirically active. All 11 training modules now in refactor/ and runnable. Phase 2
  module triage of `skyfinder/training/` COMPLETE. Remaining: `skyfinder/analysis/` (Phase 4) +
  the deferred Phase-1 data-prep code triage.
- 2026-05-24 (Phase 4 started): found the archived Round-1 sweep at `data/old_outputs/server_results/`
  (raw val_preds/val_ys per fold + C1/C2 + fold-0 test). Recomputed VAL per-bin MAE at bin_w=1.0
  (`recompute_val_bins.py`), verified faithful (matches stored final_val to 0.0000 + report F4).
  KEY: LDS trades body for tail — medium −27%, few −39%, but overall +11% / many +11% (WORSE).
  FDS ≈ baseline. Transfer vs paper is sign-flipped on overall. Plus report's F1 (C2 metadata beats
  all CNNs on LOCO test) → real story is distribution shift. `phase4-analysis.md` written. Next
  (all local, no retrain): recompute TEST at bin_w=1.0, αC2+CNN ensemble, paired bootstrap.
