# DIR-SkyFinder

Deep Imbalanced Regression on the [SkyFinder dataset](https://cs.valdosta.edu/~rpmihail/skyfinder/):
predict air temperature (°C) from a single outdoor webcam image.

The setup is image-only regression with two test conditions:
- **Validation**: held-out frames from the same cameras the model trained on.
- **Test (LOCO)**: leave-one-camera-out — held-out *cameras* the model never saw.

DIR's two ideas — **Label Distribution Smoothing (LDS)** and **Feature
Distribution Smoothing (FDS)** — target the long tail of rare temperatures.
This repo tests whether they hold up on SkyFinder, plus ablations for
robustness to noisy labels, range-targeted missingness, and rare-bin drop.

## Install

```bash
git clone <repo>
cd DIR-SkyFinder
python -m pip install -e .            # installs the `skyfinder` CLI
python -m pip install -e .[test]      # adds pytest for the smoke suite
```

`pip install -e .` puts a `skyfinder` console script on your PATH and lets
you edit the package without reinstalling.

## Quickstart

Four commands take you from a fresh clone to figures. Each step is
independent — you can rerun any of them.

```bash
# 1. Prep data: clean labels CSV, download images, build LOCO splits.
python data/prep_labels.py
python data/download_images.py
python data/filter_to_images.py
python data/splits.py

# 2. Train the headline 8-experiment sweep (4 ResNet + 4 ViT configs).
#    Locally:
skyfinder train --family main --skip-smoke
#    On Hyak:
bash ablations/submit.sh main -p gpu-l40s --gpus=1

# 3. Test-set inference (LOCO; resumable; one consolidated JSON).
skyfinder inference

# 4. Aggregate + render every figure that has the data it needs.
skyfinder analyze all
```

Outputs land in three trees:

```
results/                # training JSONs + checkpoints (per-experiment subfolders)
results/_analysis/      # aggregate.csv, C1/C2/D1 JSONs, embeddings/
figures/                # all PDFs/PNGs (with subfolders: diag/, dist/, etc.)
```

## Run a specific experiment

```bash
skyfinder train --family main --experiment baseline_resnet50 --skip-smoke
skyfinder train --list                     # all 19 families
skyfinder train --list main                # experiments in a family
skyfinder train --list main --names-only   # one name per line (SLURM-friendly)
```

## Repository layout

```text
DIR-SkyFinder/
├── skyfinder/                       # main package (pip-installable)
│   ├── cli.py                       # `skyfinder ...` console entry
│   ├── training/
│   │   ├── config.py                # Config dataclass + paths
│   │   ├── dataloader.py            # SkyFinderDataset + transforms + build_loaders
│   │   ├── model.py                 # build_model + FDSModel
│   │   ├── engine.py                # train_one_epoch, predict_split, per_bin_mae
│   │   ├── checkpoint.py            # save/load model + training-state I/O
│   │   ├── trainer.py               # run_baseline orchestration
│   │   ├── families.py              # experiment registry (19 families)
│   │   ├── lds.py                   # Label Distribution Smoothing
│   │   ├── fds.py                   # Feature Distribution Smoothing
│   │   ├── migrate.py               # one-shot flat→nested results migration
│   │   └── diagnostics.py           # convergence diagnostics for saved runs
│   └── analysis/
│       ├── aggregate.py             # JSONs → flat DataFrame
│       ├── baselines_constant.py    # C1: constant predictors (no GPU)
│       ├── baselines_metadata.py    # C2: HistGradientBoostingRegressor
│       ├── skymask_inference.py     # D1: sky-mask inference
│       ├── linear_probe.py          # D4: linear-probe delta summary
│       ├── extract_embeddings.py    # penultimate features (val + test)
│       ├── extract_trajectory.py    # per-epoch features (snapshot_every>0)
│       ├── corrupt_labels.py        # F-family train-label corruption
│       ├── dist.py                  # distribution-shift analysis
│       ├── test_inference.py        # test-set MAE for every checkpoint
│       ├── style.py                 # Nature-style matplotlib helpers
│       └── figures/                 # all publication figures (subpackage)
│           ├── main_sweep.py        # fig_main_sweep, scatter, dist_and_errbar
│           ├── curves.py            # fig_training_curves (reusable)
│           ├── ablations.py         # A1-A5, D1, D4, E1, F1-F5
│           ├── embeddings.py        # fig_embed_temp/cam/knn/per_bin/cka, by_bin
│           └── trajectory.py        # fig_traj_pca/per_bin/knn/cka
├── configs/                         # all YAML configs
│   ├── main.yaml                    # 8-experiment headline sweep
│   ├── dir_hyperparams.yaml         # A1-A5 hyperparameter robustness
│   ├── ablations.yaml               # D4, E1, F1-F5
│   ├── ablations_decomp.yaml        # F2/F5 LDS-only / FDS-only splits
│   └── analysis.yaml                # paths for the analysis pipeline
├── data/                            # labels CSV, splits JSON, images (gitignored)
├── results/                         # training outputs (gitignored)
├── figures/                         # rendered figures (gitignored)
├── ablations/
│   ├── run_family.slurm             # generic SLURM array job
│   └── submit.sh                    # auto-sized submission wrapper
├── experiments/                     # scratch experiments (cam_conditioned, test_diagnosis)
├── scripts/                         # one-off utility scripts (explore_dataset.py)
├── tests/                           # pytest smoke + regression suite
├── docs/                            # detailed docs, historical context
├── pyproject.toml
└── README.md
```

## Method

**Models:** ResNet-50 (ImageNet V2 weights) and ViT-B/16 (ImageNet V1 weights),
both with a 1-output regression head, L1 loss, Adam + cosine schedule.

**Training variants:**
- `baseline` — vanilla L1 regression.
- `lds` — Label Distribution Smoothing (loss-side, reweights by smoothed target density).
- `fds` — Feature Distribution Smoothing (architecture-side feature calibration).
- `lds_fds` — both DIR components.

**Splits:** 5-fold leave-one-camera-out (LOCO). Val cameras = train cameras
(val held out by row, not camera).

**Per-bin MAE:** test errors classified by training-set frequency in 2°C bins:
`many` (≥100 train samples), `medium` (20–100), `few` (<20). The "few" column
is the headline DIR metric.

## Ablations (one command each)

```bash
skyfinder train --family lds_sigma          # A1: LDS kernel sigma sweep
skyfinder train --family lds_reweight       # A2: LDS reweight scheme
skyfinder train --family bin_width          # A3: bucket width
skyfinder train --family fds_momentum       # A4: FDS momentum
skyfinder train --family fds_start_smooth   # A5: FDS start_smooth epoch
skyfinder train --family linear_probe       # D4: linear probe vs full fine-tune
skyfinder train --family seed_variance      # E1: 3 seeds on the headline config
skyfinder train --family corrupt_random     # F1: random label corruption
skyfinder train --family corrupt_range_drop # F2: range MNAR (drop rows)
skyfinder train --family corrupt_rare_bin   # F3: rare-bin drop
skyfinder train --family corrupt_noise      # F5: Gaussian noise on labels
```

`bash ablations/submit.sh <family> [sbatch flags]` runs any family as a SLURM array.
`skyfinder train --list` shows the full registry.

## Tests

```bash
pytest tests/         # 22 smoke + regression tests, ~7 seconds
```

Imports + CLI help + migration script + aggregate-dataframe parser are all
covered. Pre-push check.

## Migrating older runs

If you have results from before the May 2026 restructure (flat layout:
`results/<run>.pt` next to `results/<run>.json`), one command upgrades them
into the nested per-experiment subfolders the new code expects:

```bash
skyfinder data-prep --migrate-results results/ --dry-run    # plan
skyfinder data-prep --migrate-results results/              # execute
```

The script aborts if it sees a `_last.pt` at the flat root (in-flight job
indicator) — wait for the job to finish before migrating.

## References

- Deep Imbalanced Regression: https://github.com/YyzHarry/imbalanced-regression
- SkyFinder dataset: https://cs.valdosta.edu/~rpmihail/skyfinder/
