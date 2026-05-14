# DIR-SkyFinder: Temperature Prediction from Outdoor Webcam Images

This project adapts **Deep Imbalanced Regression (DIR)** to the **SkyFinder**
outdoor webcam dataset. The task is simple to state and surprisingly hard:

> Given one outdoor webcam image, predict the air temperature in degrees Celsius.

The core question is whether DIR's two imbalance-aware techniques, **Label
Distribution Smoothing (LDS)** and **Feature Distribution Smoothing (FDS)**, can
improve predictions on rare temperature ranges such as very cold and very hot
weather.

## Why This Is Interesting

SkyFinder is a static-camera dataset: many images come from the same webcam over
time. A naive random split can leak camera identity and make the model look
better than it really is. This project instead uses **leave-one-camera-out
splits**, so test cameras are unseen during training.

The target distribution is also highly imbalanced. Most images are in ordinary
temperate ranges, while extremes are rare. A vanilla CNN trained with L1 loss
can do well on common temperatures and still fail badly on the tails. DIR is
designed for exactly this kind of continuous-label imbalance.

## Dataset

The project uses the [SkyFinder dataset](https://cs.valdosta.edu/~rpmihail/skyfinder/):

- ~90k outdoor webcam images from 53 static AMOS cameras
- per-frame weather metadata in `complete_table_with_mcr.csv`
- `TempM` is used as the target temperature in degrees Celsius
- images are loaded as `data/images/<CamId>/<Filename>`
- trainable subset after filtering missing labels/images: about 81k images from 47 cameras

The code does **image-only** prediction. Metadata such as camera ID, hour, month,
latitude, and longitude is used for splitting/analysis, not as model input.

## Method

Models:

- ResNet-50, ImageNet-pretrained
- ViT-B/16, ImageNet-pretrained

Training variants:

- `baseline`: vanilla L1 regression
- `lds`: Label Distribution Smoothing, loss-side reweighting by smoothed target density
- `fds`: Feature Distribution Smoothing, feature calibration before the regression head
- `lds_fds`: both DIR components together

Evaluation:

- 5-fold leave-one-camera-out split
- validation MAE overall
- validation MAE by target-frequency bins: `many`, `medium`, `few`
- ablations for hyperparameters, corruption robustness, sky masks, and embedding diagnostics

## Repository Layout

```text
DIR_Code/
├── dir_skyfinder/
│   ├── baseline.py          # main trainer: ResNet/ViT, LDS, FDS, checkpoints
│   ├── lds.py               # LDS weights and weighted L1
│   ├── fds.py               # FDS module
│   └── utils.py             # saved-run diagnostics
├── data/
│   ├── prep_labels.py       # raw CSV -> cleaned labels
│   ├── download_images.py   # resumable SkyFinder image downloader
│   ├── download_masks.py    # sky-mask downloader
│   ├── filter_to_images.py  # keep rows with local JPEGs
│   ├── splits.py            # 5-fold leave-one-camera-out splits
│   └── splits/loco_5fold.json
├── analysis/
│   ├── aggregate.py         # JSON results -> aggregate CSV
│   ├── figures.py           # publication-style figures
│   ├── c1.py                # constant-predictor baselines
│   ├── d1.py                # sky-mask inference ablation
│   └── embeddings.py        # feature extraction diagnostics
├── ablations/
│   ├── config_ab.yaml       # ablation sweep config
│   ├── config_ab2.yaml      # companion F2/F5 ablations
│   ├── run_ab.slurm
│   └── run_ab2.slurm
├── config.yaml              # headline 5-fold sweep
├── run.py                   # YAML-driven experiment runner
├── run.sh                   # conda setup + run.py wrapper
├── run.slurm                # Hyak Slurm array job
├── analysis.py              # analysis CLI
└── requirements.txt
```

Large files such as `data/images/`, checkpoints, and generated figures should
not be committed to Git.

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare data:

```bash
python data/prep_labels.py
python data/download_images.py
python data/filter_to_images.py
python data/splits.py
```

Run a dry-run check:

```bash
python run.py --config config.yaml --dry-run
```

Run one experiment locally or on an interactive GPU node:

```bash
python run.py --config config.yaml --experiment baseline_resnet50 --skip-smoke
```

Submit the full configured sweep on Hyak:

```bash
sbatch run.slurm
```

Results are written to:

```text
results/<run_name>.json
results/<run_name>.pt
```

The JSON stores metrics, history, and predictions. The `.pt` file stores the
best-validation checkpoint for that run.

## Analysis

Aggregate result JSON files:

```bash
python analysis.py aggregate
```

Render figures:

```bash
python analysis.py figures
```

Run the lightweight analysis pipeline:

```bash
python analysis.py all
```

Available analysis subcommands:

```bash
python analysis.py c1
python analysis.py d1
python analysis.py embeddings
python analysis.py aggregate
python analysis.py figures
python analysis.py all
```

Figures are saved under:

```text
figures/
```

## Ablations

The ablation suite tests whether the DIR improvements are robust or fragile:

- LDS kernel sigma
- LDS reweighting rule
- temperature bucket width
- FDS momentum and smoothing start epoch
- linear-probe vs full fine-tuning
- seed variance
- training-label corruption
- range-targeted missingness
- sky-mask inference
- embedding-space diagnostics

Submit ablation arrays with:

```bash
sbatch ablations/run_ab.slurm
sbatch ablations/run_ab2.slurm
```

## Notes

This is research code, optimized for iteration rather than packaging. Most
settings live in YAML files or simple module-level constants so experiments can
be changed quickly.

The main implementation intentionally stays close to the DIR paper's setup:
image backbone, scalar regression head, LDS as a loss-side intervention, and
FDS as a feature-side calibration layer.

## References

- Deep Imbalanced Regression: https://github.com/YyzHarry/imbalanced-regression
- SkyFinder dataset: https://cs.valdosta.edu/~rpmihail/skyfinder/
