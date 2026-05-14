#!/usr/bin/env bash
# One-time data setup for Hyak (or any new machine).
#
# Runs on the login node — downloads the SkyFinder dataset, builds the cleaned
# label tables and the LOCO splits. After this finishes, the GPU SLURM jobs
# can run without any network access.
#
# Idempotent: each step skips if its output already exists, so re-running
# after an interruption resumes where it stopped.
#
# Usage:  bash prepare_data.sh
#
# Expected wall time: ~30-60 min (dominated by the 81k image downloads,
# which the academic host rate-limits to a few concurrent connections).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${CONDA_ENV_PREFIX:-${PROJECT_DIR}/.conda/skyfinder}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

cd "${PROJECT_DIR}"

# --- conda env (mirrors run.sh) ---
if command -v module >/dev/null 2>&1; then
    module load conda
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available. On Hyak login nodes, try: module load conda" >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
if [ ! -d "${ENV_PREFIX}" ]; then
    conda create --yes --prefix "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_PREFIX}"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# --- 1) Metadata CSV (~92 MB, ~30 sec) ---
echo ""
echo "=== [1/5] metadata CSV ==="
python -c "
from urllib.request import urlretrieve
from pathlib import Path
csv = Path('data/complete_table_with_mcr.csv')
if csv.exists():
    print(f'[skip] {csv} ({csv.stat().st_size:,} bytes)')
else:
    Path('data').mkdir(exist_ok=True)
    print(f'[download] {csv}...')
    urlretrieve('https://cs.valdosta.edu/~rpmihail/skyfinder/analysis/complete_table_with_mcr.csv', csv)
    print(f'[done] {csv} ({csv.stat().st_size:,} bytes)')
"

# --- 2) Cleaned labels (drop NaN / -9999 / -999 TempM rows) ---
echo ""
echo "=== [2/5] prep_labels.py -> data/labels.parquet ==="
python data/prep_labels.py

# --- 3) Images (~3 GB, ~30-60 min; resumable via .part-then-rename + skip-if-exists) ---
echo ""
echo "=== [3/5] download_images.py -> data/images/<CamId>/*.jpg ==="
python data/download_images.py

# --- 4) Filter labels to rows whose JPEG actually landed on disk ---
echo ""
echo "=== [4/5] filter_to_images.py -> data/labels_with_images.csv ==="
python data/filter_to_images.py

# --- 5) 5-fold leave-one-camera-out splits ---
echo ""
echo "=== [5/5] splits.py -> data/splits/loco_5fold.json ==="
python data/splits.py

echo ""
echo "[prepare] done. Sanity checks:"
ls -lh data/labels_with_images.csv data/splits/loco_5fold.json 2>&1 || true
echo "  image dirs: $(ls data/images | wc -l | tr -d ' ')  (expected 53; ~47 non-empty)"
echo ""
echo "Next step: sbatch run.slurm"
