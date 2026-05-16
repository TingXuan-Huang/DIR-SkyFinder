#!/usr/bin/env bash
# Run the analysis pipeline end-to-end in an interactive Hyak session.
#
# Typical use:
#     srun -p gpu-2080ti --gpus=1 --mem=64G --time=2:00:00 --pty bash
#     cd /path/to/DIR_Code
#     bash analysis.sh
#
# Each step is gated by `|| true` so D1 (which needs sky masks) or embeddings
# (which need trained checkpoints) won't block the CPU-only steps that follow.
# Re-run safely: outputs go to `ablations/results/` and `figures/`; existing
# files are overwritten.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${CONDA_ENV_PREFIX:-${PROJECT_DIR}/.conda/skyfinder}"

cd "${PROJECT_DIR}"

# --- env activation (mirrors run.sh; assumes env exists) ---
if command -v module >/dev/null 2>&1; then
    module load conda
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not available. On Hyak: module load conda" >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
if [ ! -d "${ENV_PREFIX}" ]; then
    echo "conda env not found at ${ENV_PREFIX}" >&2
    echo "create it first with: bash run.sh config.yaml --help" >&2
    exit 1
fi
conda activate "${ENV_PREFIX}"

# --- cache dirs match run_ab.slurm so we don't re-download torch/HF artifacts ---
export SCRATCH="${SCRATCH:-/mmfs1/gscratch/stf/${USER}}"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export TORCH_HOME="${SCRATCH}/.cache/torch"
export HF_HOME="${SCRATCH}/.cache/huggingface"
mkdir -p "${TORCH_HOME}" "${HF_HOME}"

banner() { printf '\n============================================================\n[%s] %s\n============================================================\n' "$(date +%H:%M:%S)" "$*"; }

# --- 0. Install / refresh deps (sklearn + umap-learn were added recently) ---
banner "Install/refresh Python deps (requirements.txt)"
python -m pip install -r requirements.txt || echo "[warn] pip install failed"

# --- 1. C1: constant predictors (CPU, ~10 s) ---
banner "C1: constant predictors (CPU)"
python analysis.py c1 || echo "[warn] c1 failed"

# --- 2. C2: metadata-only HistGBR (CPU, ~5 min) ---
banner "C2: metadata-only HistGBR (CPU)"
python analysis.py c2 || echo "[warn] c2 failed"

# --- 3a. Download sky masks for fold 0 val cameras (network, ~30 s) ---
banner "Download sky masks (fold 0 val cams) -> data/masks/"
python data/download_masks.py --fold 0 || echo "[warn] mask download failed"

# --- 3b. D1: sky-mask inference (GPU, ~5 min) ---
banner "D1: sky-mask inference (GPU)"
python analysis.py d1 || echo "[warn] d1 failed"

# --- 4. Embeddings: extract 2048-d penultimate features (GPU, ~5 min) ---
banner "Embeddings: extract penultimate features (GPU; needs checkpoints)"
python analysis.py embeddings || echo "[warn] embeddings failed (likely missing checkpoints in results/)"

# --- 4c. Test-set inference: run every saved checkpoint on the LOCO test split ---
banner "Inference: test-set MAE for every checkpoint (GPU) -> results/test_inference.json"
python inference.py || echo "[warn] inference failed (likely missing checkpoints)"

# --- 5. Aggregate: build flat CSV from all result JSONs (CPU, ~1 s) ---
banner "Aggregate: build aggregate.csv"
python analysis.py aggregate || echo "[warn] aggregate failed"

# --- 6. Figures: render all Nature-style PDFs + PNGs (CPU, ~30 s) ---
banner "Figures: render all (figures/)"
python analysis.py figures || echo "[warn] figures failed"

banner "Done. Outputs:"
echo "  ablations/results/c1_constants.json"
echo "  ablations/results/c2_metadata_only.json"
echo "  ablations/results/d1_skymask.json           (if masks present)"
echo "  ablations/results/embeddings/*.npz          (if checkpoints present)"
echo "  results/test_inference.json                 (test MAE for every saved checkpoint)"
echo "  ablations/results/aggregate.csv"
echo "  figures/*.{pdf,png}"
