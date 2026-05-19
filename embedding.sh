#!/usr/bin/env bash
# embedding.sh — extract penultimate features and render every embedding figure.
#
# Two phases:
#   1. Single-snapshot embeddings  -> per-(run, split) .npz files
#                                     -> 5 figures: pca/umap × temp/cam, k-NN MAE,
#                                                   per-bin cosine, CKA across configs
#   2. Trajectory embeddings       -> per-(run, ep, split) .npz files for every
#                                     `_ep{N}.pt` snapshot the run produced
#                                     -> 5 trajectory figure types per run:
#                                          PCA × temp, PCA × cam, per-bin cosine,
#                                          CKA across epochs, k-NN MAE vs epoch
#
# Requires (per run you want to plot):
#   - results/<run>/<run>_fold{FOLD}.pt              (always)
#   - results/<run>/<run>_fold{FOLD}_ep{N}.pt        (for trajectory; produced when
#                                                     training had Config.snapshot_every > 0)
#
# Usage (interactive Hyak session, single GPU is enough):
#     srun -p gpu-2080ti --gpus=1 --mem=64G --time=4:00:00 --pty bash
#     cd /path/to/DIR-SkyFinder
#     bash embedding.sh
#
# Customize via env vars (override on the command line):
#     RUNS="lds_fds_resnet50"           # default: 4 ResNet headline configs
#     FOLD=0                            # default: 0
#     SPLITS="val test"                 # default: val test  (add "train" for full coverage)
#     TRAIN_SUBSAMPLE=5000              # default: 5000  (only matters if SPLITS includes train)
#     SKIP_SINGLE_SNAP=0                # 1 = skip phase 1 (only do trajectory)
#     SKIP_TRAJECTORY=0                 # 1 = skip phase 2 (only do single-snapshot)

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${CONDA_ENV_PREFIX:-${PROJECT_DIR}/.conda/skyfinder}"

cd "${PROJECT_DIR}"

# --- env activation (mirrors analysis.sh / run.sh) ---
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

# --- cache dirs match run_family.slurm so we don't re-download torch/HF artifacts ---
export SCRATCH="${SCRATCH:-/mmfs1/gscratch/stf/${USER}}"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export TORCH_HOME="${SCRATCH}/.cache/torch"
export HF_HOME="${SCRATCH}/.cache/huggingface"
mkdir -p "${TORCH_HOME}" "${HF_HOME}"

banner() { printf '\n============================================================\n[%s] %s\n============================================================\n' "$(date +%H:%M:%S)" "$*"; }

# --- parameter defaults ---
RUNS="${RUNS:-baseline_resnet50 lds_resnet50 fds_resnet50 lds_fds_resnet50}"
FOLD="${FOLD:-0}"
SPLITS="${SPLITS:-val test}"
TRAIN_SUBSAMPLE="${TRAIN_SUBSAMPLE:-5000}"
SKIP_SINGLE_SNAP="${SKIP_SINGLE_SNAP:-0}"
SKIP_TRAJECTORY="${SKIP_TRAJECTORY:-0}"

echo "[plan]  runs           = ${RUNS}"
echo "[plan]  fold           = ${FOLD}"
echo "[plan]  splits         = ${SPLITS}"
echo "[plan]  train_subsample= ${TRAIN_SUBSAMPLE}"
echo "[plan]  single_snap    = $([ "${SKIP_SINGLE_SNAP}" = "0" ] && echo on || echo off)"
echo "[plan]  trajectory     = $([ "${SKIP_TRAJECTORY}" = "0" ] && echo on || echo off)"

# --- 1. Single-snapshot embeddings (post-training .pt only) ---
# Produces ablations/results/embeddings/<run>_fold{FOLD}_<split>.npz for each (run, split).
# Then the single-snapshot embedding figures populate from those.
if [ "${SKIP_SINGLE_SNAP}" = "0" ]; then
    banner "1) Single-snapshot embeddings  (val + test)"
    python analysis.py embeddings --fold "${FOLD}" || echo "[warn] single-snapshot embeddings failed"
fi

# --- 2. Per-epoch trajectory embeddings (each _ep{N}.pt snapshot) + figures ---
# Produces ablations/results/embeddings/trajectory/<run>_fold{FOLD}_ep{N}_<split>.npz.
# --with-figures triggers make_trajectory which renders:
#   fig_traj_pca_<run>_fold{F}_<split>_temp.{pdf,png}    PCA grid colored by true temp
#   fig_traj_pca_<run>_fold{F}_<split>_cam.{pdf,png}     PCA grid colored by camera
#   fig_traj_per_bin_<run>_fold{F}_<split>.{pdf,png}     per-bin cosine sim, per epoch
#   fig_traj_cka_<run>_fold{F}_<split>.{pdf,png}         pairwise CKA across epochs
#   fig_traj_knn_<run>_fold{F}.{pdf,png}                 k-NN MAE vs epoch (3 splits)
if [ "${SKIP_TRAJECTORY}" = "0" ]; then
    banner "2) Trajectory embeddings  (per snapshot epoch)"
    python analysis.py trajectory \
        --runs ${RUNS} \
        --splits ${SPLITS} \
        --fold "${FOLD}" \
        --train-subsample "${TRAIN_SUBSAMPLE}" \
        --with-figures \
        || echo "[warn] trajectory extraction/figures failed"
fi

banner "Done. Outputs:"
echo "  ablations/results/embeddings/<run>_fold${FOLD}_<split>.npz             (single-snapshot)"
echo "  ablations/results/embeddings/trajectory/<run>_fold${FOLD}_ep<N>_<split>.npz   (per-epoch)"
echo
echo "  figures/fig_embed_temp_<split>.{pdf,png}            single-snap PCA+UMAP × temp (2x4 grid)"
echo "  figures/fig_embed_cam_<split>.{pdf,png}             single-snap PCA+UMAP × cam"
echo "  figures/fig_embed_knn_<split>.{pdf,png}             single-snap k-NN MAE bar"
echo "  figures/fig_embed_per_bin_<split>.{pdf,png}         single-snap per-bin cosine, 4 panels"
echo "  figures/fig_embed_cka_<split>.{pdf,png}             single-snap CKA between configs"
echo
echo "  figures/fig_traj_pca_<run>_fold${FOLD}_<split>_temp.{pdf,png}    PCA grid × temp"
echo "  figures/fig_traj_pca_<run>_fold${FOLD}_<split>_cam.{pdf,png}     PCA grid × cam"
echo "  figures/fig_traj_per_bin_<run>_fold${FOLD}_<split>.{pdf,png}     per-bin cosine, per epoch"
echo "  figures/fig_traj_cka_<run>_fold${FOLD}_<split>.{pdf,png}         CKA across epochs"
echo "  figures/fig_traj_knn_<run>_fold${FOLD}.{pdf,png}                 k-NN MAE vs epoch"
