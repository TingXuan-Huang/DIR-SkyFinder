#!/usr/bin/env bash
# Submit an experiment family (or single experiment) as a SLURM array.
# Computes the array size from the run.py registry, no hardcoding.
#
# Usage:
#     bash ablations/submit.sh <family> [--experiment NAME] [--concurrent N] [sbatch flags...]
#
# Examples:
#     # Whole family, 3 concurrent (default), on gpu-2080ti:
#     bash ablations/submit.sh corrupt_random -p gpu-2080ti --gpus=1
#
#     # Single experiment from a family (1-task array):
#     bash ablations/submit.sh corrupt_random --experiment f1_rate25_baseline -p gpu-2080ti --gpus=1
#
#     # Adjust concurrency:
#     bash ablations/submit.sh label_corruption --concurrent 6 -p gpu-2080ti --gpus=1
set -euo pipefail

usage() {
    cat <<EOF
usage: bash ablations/submit.sh <family> [--experiment NAME] [--concurrent N] [sbatch flags...]

discovery:
    python run.py --list                          # all families with run-counts
    python run.py --list <family>                 # experiments inside one family
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage; exit 0
fi

FAMILY="$1"; shift
EXPERIMENT=""
CONCURRENT=3
SBATCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --concurrent) CONCURRENT="$2"; shift 2 ;;
        *)            SBATCH_ARGS+=("$1"); shift ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ -n "${EXPERIMENT}" ]]; then
    # Single-experiment mode: 1-task array (always size 1).
    ARRAY="0-0"
    echo "[submit] family=${FAMILY}  experiment=${EXPERIMENT}  array=${ARRAY}"
    export FAMILY="${FAMILY}" EXPERIMENT="${EXPERIMENT}"
else
    # Whole family: ask run.py how many experiments are in it.
    N=$(python run.py --count "${FAMILY}")
    if [[ -z "${N}" || "${N}" -eq 0 ]]; then
        echo "no experiments in family '${FAMILY}'" >&2
        echo "available families:" >&2
        python run.py --list >&2
        exit 1
    fi
    ARRAY="0-$((N - 1))%${CONCURRENT}"
    echo "[submit] family=${FAMILY}  count=${N}  array=${ARRAY}  concurrent=${CONCURRENT}"
    export FAMILY="${FAMILY}"
fi

mkdir -p ablations/logs
sbatch --export=ALL --array="${ARRAY}" "${SBATCH_ARGS[@]}" ablations/run_family.slurm
