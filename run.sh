#!/usr/bin/env bash
# Conda-env-aware wrapper around `run.py`. Supports both invocation styles:
#
#   1. By family (new):
#        bash run.sh --family linear_probe
#        bash run.sh --family corrupt_random --experiment f1_rate25_baseline
#
#   2. By YAML (back-compat with pre-registry callers):
#        bash run.sh config.yaml
#        bash run.sh ablations/config_ab.yaml --experiment d4_linprobe_resnet --skip-smoke
#
# Detection: if the first arg starts with "--" we assume new-style flags and
# forward verbatim; otherwise the first arg is the YAML path.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${CONDA_ENV_PREFIX:-${PROJECT_DIR}/.conda/skyfinder}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
    module load conda
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available. On Hyak compute nodes, try: module load conda" >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

if [ ! -d "${ENV_PREFIX}" ]; then
    conda create --yes --prefix "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_PREFIX}"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ "${1:-}" == --* || -z "${1:-}" ]]; then
    # New style: forward all args directly.
    python run.py "$@"
else
    # Old style: first positional arg is the YAML.
    CONFIG_PATH="$1"
    shift
    python run.py --config "${CONFIG_PATH}" "$@"
fi
