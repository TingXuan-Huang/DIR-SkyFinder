#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config.yaml}"
shift || true  # consume the config path; remaining "$@" forwards to run.py (e.g. --experiment NAME)
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
python run.py --config "${CONFIG_PATH}" "$@"
