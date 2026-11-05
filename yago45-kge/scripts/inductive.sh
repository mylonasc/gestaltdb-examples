#!/usr/bin/env bash
# Shared GPU environment for inductive preparation, training, evaluation/reporting.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${YAGO_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
export KERAS_BACKEND=tensorflow
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
CUDA_LIB_PATH="$("${PYTHON}" - <<'PY'
import pathlib, site
root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(p / "lib") for p in root.iterdir() if (p / "lib").exists()) if root.exists() else "")
PY
)"
export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
ACTION="${1:?usage: inductive.sh prepare|temporal|train|tune|eval|report|cache [args]}"
shift
exec "${PYTHON}" -m "yago.inductive_${ACTION}" "$@"
