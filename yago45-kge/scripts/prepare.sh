#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
export KERAS_BACKEND=tensorflow
CUDA_LIB_PATH="$(${PYTHON} - <<'PY'
import pathlib, site
root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(p / "lib") for p in root.iterdir() if (p / "lib").exists())) if root.exists() else print("")
PY
)"
if [[ -n "${CUDA_LIB_PATH}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
fi
exec "${PYTHON}" "${ROOT_DIR}/yago/prepare_cli.py" "$@"
