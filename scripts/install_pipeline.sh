#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m ensurepip --upgrade
python -m pip install --upgrade pip

python -m pip install -r "${ROOT_DIR}/requirements.pipeline.txt"

echo "Pipeline dependencies installed."
echo "GEM-X dependencies are not installed by this script."
echo "See GEM-X/README.md and GEM-X/docs/INSTALL.md for GPU setup."
