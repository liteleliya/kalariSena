#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMX_DIR="${ROOT_DIR}/GEM-X"
UNITREE_DIR="${ROOT_DIR}/unitree_rl_mjlab"

GEMX_URL="https://github.com/NVlabs/GEM-X.git"
UNITREE_URL="https://github.com/unitreerobotics/unitree_rl_mjlab.git"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but was not found."
  exit 1
fi

if [[ ! -d "${GEMX_DIR}" ]]; then
  echo "Cloning GEM-X..."
  git clone "${GEMX_URL}" "${GEMX_DIR}"
else
  echo "GEM-X already exists, skipping clone."
fi

if [[ -d "${GEMX_DIR}/.git" ]]; then
  echo "Initializing GEM-X submodules..."
  (cd "${GEMX_DIR}" && git submodule update --init --recursive)
fi

if [[ ! -d "${UNITREE_DIR}" ]]; then
  echo "Cloning unitree_rl_mjlab..."
  git clone "${UNITREE_URL}" "${UNITREE_DIR}"
else
  echo "unitree_rl_mjlab already exists, skipping clone."
fi

echo "Subproject setup complete."
