#!/usr/bin/env bash
set -euo pipefail

# scripts/setup_env.sh
# Usage:
#   scripts/setup_env.sh --conda --name frepdetr
#   scripts/setup_env.sh --venv --venv-path .venv
#
# This script helps create either a conda environment or a venv and install
# Python dependencies from requirements.txt. It does NOT install CUDA-specific
# PyTorch wheels automatically; follow the printed hint to install the correct
# torch wheel for your system (see https://pytorch.org/get-started/locally/).

NAME="frepdetr"
VENV_PATH=".venv"
MODE=""

function usage() {
  cat <<EOF
Usage: $0 [--conda|--venv] [--name NAME] [--venv-path PATH]

Examples:
  # create conda env named frepdetr
  $0 --conda --name frepdetr

  # create venv in .venv and install deps
  $0 --venv --venv-path .venv
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda)
      MODE=conda
      shift
      ;;
    --venv)
      MODE=venv
      shift
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --venv-path)
      VENV_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Please choose --conda or --venv"
  usage
fi

if [[ "$MODE" == "conda" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH. Please install Miniconda/Anaconda or use --venv." >&2
    exit 2
  fi
  echo "Creating conda environment: $NAME"
  conda create -n "$NAME" python=3.10 -y
  echo "Activate it with: conda activate $NAME"
  echo "Installing pip, then project requirements..."
  conda run -n "$NAME" pip install --upgrade pip
  # Note: user should install torch wheel specific to CUDA using instructions from pytorch.org
  conda run -n "$NAME" pip install -r requirements.txt
  echo "Done. Remember to install the correct torch wheel if needed per https://pytorch.org/get-started/locally/"

elif [[ "$MODE" == "venv" ]]; then
  echo "Creating venv at: $VENV_PATH"
  python -m venv "$VENV_PATH"
  echo "Activate: source $VENV_PATH/bin/activate"
  echo "Upgrading pip and installing requirements..."
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
  echo "Done. If you need CUDA-specific torch wheels, install them per https://pytorch.org/get-started/locally/"
fi

echo "Tip: For macOS, if pycocotools fails to build, try: xcode-select --install && pip install cython && pip install pycocotools-binary"
