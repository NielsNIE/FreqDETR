#!/usr/bin/env bash
# Small helper script with copy/pasteable run examples.
# Usage examples:
#  ./scripts/run_example.sh local
#  ./scripts/run_example.sh multi 4 /absolute/path/to/Maize/data.yaml

set -euo pipefail
MODE=${1:-local}
NUM=${2:-4}
DATA=${3:-configs/dataset/data.yaml}
CONFIG=${4:-configs/model/freqdetr_base.yaml}

if [[ "$MODE" == "local" ]]; then
  echo "Running single-process (debug/local) example:"
  echo "python -m src.train.trainer --config $CONFIG --data $DATA"
elif [[ "$MODE" == "multi" ]]; then
  echo "Running multi-GPU example with ${NUM} GPUs (torchrun):"
  echo "torchrun --nproc_per_node=${NUM} python run_trainer.py --config $CONFIG --data $DATA"
else
  echo "Unknown mode: $MODE"
  echo "Usage: $0 [local|multi <num_gpus> <data_path> [<config>]]"
  exit 1
fi
