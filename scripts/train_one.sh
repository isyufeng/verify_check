#!/bin/bash
# One training run. Usage:
#   bash scripts/train_one.sh DATASET TASK ENCODER SEED [LR] [extra flags]
# e.g. bash scripts/train_one.sh CB ccd_seq roberta 42 --init_vcd_seed 42
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
DATASET=$1; TASK=$2; ENCODER=$3; SEED=$4; shift 4
LR=""
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then LR=$1; shift; fi
python src/training/train.py --dataset "${DATASET}" --task "${TASK}" --encoder "${ENCODER}" \
    --seed "${SEED}" ${LR:+--lr ${LR}} "$@"
