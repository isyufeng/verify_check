#!/bin/bash
# Full training grid (one GPU job per line is recommended; this script runs them sequentially).
# Runs that fail the collapse rule (src/analysis/collapse.py) are replaced by an extra seed (789).
set -euo pipefail
cd "$(dirname "$0")/.."
T="bash scripts/train_one.sh"
SEEDS="42 123 456"

# ELECTRA learning-rate selection: seed 42 at both grid values, then pick the winner.
for ds in CT22-EN CB; do for task in vcd ccd; do for lr in 1e-5 2e-5; do
    $T $ds $task electra-large 42 $lr
done; done; done
python src/analysis/collapse.py
python src/analysis/electra_lr.py
ELECTRA_LR=1e-5   # selected value (see results/analysis/electra_lr_selection.json)

# Single-task VCD and conditional CCD.
for ds in CT22-EN CB; do for task in vcd ccd; do for s in $SEEDS; do
    $T $ds $task roberta $s
    $T $ds $task bert-large $s
    $T $ds $task xlm-roberta-large $s
    [ $s = 42 ] || $T $ds $task electra-large $s $ELECTRA_LR
done; done; done
for ds in CT22-AR CT22-BG CT22-NL CT22-TR; do for task in vcd ccd; do for s in $SEEDS; do
    $T $ds $task xlm-roberta-large $s
done; done; done

# VCD-informed CCD variants on CT22-EN and CB.
for ds in CT22-EN CB; do for s in $SEEDS; do
    $T $ds ccd_seq roberta $s --init_vcd_seed $s
    $T $ds ccd_seq bert-large $s --init_vcd_seed $s
    $T $ds ccd_seq electra-large $s $ELECTRA_LR --init_vcd_seed $s
    $T $ds mtl roberta $s
    $T $ds ccd_full roberta $s
done; done
python src/analysis/collapse.py
