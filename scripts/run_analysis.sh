#!/bin/bash
# Analyses on the saved predictions, embeddings and attributions (CPU, except the first two steps).
# Usage: bash scripts/run_analysis.sh [N_BOOT]
set -euo pipefail
cd "$(dirname "$0")/.."
N_BOOT=${1:-1000}

# GPU: [CLS] embeddings of the pretrained (not fine-tuned) encoders, used as the RSA/CKA floor.
for enc in roberta bert-large electra-large; do
    python src/analysis/embeddings.py --encoder ${enc} --datasets CT22-EN CB
done
python src/analysis/embeddings.py --encoder xlm-roberta-large \
    --datasets CT22-EN CB CT22-AR CT22-BG CT22-NL CT22-TR
# GPU: conditional CCD scores on the full test split, needed for the gating (cascade) analysis.
python src/analysis/infer_allrows.py

python src/analysis/collapse.py
python src/analysis/attribution_overlap.py
python src/analysis/probe.py
python src/analysis/rsa_suite.py --n_boot ${N_BOOT}
python src/analysis/cka.py
python src/analysis/transfer.py
python src/analysis/stats.py
python src/analysis/stability.py
