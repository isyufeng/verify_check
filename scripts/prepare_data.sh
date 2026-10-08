#!/bin/bash
# Build the paired (V, CW) JSONL files used by every experiment.
# Expects the CheckThat! 2022 Task 1 release under data/raw/clef2022-checkthat-lab/
# and ClaimBuster (claimbuster_all.tsv) under data/raw/claimbuster/.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/data/build_ct22_multilang.py
python src/data/prepare_claimbuster.py --input data/raw/claimbuster/claimbuster_all.tsv \
    --output_dir data/raw/claimbuster
