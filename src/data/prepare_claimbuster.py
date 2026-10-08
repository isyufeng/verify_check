# -*- coding: utf-8 -*-
"""
Prepare ClaimBuster data for training and RSA analysis.

Reads claimbuster_all.tsv (from HuggingFace Nithiwat/claimbuster),
maps 3-class labels to VCD/CCD binary labels, creates stratified
train/dev/test splits, and outputs JSONL files compatible with
the existing data pipeline.

Label mapping (verified against Arslan et al. 2020, Tables 3/6 and the 1032
groundtruth sentences: NFS=15416, UFS=2466, CFS=5651 in total):
  original 0 = UFS (Unimportant Factual)   -> verifiable=1, check_worthy=0
  original 1 = CFS (Check-worthy Factual)  -> verifiable=1, check_worthy=1
  original 2 = NFS (Non-Factual Sentence)  -> verifiable=0, check_worthy=0
The `verifiable`/`check_worthy` columns of claimbuster_all.tsv encode the opposite (wrong)
NFS/UFS assignment and are ignored; both are derived from `original_label` here.

Usage:
    python src/data/prepare_claimbuster.py \
        --input data/raw/claimbuster/claimbuster_all.tsv \
        --output_dir data/raw/claimbuster
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


LABEL_TO_VCW = {0: (1, 0), 1: (1, 1), 2: (0, 0)}  # original_label -> (verifiable, check_worthy)


def load_tsv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            rows.append({
                'claim_id': str(i),
                'text': row['text'],
                'original_label': int(row['original_label']),
                'verifiable': LABEL_TO_VCW[int(row['original_label'])][0],
                'check_worthy': LABEL_TO_VCW[int(row['original_label'])][1],
            })
    return rows


def save_jsonl(records, path):
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"  Saved {len(records)} records to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw/claimbuster/claimbuster_all.tsv')
    parser.add_argument('--output_dir', default='data/raw/claimbuster')
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--dev_size', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_tsv(args.input)
    print(f"Loaded {len(rows)} total samples")

    # Stratified split using original 3-class label to preserve proportions
    labels_3class = [r['original_label'] for r in rows]

    # First split: train+dev vs test
    train_dev, test = train_test_split(
        rows, test_size=args.test_size, random_state=args.seed,
        stratify=labels_3class)

    # Second split: train vs dev
    td_labels = [r['original_label'] for r in train_dev]
    dev_ratio = args.dev_size / (1.0 - args.test_size)
    train, dev = train_test_split(
        train_dev, test_size=dev_ratio, random_state=args.seed,
        stratify=td_labels)

    print(f"\nSplit sizes: train={len(train)}, dev={len(dev)}, test={len(test)}")

    # Print per-split label distribution
    for name, split in [('train', train), ('dev', dev), ('test', test)]:
        n_ufs = sum(1 for r in split if r['original_label'] == 0)
        n_cfs = sum(1 for r in split if r['original_label'] == 1)
        n_nfs = sum(1 for r in split if r['original_label'] == 2)
        n_v = sum(1 for r in split if r['verifiable'] == 1)
        n_cw = sum(1 for r in split if r['check_worthy'] == 1)
        print(f"  {name}: NFS={n_nfs} CFS={n_cfs} UFS={n_ufs} | "
              f"verifiable={n_v}/{len(split)} ({n_v/len(split)*100:.1f}%) | "
              f"check-worthy={n_cw}/{n_v} ({n_cw/n_v*100:.1f}% of verifiable)")

    print("\nSaving JSONL files...")
    for name, split in [('train', train), ('dev', dev), ('test', test)]:
        records = [{'id': r['claim_id'], 'text': r['text'], 'verifiable': r['verifiable'],
                    'check_worthy': r['check_worthy']} for r in split]
        save_jsonl(records, out / f'claimbuster_{name}.jsonl')

    print("\nDone.")


if __name__ == '__main__':
    main()
