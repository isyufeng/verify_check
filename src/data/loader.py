"""
Unified data loading .

Every dataset is read from JSONL into records with a common schema:
    id, text, verifiable (0/1), check_worthy (0/1, or -1 when V=0 or CW is null)

Task views:
    vcd -> all rows, label = verifiable
    ccd -> rows with check_worthy != -1 (V=1 subset), label = check_worthy
    ccd_full -> all rows, label = check_worthy with V=0 mapped to 0
    ccd_seq -> same rows as ccd; the encoder is initialised from a VCD run
    mtl -> all rows; the CCD head ignores check_worthy == -1
"""
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_CT22_LANGS = {'EN': 'english', 'AR': 'arabic', 'BG': 'bulgarian', 'NL': 'dutch', 'TR': 'turkish'}

DATASET_PATHS = {
    **{f'CT22-{k}': f'data/raw/ct22_multilang/{v}/ct22_{v}_{{split}}.jsonl'
       for k, v in _CT22_LANGS.items()},
    'CB': 'data/raw/claimbuster/claimbuster_{split}.jsonl',
}

# (N, N with V=1, N CCD-eligible, N CW=1 among CCD-eligible).
EXPECTED_COUNTS = {
    'CT22-EN': {'train': (3324, 2122, 2122, 447), 'dev': (307, 195, 195, 44), 'test': (251, 149, 149, 39)},
    'CT22-AR': {'train': (3512, 2398, 2398, 906), 'dev': (339, 235, 235, 100), 'test': (1248, 682, 682, 247)},
    'CT22-BG': {'train': (2710, 1871, 1871, 378), 'dev': (251, 177, 177, 36), 'test': (329, 130, 130, 57)},
    'CT22-NL': {'train': (1950, 929, 923, 377), 'dev': (181, 72, 72, 28), 'test': (1357, 608, 603, 315)},
    'CT22-TR': {'train': (2417, 1589, 1589, 422), 'dev': (222, 150, 150, 45), 'test': (512, 303, 303, 14)},
    # CB: original_label 0 = UFS, 1 = CFS, 2 = NFS.
    'CB': {'train': (17649, 6087, 6087, 4238), 'dev': (2354, 812, 812, 565), 'test': (3530, 1218, 1218, 848)},
}

SPLITS = ('train', 'dev', 'test')
TASKS = ('vcd', 'ccd', 'mtl', 'ccd_full', 'ccd_seq')


def _normalize(raw: Dict) -> Dict:
    v = int(raw['verifiable'])
    cw = raw.get('check_worthy')
    # V=0 rows never carry a CCD label, even if the file has an explicit CW=0 (Turkish/Arabic/Dutch).
    cw = -1 if (v == 0 or cw is None) else int(cw)
    return {
        'id': str(raw.get('id', raw.get('claim_id'))),
        'text': raw.get('claim', raw.get('text', '')),
        'verifiable': v,
        'check_worthy': cw,
    }


def load_split(dataset: str, split: str, check_counts: bool = True) -> List[Dict]:
    path = PROJECT_ROOT / DATASET_PATHS[dataset].format(split=split)
    with open(path, encoding='utf-8') as f:
        records = [_normalize(json.loads(line)) for line in f if line.strip()]

    if check_counts:
        n = len(records)
        n_v1 = sum(r['verifiable'] for r in records)
        ccd = [r for r in records if r['check_worthy'] != -1]
        got = (n, n_v1, len(ccd), sum(r['check_worthy'] for r in ccd))
        expected = EXPECTED_COUNTS[dataset][split]
        if got != expected:
            raise ValueError(f"{dataset}/{split}: counts {got} != expected {expected} ({path})")
        if len({r['id'] for r in records}) != n:
            raise ValueError(f"{dataset}/{split}: duplicate ids")
    return records


def task_view(records: List[Dict], task: str) -> List[Dict]:
    if task in ('ccd', 'ccd_seq'):
        return [r for r in records if r['check_worthy'] != -1]
    if task == 'ccd_full':
        # every row, V=0 treated as CW=0.
        return [{**r, 'check_worthy': int(r['check_worthy'] == 1)} for r in records]
    if task in ('vcd', 'mtl'):
        return list(records)
    raise ValueError(f"Unknown task: {task}")


def task_labels(records: List[Dict], task: str) -> List[int]:
    """Single-task label for vcd/ccd views."""
    key = 'verifiable' if task == 'vcd' else 'check_worthy'
    return [r[key] for r in records]


class ClaimDataset(Dataset):
    """Tokenized records carrying both labels; the training loop picks what it needs."""

    def __init__(self, records: List[Dict], tokenizer, max_length: int = 128):
        self.records = records
        self.enc = tokenizer([r['text'] for r in records], truncation=True,
                             max_length=max_length, padding='max_length',
                             return_tensors='pt')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            'input_ids': self.enc['input_ids'][idx],
            'attention_mask': self.enc['attention_mask'][idx],
            'vcd_label': torch.tensor(r['verifiable'], dtype=torch.long),
            'ccd_label': torch.tensor(r['check_worthy'], dtype=torch.long),
        }


if __name__ == '__main__':
    for ds in DATASET_PATHS:
        for sp in SPLITS:
            recs = load_split(ds, sp)
            print(f"{ds:8s} {sp:5s} n={len(recs):5d} ccd={len(task_view(recs, 'ccd')):5d} OK")
