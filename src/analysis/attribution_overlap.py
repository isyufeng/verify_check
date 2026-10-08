"""
VCD-vs-CCD token-attribution overlap on IG outputs, with length control.

Choices fixed before looking at results:
  - unit: subword tokens of the shared tokenizer, special tokens excluded; ranking by |attr|;
  - Jaccard@k for k in {5, 10, 15, 20}; items with n_content <= k are excluded from that k
    (their overlap is 1 by construction) and the excluded count is reported;
  - top-p% with p in {10%, 20%} (k = ceil(p * n_content));
  - baseline: exact hypergeometric expectation of Jaccard for the same (n, k), reported per
    item and as observed minus expected;
  - pairing: k-th active VCD seed with k-th active CCD seed; scopes v1only (primary), mixed.

    python src/analysis/attribution_overlap.py
"""
import argparse
import itertools
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.analysis.collapse import excluded_cells
from src.analysis.rsa import active_runs

RESULTS = Path(__file__).resolve().parent.parent.parent / 'results'
OUT_DIR = RESULTS / 'analysis' / 'attribution'
DATASETS = ('CT22-EN', 'CB', 'CT22-AR', 'CT22-BG', 'CT22-NL', 'CT22-TR')
ENCODERS = ('roberta', 'bert-large', 'electra-large', 'xlm-roberta-large')
KS = (5, 10, 15, 20)
PCTS = (0.10, 0.20)


@lru_cache(maxsize=None)
def expected_jaccard(n, k):
    total = 0.0
    for i in range(max(0, 2 * k - n), k + 1):
        p = math.comb(k, i) * math.comb(n - k, k - i) / math.comb(n, k)
        total += p * i / (2 * k - i)
    return total


def load_attr(run):
    d = np.load(RESULTS / 'attribution' / f'{run}.npz')
    off = d['offsets']
    items = []
    for i in range(len(off) - 1):
        a, s, t = d['attr'][off[i]:off[i + 1]], d['special'][off[i]:off[i + 1]], d['token_ids'][off[i]:off[i + 1]]
        items.append((np.abs(a[~s]), t[~s]))
    return d['ids'], d['verifiable'], d['check_worthy'], items


def topk(scores, k):
    return set(np.argsort(-scores, kind='stable')[:k].tolist())


def overlap_stats(items_a, items_b, mask):
    out = {}
    lengths = np.array([len(items_a[i][0]) for i in range(len(items_a))])
    for k in KS:
        obs, exp, n_excl = [], [], 0
        for i in np.flatnonzero(mask):
            n = lengths[i]
            if n <= k:
                n_excl += 1
                continue
            A, B = topk(items_a[i][0], k), topk(items_b[i][0], k)
            obs.append(len(A & B) / len(A | B))
            exp.append(expected_jaccard(n, k))
        out[f'jaccard@{k}'] = {'obs': float(np.mean(obs)), 'expected': float(np.mean(exp)),
                               'excess': float(np.mean(obs) - np.mean(exp)),
                               'n_items': len(obs), 'n_excluded_short': n_excl}
    for p in PCTS:
        obs, exp = [], []
        for i in np.flatnonzero(mask):
            n = lengths[i]
            k = max(1, math.ceil(p * n))
            if n <= k:
                continue
            A, B = topk(items_a[i][0], k), topk(items_b[i][0], k)
            obs.append(len(A & B) / len(A | B))
            exp.append(expected_jaccard(n, k))
        out[f'top{int(p * 100)}pct'] = {'obs': float(np.mean(obs)), 'expected': float(np.mean(exp)),
                                       'excess': float(np.mean(obs) - np.mean(exp)), 'n_items': len(obs)}
    out['median_content_tokens'] = float(np.median(lengths[mask]))
    return out


def analyse(dataset, encoder, task_b='ccd'):
    va, vb = active_runs(dataset, 'vcd', encoder), active_runs(dataset, task_b, encoder)
    if len(va) < 3 or len(vb) < 3:
        return None
    res = {'dataset': dataset, 'encoder': encoder, 'task_b': task_b, 'per_seed_pair': []}
    for ra, rb in zip(va, vb):
        ids_a, v, cw, items_a = load_attr(ra['run'])
        ids_b, _, _, items_b = load_attr(rb['run'])
        assert (ids_a == ids_b).all()
        assert all((x[1] == y[1]).all() for x, y in zip(items_a, items_b)), 'token mismatch'
        res['per_seed_pair'].append({
            'vcd_seed': int(ra['seed']), f'{task_b}_seed': int(rb['seed']),
            'v1only': overlap_stats(items_a, items_b, cw != -1),
            'mixed': overlap_stats(items_a, items_b, np.ones(len(v), dtype=bool))})
    agg = {}
    for scope in ('v1only', 'mixed'):
        keys = [k for k in res['per_seed_pair'][0][scope] if k != 'median_content_tokens']
        agg[scope] = {k: {m: float(np.mean([p[scope][k][m] for p in res['per_seed_pair']]))
                          for m in ('obs', 'expected', 'excess')} for k in keys}
        agg[scope]['median_content_tokens'] = res['per_seed_pair'][0][scope]['median_content_tokens']
    res['mean_over_seed_pairs'] = agg
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    excluded = {(d, t, e) for d, t, e, _ in excluded_cells()}
    summary = []
    for ds, enc in itertools.product(args.datasets, ENCODERS):
        if (ds, 'vcd', enc) in excluded:
            continue
        res = analyse(ds, enc)
        if res is None:
            continue
        json.dump(res, open(OUT_DIR / f'{ds}__{enc}.json', 'w'), indent=2)
        s = res['mean_over_seed_pairs']['v1only']
        summary.append({'cell': f'{ds}/{enc}', 'median_tok': s['median_content_tokens'],
                        **{k: f"{s[k]['obs']:.3f}/{s[k]['expected']:.3f}/{s[k]['excess']:+.3f}"
                           for k in ('jaccard@5', 'jaccard@10', 'jaccard@20', 'top20pct')}})
        print(summary[-1], flush=True)
    json.dump(summary, open(OUT_DIR / 'summary.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
