"""
seed stability of VCD, CCD and VCD-informed CCD variants (exploratory).

Per dataset x encoder and model type (vcd, ccd, ccd_seq, mtl, ccd_full):
  - representation stability: mean RSA and linear CKA over seed pairs (CCD test set);
  - prediction stability on the task's own test set: mean Cohen's kappa of decisions and
    Spearman rho of positive-class probabilities over seed pairs;
  - std over seeds of test F1 (positive class) and ROC-AUC.
H1: CCD less stable than VCD (all cells). H2: ccd_seq / mtl more stable than ccd.

    python src/analysis/stability.py
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.analysis.rsa import active_runs, load_emb
from src.analysis.rsa_suite import DATASETS, ENCODERS, linear_cka
from src.data.loader import load_split

RESULTS = Path(__file__).resolve().parent.parent.parent / 'results'
TASKS = ('vcd', 'ccd', 'ccd_seq', 'mtl', 'ccd_full')


def preds(run, head, ids):
    with open(RESULTS / 'runs' / run / f'pred_test_{head}.csv') as f:
        p = {r['id']: (int(r['pred']), float(r['prob_pos'])) for r in csv.DictReader(f)}
    return np.array([p[i][0] for i in ids]), np.array([p[i][1] for i in ids])


def stability(dataset, encoder, task):
    runs = active_runs(dataset, task, encoder)
    if len(runs) < 3:
        return None
    test = load_split(dataset, 'test')
    ccd_ids = [r['id'] for r in test if r['check_worthy'] != -1]
    if task == 'vcd':
        ids, y = [r['id'] for r in test], np.array([r['verifiable'] for r in test])
    else:
        ids, y = ccd_ids, np.array([r['check_worthy'] for r in test if r['check_worthy'] != -1])
    head = 'vcd' if task == 'vcd' else 'ccd'

    embs = []
    for r in runs:
        x, _, cw, _ = load_emb(r['run'], dataset)
        embs.append(x[cw != -1])
    rdms = [pdist(e, 'cosine') for e in embs]
    rsa = [float(spearmanr(a, b)[0]) for a, b in itertools.combinations(rdms, 2)]
    cka = [linear_cka(a, b) for a, b in itertools.combinations(embs, 2)]

    P = [preds(r['run'], head, ids) for r in runs]
    kappa = [cohen_kappa_score(a[0], b[0]) for a, b in itertools.combinations(P, 2)]
    rho = [float(spearmanr(a[1], b[1])[0]) for a, b in itertools.combinations(P, 2)]
    f1s = [f1_score(y, p, zero_division=0) for p, _ in P]
    aucs = [roc_auc_score(y, q) for _, q in P]
    return {'dataset': dataset, 'encoder': encoder, 'task': task, 'seeds': [int(r['seed']) for r in runs],
            'rsa_within': float(np.mean(rsa)), 'cka_within': float(np.mean(cka)),
            'kappa': float(np.mean(kappa)), 'prob_rho': float(np.mean(rho)),
            'f1_std': float(np.std(f1s)), 'auc_std': float(np.std(aucs)),
            'f1_mean': float(np.mean(f1s)), 'auc_mean': float(np.mean(aucs))}


def paired(rows, a, b, metric, lower_is_stabler=False):
    """Cells where task b is less stable than task a on `metric` (or more stable, for H2)."""
    by = {(r['dataset'], r['encoder'], r['task']): r for r in rows}
    cells = sorted({(d, e) for d, e, t in by if (d, e, a) in by and (d, e, b) in by})
    diffs = [by[(d, e, b)][metric] - by[(d, e, a)][metric] for d, e in cells]
    if not diffs:
        return None
    worse = [d > 0 if lower_is_stabler else d < 0 for d in diffs]
    k = int(sum(worse))
    return {'metric': metric, 'n_cells': len(diffs), f'n_{b}_less_stable_than_{a}': k,
            'p_sign_one_sided': float(binomtest(k, len(diffs), 0.5, alternative='greater').pvalue),
            'median_diff': float(np.median(diffs)), 'cells': [f'{d}/{e}' for d, e in cells],
            'diffs': [float(x) for x in diffs]}


def main():
    rows = []
    for ds, enc, task in itertools.product(DATASETS, ENCODERS, TASKS):
        r = stability(ds, enc, task)
        if r:
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'seeds'}, flush=True)
    tests = {}
    for metric, low in (('rsa_within', False), ('cka_within', False), ('kappa', False), ('prob_rho', False),
                        ('f1_std', True), ('auc_std', True)):
        tests[f'H1_ccd_vs_vcd__{metric}'] = paired(rows, 'vcd', 'ccd', metric, low)
        for alt in ('ccd_seq', 'mtl'):
            # H2: alt more stable than ccd  <=> ccd less stable than alt
            tests[f'H2_{alt}_vs_ccd__{metric}'] = paired(rows, alt, 'ccd', metric, low)
    out = RESULTS / 'analysis' / 'stability.json'
    json.dump({'rows': rows, 'tests': tests}, open(out, 'w'), indent=2)
    for k, v in tests.items():
        if v:
            key = [x for x in v if x.startswith('n_') and x != 'n_cells'][0]
            print(f"{k:34s} {key[2:]:>32s}: {v[key]}/{v['n_cells']}  p={v['p_sign_one_sided']:.3f}  "
                  f"median(ccd - other)={v['median_diff']:+.3f}")


if __name__ == '__main__':
    main()
