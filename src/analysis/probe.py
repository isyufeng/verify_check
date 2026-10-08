"""
cross-task probing on frozen [CLS] embeddings.

For each target, every representation source is probed on the same test rows:
  CW target: V=1 test rows with a CW label (the CCD test set);  V target: all test rows.
Sources: the VCD encoder, the conditional CCD encoder, the pretrained encoder, and (when
present) ccd_full, ccd_seq and the MTL shared encoder. Note the conditional CCD encoder never
saw V=0 inputs in training, so CCD-enc -> V is evaluated partly out of its training distribution.

Probe: 5-fold stratified CV (random_state 42), Pipeline(StandardScaler, LR C=0.01); the
scaler is fit inside each fold. Metrics: ROC-AUC and PR-AUC, averaged over a source's seeds.

    python src/analysis/probe.py
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.analysis.collapse import excluded_cells
from src.analysis.rsa import active_runs
from src.analysis.rsa_suite import ar_leaked_ids

RESULTS = Path(__file__).resolve().parent.parent.parent / 'results'
OUT_DIR = RESULTS / 'analysis' / 'probe'
DATASETS = ('CT22-EN', 'CB', 'CT22-AR', 'CT22-BG', 'CT22-NL', 'CT22-TR')
ENCODERS = ('roberta', 'bert-large', 'electra-large', 'xlm-roberta-large')
SOURCE_TASKS = ('vcd', 'ccd', 'ccd_full', 'ccd_seq', 'mtl')


def probe_scores(x, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    prob = cross_val_predict(clf, x, y, cv=cv, method='predict_proba')[:, 1]
    return {'auc': float(roc_auc_score(y, prob)), 'pr_auc': float(average_precision_score(y, prob)),
            'pos_rate': float(y.mean()), 'n': int(len(y))}


def load(path):
    d = np.load(path)
    return d['cls'].astype(np.float32), d['verifiable'], d['check_worthy'], d['ids']


def sources_for(dataset, encoder):
    excluded = {(d, t, e) for d, t, e, _ in excluded_cells()}
    out = {}
    for task in SOURCE_TASKS:
        if (dataset, task, encoder) in excluded:
            continue
        runs = active_runs(dataset, task, encoder)
        if len(runs) >= 3:
            out[task] = [RESULTS / 'embeddings' / r['run'] / f'{dataset}_test.npz' for r in runs]
    pre = RESULTS / 'embeddings' / f'pretrained_{encoder}' / f'{dataset}_test.npz'
    if pre.exists() and 'vcd' in out:
        out['pretrained'] = [pre]
    return out


def analyse(dataset, encoder, drop_ids=None):
    srcs = sources_for(dataset, encoder)
    if not {'vcd', 'ccd'} <= set(srcs):
        return None
    res = {'dataset': dataset, 'encoder': encoder, 'dropped_ids': len(drop_ids or ()), 'targets': {}}
    for target in ('cw', 'v'):
        per_src = {}
        for src, paths in srcs.items():
            vals = []
            for p in paths:
                x, v, cw, ids = load(p)
                keep = np.ones(len(v), dtype=bool) if not drop_ids else ~np.isin(ids, list(drop_ids))
                if target == 'cw':
                    keep &= cw != -1  # CCD test set: V=1 rows with a CW label (NL has 5 without)
                    y = cw[keep].astype(int)
                else:
                    y = v[keep].astype(int)
                vals.append(probe_scores(x[keep], y))
            per_src[src] = {
                'auc_mean': float(np.mean([s['auc'] for s in vals])), 'auc_std': float(np.std([s['auc'] for s in vals])),
                'pr_auc_mean': float(np.mean([s['pr_auc'] for s in vals])),
                'n': vals[0]['n'], 'pos_rate': vals[0]['pos_rate'], 'per_seed_auc': [s['auc'] for s in vals]}
        own, cross = ('ccd', 'vcd') if target == 'cw' else ('vcd', 'ccd')
        per_src['_summary'] = {
            'own_source': own, 'cross_source': cross,
            'cross_minus_own_auc': per_src[cross]['auc_mean'] - per_src[own]['auc_mean'],
            'cross_minus_pretrained_auc': (per_src[cross]['auc_mean'] - per_src['pretrained']['auc_mean']
                                           if 'pretrained' in per_src else None),
        }
        res['targets'][target] = per_src
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, enc in itertools.product(args.datasets, ENCODERS):
        variants = [('', None)] + ([('__noleak', ar_leaked_ids())] if ds == 'CT22-AR' else [])
        for tag, drop in variants:
            res = analyse(ds, enc, drop)
            if res is None:
                continue
            json.dump(res, open(OUT_DIR / f'{ds}__{enc}{tag}.json', 'w'), indent=2)
            for t, d in res['targets'].items():
                rows.append({'cell': f'{ds}/{enc}{tag}', 'target': t,
                             **{s: round(v['auc_mean'], 3) for s, v in d.items() if s != '_summary'},
                             'cross-own': round(d['_summary']['cross_minus_own_auc'], 3)})
                print(rows[-1], flush=True)
    json.dump(rows, open(OUT_DIR / 'summary.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
