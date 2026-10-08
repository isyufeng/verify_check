"""
does sharing or transfer help CCD, compared with single-task CCD?
All comparisons on the same V=1 test rows.

Comparisons (k-th active run of each side paired by rank):
  ccd_seq  vs ccd   VCD->CCD sequential fine-tuning
  mtl      vs ccd   shared-encoder MTL (CCD head); also MTL VCD head vs single VCD
  ccd_full vs ccd   full-scope CCD (V=0 as CW=0), evaluated on V=1 rows
  proxy    vs ccd   P(V=1) from the VCD model used directly as the CW score (AUC / PR-AUC only)
Uncertainty: 2000 bootstrap resamples of test items (shared across the two systems) and of
seed pairs; 95% percentile CI of the mean difference. Verdict: gain (CI > 0), drop (CI < 0),
uncertain (CI spans 0).

    python src/analysis/transfer.py
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.analysis.rsa import active_runs
from src.data.loader import load_split

RESULTS = Path(__file__).resolve().parent.parent.parent / 'results'
OUT_DIR = RESULTS / 'analysis' / 'transfer'
N_BOOT = int(__import__("os").environ.get("TRANSFER_NBOOT", 2000))
METRICS = ('f1_pos', 'pr_auc', 'roc_auc')


def read_preds(run, head):
    with open(RESULTS / 'runs' / run / f'pred_test_{head}.csv') as f:
        rows = list(csv.DictReader(f))
    return {r['id']: (int(r['pred']), float(r['prob_pos'])) for r in rows}


def scores(y, pred, prob):
    out = {'pr_auc': average_precision_score(y, prob), 'roc_auc': roc_auc_score(y, prob)}
    if pred is not None:
        out['f1_pos'] = f1_score(y, pred, zero_division=0)
    return out


def system_arrays(runs, head, ids):
    arrs = []
    for r in runs:
        p = read_preds(r['run'], head)
        arrs.append((np.array([p[i][0] for i in ids]), np.array([p[i][1] for i in ids])))
    return arrs


def compare(dataset, sys_a, sys_b, y, metrics, rng):
    """Mean over rank-paired seeds of metric(b) - metric(a), with a two-level bootstrap."""
    n, k = len(y), min(len(sys_a), len(sys_b))
    pairs = list(zip(sys_a[:k], sys_b[:k]))

    def diff(idx, pair_idx):
        out = {m: [] for m in metrics}
        for j in pair_idx:
            (pa, qa), (pb, qb) = pairs[j]
            sa = scores(y[idx], pa[idx] if pa is not None else None, qa[idx])
            sb = scores(y[idx], pb[idx] if pb is not None else None, qb[idx])
            for m in metrics:
                out[m].append(sb[m] - sa[m])
        return {m: float(np.mean(v)) for m, v in out.items()}

    point = diff(np.arange(n), range(k))
    boots = {m: [] for m in metrics}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        d = diff(idx, rng.integers(0, k, k))
        for m in metrics:
            boots[m].append(d[m])
    res = {}
    for m in metrics:
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        verdict = 'gain' if lo > 0 else ('drop' if hi < 0 else 'uncertain')
        res[m] = {'diff': point[m], 'ci95': [float(lo), float(hi)], 'verdict': verdict}
    res['abs_b'] = {m: float(np.mean([scores(y, pb, qb)[m] for _, (pb, qb) in pairs]))
                    for m in metrics}
    res['abs_a'] = {m: float(np.mean([scores(y, pa, qa)[m] for (pa, qa), _ in pairs]))
                    for m in metrics}
    res['n_pairs'] = k
    return res


def read_allrows(run):
    with open(RESULTS / 'runs' / run / 'pred_test_ccd_allrows.csv') as f:
        return {r['id']: (int(r['pred']), float(r['prob_pos'])) for r in csv.DictReader(f)}


def cascade_eval(dataset, encoder, rng):
    """end-to-end on the full test split (V=0 rows count as CW=0)."""
    vcd, ccd = active_runs(dataset, 'vcd', encoder), active_runs(dataset, 'ccd', encoder)
    if len(vcd) < 3 or len(ccd) < 3:
        return None
    if not all((RESULTS / 'runs' / r['run'] / 'pred_test_ccd_allrows.csv').exists() for r in ccd):
        return None
    test = load_split(dataset, 'test')
    ids = [r['id'] for r in test]
    y = np.array([int(r['check_worthy'] == 1) for r in test])
    gold_v = np.array([r['verifiable'] for r in test])
    v_preds = [read_preds(r['run'], 'vcd') for r in vcd]
    c_preds = [read_allrows(r['run']) for r in ccd]
    systems = {'cascade': [], 'oracle_gate': [], 'no_gate': []}
    leak = []
    for vp, cp in zip(v_preds, c_preds):
        vg = np.array([vp[i][0] for i in ids])
        cpred = np.array([cp[i][0] for i in ids])
        cprob = np.array([cp[i][1] for i in ids])
        systems['cascade'].append((vg * cpred, np.where(vg == 1, cprob, 0.0)))
        systems['oracle_gate'].append((gold_v * cpred, np.where(gold_v == 1, cprob, 0.0)))
        systems['no_gate'].append((cpred, cprob))
        leak.append(float(((vg == 0) & (y == 1)).sum() / y.sum()))
    res = {'dataset': dataset, 'encoder': encoder, 'n_test': len(y), 'cw_rate_all': float(y.mean()),
           'gate_miss_rate_mean': float(np.mean(leak)), 'gate_miss_rate_per_pair': leak,
           'abs': {name: {m: float(np.mean([scores(y, p, q)[m] for p, q in arrs])) for m in METRICS}
                   for name, arrs in systems.items()}}
    res['cascade_vs_no_gate'] = compare(dataset, systems['no_gate'], systems['cascade'], y, METRICS, rng)
    res['cascade_vs_oracle_gate'] = compare(dataset, systems['oracle_gate'], systems['cascade'], y, METRICS, rng)
    full = active_runs(dataset, 'ccd_full', encoder)
    if len(full) >= 3:
        fa = system_arrays(full, 'ccd', ids)
        res['ccd_full_vs_cascade'] = compare(dataset, systems['cascade'], fa, y, METRICS, rng)
        res['abs']['ccd_full'] = {m: float(np.mean([scores(y, p, q)[m] for p, q in fa])) for m in METRICS}
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results, summary = [], []
    for dataset, encoder in itertools.product(('CT22-EN', 'CB'),
                                              ('roberta', 'bert-large', 'electra-large', 'xlm-roberta-large')):
        test = [r for r in load_split(dataset, 'test') if r['verifiable'] == 1]
        ids = [r['id'] for r in test]
        y = np.array([r['check_worthy'] for r in test])
        base_runs = active_runs(dataset, 'ccd', encoder)
        if len(base_runs) < 3:
            continue
        base = system_arrays(base_runs, 'ccd', ids)
        candidates = {
            'ccd_seq': ('ccd_seq', 'ccd', METRICS),
            'mtl': ('mtl', 'ccd', METRICS),
            'ccd_full': ('ccd_full', 'ccd', METRICS),
            'proxy_vcd_prob': ('vcd', 'vcd', ('pr_auc', 'roc_auc')),
        }
        for name, (task, head, metrics) in candidates.items():
            runs = active_runs(dataset, task, encoder)
            if len(runs) < 3:
                continue
            other = system_arrays(runs, head, ids)
            if name == 'proxy_vcd_prob':
                other = [(None, prob) for _, prob in other]
            res = compare(dataset, base, other, y, metrics, rng)
            rec = {'dataset': dataset, 'encoder': encoder, 'system': name, 'baseline': 'ccd',
                   'n_test_v1': len(y), 'cw_rate': float(y.mean()), **res}
            results.append(rec)
            summary.append({'cell': f'{dataset}/{encoder}', 'system': name,
                            **{m: f"{res[m]['diff']:+.3f} [{res[m]['ci95'][0]:+.3f},{res[m]['ci95'][1]:+.3f}] {res[m]['verdict']}"
                               for m in metrics}})
            print(summary[-1], flush=True)
        # MTL VCD head vs single-task VCD, all test rows
        mtl, vcd = active_runs(dataset, 'mtl', encoder), active_runs(dataset, 'vcd', encoder)
        if len(mtl) >= 3 and len(vcd) >= 3:
            all_test = load_split(dataset, 'test')
            ids_all = [r['id'] for r in all_test]
            yv = np.array([r['verifiable'] for r in all_test])
            res = compare(dataset, system_arrays(vcd, 'vcd', ids_all), system_arrays(mtl, 'vcd', ids_all),
                          yv, METRICS, rng)
            results.append({'dataset': dataset, 'encoder': encoder, 'system': 'mtl_vcd_head',
                            'baseline': 'vcd', 'n_test': len(yv), **res})
            print({'cell': f'{dataset}/{encoder}', 'system': 'mtl_vcd_head vs vcd',
                   **{m: f"{res[m]['diff']:+.3f} {res[m]['verdict']}" for m in METRICS}}, flush=True)
    cascades = []
    for dataset, encoder in itertools.product(('CT22-EN', 'CB'),
                                              ('roberta', 'bert-large', 'electra-large', 'xlm-roberta-large')):
        c = cascade_eval(dataset, encoder, rng)
        if c is None:
            continue
        cascades.append(c)
        print({'cascade': f'{dataset}/{encoder}', 'miss_rate': round(c['gate_miss_rate_mean'], 3),
               **{k: f"{c[k]['f1_pos']['diff']:+.3f} {c[k]['f1_pos']['verdict']} / auc {c[k]['roc_auc']['diff']:+.3f}"
                  for k in ('cascade_vs_no_gate', 'cascade_vs_oracle_gate', 'ccd_full_vs_cascade') if k in c}},
              flush=True)
    json.dump(cascades, open(OUT_DIR / 'cascade_results.json', 'w'), indent=2)
    json.dump(results, open(OUT_DIR / 'transfer_results.json', 'w'), indent=2)
    json.dump(summary, open(OUT_DIR / 'summary.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
