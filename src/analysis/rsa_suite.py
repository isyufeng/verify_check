"""
RSA baselines and uncertainty on [CLS] embeddings.

For every dataset x encoder cell with active runs for both tasks of a pair
(default pair: vcd vs ccd; also vcd vs ccd_full / ccd_seq when present):
  - cross-task RSA: rank-paired seeds and the mean over all seed pairs;
  - within-task RSA across seeds (ceiling) and pretrained-vs-fine-tuned RSA (floor);
  - item-bootstrap CIs of the cell mean: data-only, and data+seed (seed pairs resampled too);
  - linear CKA (exploratory);
  - CT22-AR sensitivity without the 169 test tweets also present in train.
Cross-input RSA (CT22-EN encoders on CB text and vice versa) is run as exploratory.

Bootstrap approximation: the full RDMs are rank-transformed once and each resample uses
the Pearson correlation of those ranks (pairs of identical resampled items dropped).
Point estimates are exact Spearman.

    python src/analysis/rsa_suite.py [--n_boot 1000]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.analysis.collapse import excluded_cells
from src.analysis.rsa import active_runs, load_emb
from src.data.loader import load_split

RESULTS = Path(__file__).resolve().parent.parent.parent / 'results'
OUT_DIR = RESULTS / 'analysis' / 'rsa_suite'
DATASETS = ('CT22-EN', 'CB', 'CT22-AR', 'CT22-BG', 'CT22-NL', 'CT22-TR')
ENCODERS = ('roberta', 'bert-large', 'electra-large', 'xlm-roberta-large')
PAIRS = (('vcd', 'ccd'), ('vcd', 'ccd_full'), ('vcd', 'ccd_seq'))


def ar_leaked_ids():
    train_ids = {r['id'] for r in load_split('CT22-AR', 'train')}
    return {r['id'] for r in load_split('CT22-AR', 'test') if r['id'] in train_ids}


def spearman_rsa(xa, xb):
    return float(spearmanr(pdist(xa, 'cosine'), pdist(xb, 'cosine'))[0])


def linear_cka(xa, xb):
    xa = xa - xa.mean(0)
    xb = xb - xb.mean(0)
    hsic = np.linalg.norm(xa.T @ xb, 'fro') ** 2
    return float(hsic / (np.linalg.norm(xa.T @ xa, 'fro') * np.linalg.norm(xb.T @ xb, 'fro')))


def ranked_rdm(x):
    """Square matrix of the ranks of the condensed cosine RDM."""
    return squareform(rankdata(pdist(x, 'cosine')).astype(np.float32), checks=False)


def boot_rsa(ranked_pairs, n, n_boot, rng, resample_seeds):
    iu = np.triu_indices(n, k=1)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        keep = idx[iu[0]] != idx[iu[1]]
        ri, rj = idx[iu[0]][keep], idx[iu[1]][keep]
        pairs = ([ranked_pairs[k] for k in rng.integers(0, len(ranked_pairs), len(ranked_pairs))]
                 if resample_seeds else ranked_pairs)
        vals = [np.corrcoef(ra[ri, rj], rb[ri, rj])[0, 1] for ra, rb in pairs]
        out[b] = np.mean(vals)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def mask_for(cw, ids, scope, drop_ids=None):
    # v1only = the CCD test set: V=1 rows that carry a CW label (NL has 5 V=1 rows without one)
    m = (cw != -1) if scope == 'v1only' else np.ones(len(cw), dtype=bool)
    if drop_ids:
        m &= ~np.isin(ids, list(drop_ids))
    return m


def analyse_cell(dataset, encoder, task_a, task_b, eval_dataset, n_boot, rng, drop_ids=None):
    ra, rb = active_runs(dataset, task_a, encoder), active_runs(dataset, task_b, encoder)
    if len(ra) < 3 or len(rb) < 3:
        return None
    ea = [load_emb(r['run'], eval_dataset) for r in ra]
    eb = [load_emb(r['run'], eval_dataset) for r in rb]
    cw, ids = ea[0][2], ea[0][3]
    assert all((e[3] == ids).all() for e in ea + eb), 'row order mismatch'
    pre_dir = RESULTS / 'embeddings' / f'pretrained_{encoder}' / f'{eval_dataset}_test.npz'
    pre = np.load(pre_dir)['cls'].astype(np.float32) if pre_dir.exists() else None

    res = {'dataset': dataset, 'eval_dataset': eval_dataset, 'encoder': encoder,
           'task_a': task_a, 'task_b': task_b,
           'seeds_a': [int(r['seed']) for r in ra], 'seeds_b': [int(r['seed']) for r in rb],
           'dropped_ids': len(drop_ids or ())}
    for scope in ('v1only', 'mixed'):
        m = mask_for(cw, ids, scope, drop_ids)
        A = [e[0][m] for e in ea]
        B = [e[0][m] for e in eb]
        paired = [spearman_rsa(a, b) for a, b in zip(A, B)]
        allp = [spearman_rsa(a, b) for a, b in itertools.product(A, B)]
        within_a = [spearman_rsa(x, y) for x, y in itertools.combinations(A, 2)]
        within_b = [spearman_rsa(x, y) for x, y in itertools.combinations(B, 2)]
        r = {
            'n': int(m.sum()),
            'cross_paired': paired, 'cross_paired_mean': float(np.mean(paired)),
            'cross_paired_seed_std': float(np.std(paired)),
            'cross_all_pairs_mean': float(np.mean(allp)),
            f'within_{task_a}': within_a, f'within_{task_a}_mean': float(np.mean(within_a)),
            f'within_{task_b}': within_b, f'within_{task_b}_mean': float(np.mean(within_b)),
            'cka_cross_paired_mean': float(np.mean([linear_cka(a, b) for a, b in zip(A, B)])),
        }
        if pre is not None:
            r[f'pretrained_vs_{task_a}_mean'] = float(np.mean([spearman_rsa(pre[m], a) for a in A]))
            r[f'pretrained_vs_{task_b}_mean'] = float(np.mean([spearman_rsa(pre[m], b) for b in B]))
        if n_boot and scope == 'v1only':
            ranked = [(ranked_rdm(a), ranked_rdm(b)) for a, b in zip(A, B)]
            r['ci95_data'] = boot_rsa(ranked, r['n'], n_boot, rng, resample_seeds=False)
            r['ci95_data_seed'] = boot_rsa(ranked, r['n'], n_boot, rng, resample_seeds=True)
            r['n_boot'] = n_boot
        res[scope] = r
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = p.parse_args()
    rng = np.random.default_rng(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    excluded = {(d, t, e) for d, t, e, _ in excluded_cells()}

    jobs = []
    for ds, enc, (ta, tb) in itertools.product(args.datasets, ENCODERS, PAIRS):
        if (ds, ta, enc) in excluded or (ds, tb, enc) in excluded:
            continue
        jobs.append((ds, enc, ta, tb, ds, None, ''))
        if ds == 'CT22-AR' and (ta, tb) == ('vcd', 'ccd'):
            jobs.append((ds, enc, ta, tb, ds, ar_leaked_ids(), '__noleak'))
        if (ta, tb) == ('vcd', 'ccd') and ds in ('CT22-EN', 'CB'):
            other = 'CB' if ds == 'CT22-EN' else 'CT22-EN'
            jobs.append((ds, enc, ta, tb, other, None, ''))  # exploratory cross-input

    summary = []
    for ds, enc, ta, tb, ev, drop, tag in jobs:
        res = analyse_cell(ds, enc, ta, tb, ev, args.n_boot if ev == ds else 0, rng, drop)
        if res is None:
            continue
        name = f'{ds}__{enc}__{ta}-vs-{tb}__on_{ev}{tag}.json'
        json.dump(res, open(OUT_DIR / name, 'w'), indent=2)
        s = res['v1only']
        ci = s.get('ci95_data_seed')
        summary.append({'file': name, 'n': s['n'], 'cross': round(s['cross_paired_mean'], 3),
                        'ci_data_seed': [round(x, 3) for x in ci] if ci else None,
                        f'within_{ta}': round(s[f'within_{ta}_mean'], 3),
                        f'within_{tb}': round(s[f'within_{tb}_mean'], 3),
                        'mixed_cross': round(res['mixed']['cross_paired_mean'], 3)})
        print(summary[-1], flush=True)
    json.dump(summary, open(OUT_DIR / 'summary.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
