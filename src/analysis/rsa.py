"""
Cross-encoder RSA on [CLS] embeddings: cosine RDM, Spearman rho,
all eligible test rows (no subsampling), scopes v1only (primary) and mixed.

Only seeds marked active in results/run_registry.csv are used. Two pairings are
reported: rank-paired (k-th active VCD seed vs k-th active CCD seed) and the mean over
all active VCD x CCD seed pairs (pairing-free).

    python src/analysis/rsa.py --dataset CT22-EN --encoder roberta
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

RESULTS = Path(__file__).resolve().parent.parent.parent / 'results'


def selected_lr(encoder):
    """ELECTRA runs at the non-selected lr are selection-only."""
    if encoder != 'electra-large':
        return None
    path = RESULTS / 'analysis' / 'electra_lr_selection.json'
    return json.load(open(path))['selected_lr'] if path.exists() else 'undecided'


def active_runs(dataset, task, encoder):
    lr = selected_lr(encoder)
    if lr == 'undecided':
        return []
    with open(RESULTS / 'run_registry.csv') as f:
        rows = [r for r in csv.DictReader(f)
                if r['dataset'] == dataset and r['task'] == task and r['encoder'] == encoder
                and r['active'] == 'True' and (lr is None or float(r['lr']) == lr)]
    order = (42, 123, 456, 789, 999)
    return sorted(rows, key=lambda r: order.index(int(r['seed'])))


def load_emb(run, dataset, split='test', source='runs'):
    d = np.load(RESULTS / 'embeddings' / run / f'{dataset}_{split}.npz')
    return d['cls'].astype(np.float32), d['verifiable'], d['check_worthy'], d['ids']


def rdm(x):
    return pdist(x, metric='cosine')


def rsa(xa, xb):
    return float(spearmanr(rdm(xa), rdm(xb))[0])


def scope_mask(verifiable, scope):
    return verifiable == 1 if scope == 'v1only' else np.ones_like(verifiable, dtype=bool)


def cross_task_rsa(dataset, encoder, eval_dataset=None, split='test'):
    eval_dataset = eval_dataset or dataset
    vcd = active_runs(dataset, 'vcd', encoder)
    ccd = active_runs(dataset, 'ccd', encoder)
    embs = {r['run']: load_emb(r['run'], eval_dataset, split) for r in vcd + ccd}
    ids0 = next(iter(embs.values()))[3]
    assert all((e[3] == ids0).all() for e in embs.values()), 'row order mismatch'
    v_lab = next(iter(embs.values()))[1]

    out = {'dataset': dataset, 'eval_dataset': eval_dataset, 'encoder': encoder,
           'vcd_seeds': [int(r['seed']) for r in vcd], 'ccd_seeds': [int(r['seed']) for r in ccd]}
    for scope in ('v1only', 'mixed'):
        m = scope_mask(v_lab, scope)
        paired = [rsa(embs[a['run']][0][m], embs[b['run']][0][m]) for a, b in zip(vcd, ccd)]
        allp = [rsa(embs[a['run']][0][m], embs[b['run']][0][m])
                for a, b in itertools.product(vcd, ccd)]
        out[scope] = {'n': int(m.sum()),
                      'paired': paired, 'paired_mean': float(np.mean(paired)),
                      'paired_std': float(np.std(paired, ddof=0)),
                      'all_pairs_mean': float(np.mean(allp)), 'all_pairs_std': float(np.std(allp))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--encoder', required=True)
    p.add_argument('--eval_dataset', default=None, help='Cross-input RSA (exploratory)')
    args = p.parse_args()
    res = cross_task_rsa(args.dataset, args.encoder, args.eval_dataset)
    out_dir = RESULTS / 'analysis' / 'rsa'
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}__{args.encoder}__on_{res['eval_dataset']}.json"
    with open(out_dir / name, 'w') as f:
        json.dump(res, f, indent=2)
    for scope in ('v1only', 'mixed'):
        s = res[scope]
        print(f"{args.dataset} {args.encoder} on {res['eval_dataset']} [{scope}] n={s['n']} "
              f"paired={s['paired_mean']:.3f}±{s['paired_std']:.3f} {[round(x, 3) for x in s['paired']]} "
              f"all-pairs={s['all_pairs_mean']:.3f}±{s['all_pairs_std']:.3f}")


if __name__ == '__main__':
    main()
