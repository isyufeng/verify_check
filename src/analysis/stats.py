"""
Cell-level statistics on top of rsa_suite / cka / transfer outputs.

0. Primary: cross-task similarity vs the same-task ceiling, per cell,
   for RSA and linear CKA. One-sided exact sign test and Wilcoxon signed-rank over cells, with and
   without CT22-TR (unstable ceiling); seed-level separation (max cross pair < min within pair);
   fraction of ceiling.
1. Secondary: CT22 cells vs CB cells on the v1only cross-task RSA cell mean (seeds averaged
   within a cell). Exact two-sided permutation test on the difference of means + Mann-Whitney U.
   Also reported: paired CT22-EN minus CB per shared encoder, and the RSA normalised by the
   within-task ceiling, mean(within_vcd, within_ccd) (exploratory).
2. RSA vs transfer: per dataset x encoder, cross-task RSA against the ccd_seq - ccd difference
   (descriptive; at most 6 points).

    python src/analysis/stats.py
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS = ROOT / 'results' / 'analysis'


def load_cells():
    cells = {}
    for f in (ANALYSIS / 'rsa_suite').glob('*__vcd-vs-ccd__on_*.json'):
        r = json.load(open(f))
        if r['eval_dataset'] != r['dataset'] or r['dropped_ids']:
            continue
        s = r['v1only']
        ceiling = np.mean([s['within_vcd_mean'], s['within_ccd_mean']])
        cells[(r['dataset'], r['encoder'])] = {
            'rsa': s['cross_paired_mean'], 'ci95': s.get('ci95_data_seed'),
            'ceiling': float(ceiling), 'normalised': float(s['cross_paired_mean'] / ceiling),
            'seed_separated': bool(max(s['cross_paired']) < min(s['within_vcd'] + s['within_ccd']))}
    return cells


def ceiling_test(pairs, label):
    """pairs: list of (cross, ceiling). One-sided: cross < ceiling."""
    d = np.array([c - x for x, c in pairs])
    return {'measure': label, 'n_cells': len(d), 'n_cross_below_ceiling': int((d > 0).sum()),
            'p_sign_one_sided': float(binomtest(int((d > 0).sum()), len(d), 0.5, alternative='greater').pvalue),
            'p_wilcoxon_one_sided': float(wilcoxon(d, alternative='greater').pvalue),
            'fraction_of_ceiling_median': float(np.median([x / c for x, c in pairs])),
            'fraction_of_ceiling_range': [float(min(x / c for x, c in pairs)), float(max(x / c for x, c in pairs))]}


def exact_perm_test(a, b):
    pooled = np.array(a + b)
    obs = np.mean(b) - np.mean(a)
    diffs = []
    for idx in itertools.combinations(range(len(pooled)), len(b)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(idx)] = True
        diffs.append(pooled[mask].mean() - pooled[~mask].mean())
    diffs = np.array(diffs)
    return float(obs), float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12)), len(diffs)


def group_test(cells, key):
    ct22 = [v[key] for (d, _), v in cells.items() if d.startswith('CT22')]
    cb = [v[key] for (d, _), v in cells.items() if d == 'CB']
    if len(ct22) < 2 or len(cb) < 2:
        return None
    diff, p_perm, n_perm = exact_perm_test(ct22, cb)
    return {'metric': key, 'n_ct22_cells': len(ct22), 'n_cb_cells': len(cb),
            'mean_ct22': float(np.mean(ct22)), 'mean_cb': float(np.mean(cb)),
            'cb_minus_ct22': diff, 'p_perm_two_sided': p_perm, 'n_permutations': n_perm,
            'p_mwu_two_sided': float(mannwhitneyu(ct22, cb, alternative='two-sided').pvalue)}


def main():
    cells = load_cells()
    cka = {tuple(r['cell'].split('/')): r for r in json.load(open(ANALYSIS / 'cka.json'))} if (ANALYSIS / 'cka.json').exists() else {}
    for key, v in cells.items():
        if key in cka:
            v.update({k: cka[key][k] for k in ('cka_cross', 'cka_ceiling', 'cka_pre_vs_vcd', 'cka_pre_vs_ccd')
                      if k in cka[key]})
    ceiling = {}
    for tag, keep in (('all_cells', lambda d: True), ('without_TR', lambda d: d != 'CT22-TR')):
        sel = [v for (d, _), v in cells.items() if keep(d)]
        ceiling[tag] = {'rsa': ceiling_test([(v['rsa'], v['ceiling']) for v in sel], 'rsa'),
                        'n_seed_separated': sum(v['seed_separated'] for v in sel)}
        if all('cka_cross' in v for v in sel):
            ceiling[tag]['cka'] = ceiling_test([(v['cka_cross'], v['cka_ceiling']) for v in sel], 'cka')
    res = {'cells': {f'{d}/{e}': v for (d, e), v in sorted(cells.items())},
           'primary_ceiling_test': ceiling,
           'secondary_ct22_vs_cb': group_test(cells, 'rsa'),
           'normalised_test_exploratory': group_test(cells, 'normalised')}
    res['paired_by_encoder'] = {
        e: {'ct22_en': cells[('CT22-EN', e)]['rsa'], 'cb': cells[('CB', e)]['rsa'],
            'cb_minus_ct22_en': cells[('CB', e)]['rsa'] - cells[('CT22-EN', e)]['rsa']}
        for e in {e for _, e in cells} if ('CT22-EN', e) in cells and ('CB', e) in cells}

    tpath = ANALYSIS / 'transfer' / 'transfer_results.json'
    if tpath.exists():
        pts = []
        for t in json.load(open(tpath)):
            key = (t['dataset'], t['encoder'])
            if t['system'] == 'ccd_seq' and key in cells:
                pts.append({'cell': f'{key[0]}/{key[1]}', 'rsa': cells[key]['rsa'],
                            'seq_minus_ccd_roc_auc': t['roc_auc']['diff'],
                            'seq_minus_ccd_f1_pos': t['f1_pos']['diff'],
                            'verdict_roc_auc': t['roc_auc']['verdict']})
        res['rsa_vs_transfer_descriptive'] = {
            'points': pts,
            'spearman_rsa_vs_roc_auc_gain': (float(spearmanr([p['rsa'] for p in pts],
                                                             [p['seq_minus_ccd_roc_auc'] for p in pts])[0])
                                             if len(pts) >= 3 else None)}

    out = ANALYSIS / 'stats.json'
    json.dump(res, open(out, 'w'), indent=2)
    print(json.dumps({k: res[k] for k in ('primary_ceiling_test', 'secondary_ct22_vs_cb', 'normalised_test_exploratory')}, indent=1))


if __name__ == '__main__':
    main()
