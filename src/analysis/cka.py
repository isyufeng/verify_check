"""
Linear CKA counterpart of the RSA suite (robustness check): cross-task (rank-paired seeds),
within-task across seeds (ceiling) and pretrained vs fine-tuned (floor), v1only (CCD test set).

    python src/analysis/cka.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.analysis.collapse import excluded_cells
from src.analysis.rsa import active_runs, load_emb
from src.analysis.rsa_suite import DATASETS, ENCODERS, linear_cka

RESULTS = Path(__file__).resolve().parent.parent.parent / 'results'


def main():
    excluded = {(d, t, e) for d, t, e, _ in excluded_cells()}
    out = []
    for ds, enc in itertools.product(DATASETS, ENCODERS):
        if (ds, 'vcd', enc) in excluded or (ds, 'ccd', enc) in excluded:
            continue
        rv, rc = active_runs(ds, 'vcd', enc), active_runs(ds, 'ccd', enc)
        if len(rv) < 3 or len(rc) < 3:
            continue
        V = [load_emb(r['run'], ds) for r in rv]
        C = [load_emb(r['run'], ds) for r in rc]
        m = V[0][2] != -1
        A, B = [x[0][m] for x in V], [x[0][m] for x in C]
        pre_p = RESULTS / 'embeddings' / f'pretrained_{enc}' / f'{ds}_test.npz'
        pre = np.load(pre_p)['cls'].astype(np.float32)[m] if pre_p.exists() else None
        rec = {'cell': f'{ds}/{enc}', 'n': int(m.sum()),
               'cka_cross': float(np.mean([linear_cka(a, b) for a, b in zip(A, B)])),
               'cka_within_vcd': float(np.mean([linear_cka(a, b) for a, b in itertools.combinations(A, 2)])),
               'cka_within_ccd': float(np.mean([linear_cka(a, b) for a, b in itertools.combinations(B, 2)]))}
        if pre is not None:
            rec['cka_pre_vs_vcd'] = float(np.mean([linear_cka(pre, a) for a in A]))
            rec['cka_pre_vs_ccd'] = float(np.mean([linear_cka(pre, b) for b in B]))
        rec['cka_ceiling'] = (rec['cka_within_vcd'] + rec['cka_within_ccd']) / 2
        out.append(rec)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    path = RESULTS / 'analysis' / 'cka.json'
    json.dump(out, open(path, 'w'), indent=2)


if __name__ == '__main__':
    main()
