"""
Collapse rule and seed bookkeeping .

A run is collapsed if, on the DEV split of its selected checkpoint:
  1. predicted-positive rate < 5% or > 95%, or
  2. AUC < 0.55.
For MTL, either head collapsing marks the run collapsed.

CLI: summarise results/runs/*/metrics.json into the run registry and report,
per cell, which seeds are active and which backup seeds still need to be run.

    python src/analysis/collapse.py
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PRIMARY_SEEDS = (42, 123, 456)
BACKUP_SEEDS = (789, 999)
N_REQUIRED = 3

PRED_RATE_BOUNDS = (0.05, 0.95)
MIN_DEV_AUC = 0.55

RESULTS = Path(__file__).resolve().parent.parent.parent / 'results'


def excluded_cells(path=RESULTS / 'excluded_cells.json'):
    """Cells excluded by decision, keyed like the registry plan."""
    if not Path(path).exists():
        return {}
    return {(c['dataset'], c['task'], c['encoder'], float(c['lr'])): c['reason']
            for c in json.load(open(path))}


def collapse_check(labels, probs, preds):
    labels, probs, preds = map(np.asarray, (labels, probs, preds))
    pos_rate = float(preds.mean())
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float('nan')
    reasons = []
    if not PRED_RATE_BOUNDS[0] <= pos_rate <= PRED_RATE_BOUNDS[1]:
        reasons.append(f'degenerate_pred_rate={pos_rate:.3f}')
    if not auc >= MIN_DEV_AUC:
        reasons.append(f'dev_auc={auc:.3f}')
    return {'collapsed': bool(reasons), 'reasons': reasons,
            'dev_pred_pos_rate': round(pos_rate, 4), 'dev_auc': round(auc, 4)}


def active_seeds(seed_status):
    """seed_status: {seed: collapsed(bool)} for seeds already run.

    Returns (active, next_seed, excluded). Backups are consumed strictly in order,
    one per collapsed primary; no seed is skipped or cherry-picked.
    """
    active = []
    for seed in PRIMARY_SEEDS + BACKUP_SEEDS:
        if len(active) == N_REQUIRED:
            break
        if seed not in seed_status:
            return active, seed, False
        if not seed_status[seed]:
            active.append(seed)
    return active, None, len(active) < N_REQUIRED


REGISTRY_FIELDS = ['run', 'dataset', 'task', 'encoder', 'lr', 'seed', 'best_epoch',
                   'dev_f1_macro', 'dev_auc', 'dev_pred_pos_rate', 'test_f1_macro',
                   'test_auc', 'collapsed', 'reasons', 'active']


def build_registry(runs_dir=RESULTS / 'runs', out=RESULTS / 'run_registry.csv'):
    rows = []
    for mf in sorted(Path(runs_dir).glob('*/metrics.json')):
        m = json.load(open(mf))
        heads = m['heads']
        primary = 'vcd' if m['task'] == 'vcd' else 'ccd'
        rows.append({
            'run': m['run'], 'dataset': m['dataset'], 'task': m['task'],
            'encoder': m['encoder'], 'lr': m['hparams']['lr'], 'seed': m['seed'],
            'best_epoch': m['best_epoch'],
            'dev_f1_macro': heads[primary]['dev']['f1_macro'],
            'dev_auc': heads[primary]['dev']['auc'],
            'dev_pred_pos_rate': heads[primary]['collapse']['dev_pred_pos_rate'],
            'test_f1_macro': heads[primary]['test']['f1_macro'],
            'test_auc': heads[primary]['test']['auc'],
            'collapsed': m['collapsed'],
            'reasons': ';'.join(r for h in heads.values() for r in h['collapse']['reasons']),
        })

    cells = defaultdict(dict)
    for r in rows:
        cells[(r['dataset'], r['task'], r['encoder'], r['lr'])][r['seed']] = r['collapsed']
    plan = {}
    excluded = excluded_cells()
    sel = RESULTS / 'analysis' / 'electra_lr_selection.json'
    electra_lr = json.load(open(sel))['selected_lr'] if sel.exists() else None
    for cell, status in cells.items():
        if cell[2] == 'electra-large' and electra_lr is not None and float(cell[3]) != electra_lr:
            plan[cell] = ([], None, False)  # lr-selection-only runs
            continue
        act, nxt, excl = active_seeds(status)
        if (cell[0], cell[1], cell[2], float(cell[3])) in excluded:
            act, nxt, excl = [], None, True
        plan[cell] = (act, nxt, excl)
    for r in rows:
        act = plan[(r['dataset'], r['task'], r['encoder'], r['lr'])][0]
        r['active'] = r['seed'] in act

    out.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent jobs rebuild the registry; write-then-rename keeps it consistent.
    tmp = out.with_name(f'.{out.name}.{os.getpid()}')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    return rows, plan


if __name__ == '__main__':
    rows, plan = build_registry()
    print(f"{len(rows)} runs -> {RESULTS / 'run_registry.csv'}")
    for (ds, task, enc, lr), (act, nxt, excl) in sorted(plan.items(), key=str):
        if excl:
            state = 'EXCLUDED'
        elif not act and nxt is None:
            state = 'lr-selection only'
        else:
            state = 'complete' if nxt is None else f'next seed {nxt}'
        print(f"{ds:8s} {task:3s} {enc:18s} lr={lr:<9g} active={act} -> {state}")
