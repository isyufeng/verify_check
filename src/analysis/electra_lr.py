"""
ELECTRA learning-rate selection: seed 42 on CT22-EN/CB x VCD/CCD for each
lr in {1e-5, 2e-5}. Winner = most non-collapsed runs out of 4; ties broken by the mean
dev F1-macro over the 4 runs. Writes results/analysis/electra_lr_selection.json.
"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / 'results'
CELLS = [(ds, t) for ds in ('CT22-EN', 'CB') for t in ('vcd', 'ccd')]
GRID = (1e-5, 2e-5)


def main():
    summary = {}
    for lr in GRID:
        runs = []
        for ds, t in CELLS:
            m = json.load(open(RESULTS / 'runs' / f'{ds}__{t}__electra-large__lr{lr:g}__s42' / 'metrics.json'))
            runs.append({'cell': f'{ds}/{t}', 'collapsed': m['collapsed'],
                         'dev_f1_macro': m['heads'][t]['dev']['f1_macro'],
                         'dev_auc': m['heads'][t]['dev']['auc']})
        summary[f'{lr:g}'] = {
            'runs': runs,
            'n_ok': sum(not r['collapsed'] for r in runs),
            'mean_dev_f1_macro': sum(r['dev_f1_macro'] for r in runs) / len(runs),
        }
    best = max(GRID, key=lambda lr: (summary[f'{lr:g}']['n_ok'], summary[f'{lr:g}']['mean_dev_f1_macro']))
    out = {'rule': 'max n_ok, tie-break mean dev f1_macro', 'selected_lr': best, 'per_lr': summary}
    path = RESULTS / 'analysis' / 'electra_lr_selection.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, 'w'), indent=2)
    for lr, s in summary.items():
        print(f"lr={lr}: n_ok={s['n_ok']}/4 mean_dev_f1m={s['mean_dev_f1_macro']:.4f} "
              + ' '.join(f"{r['cell']}:{'X' if r['collapsed'] else 'ok'}({r['dev_f1_macro']:.3f})" for r in s['runs']))
    print(f"selected lr = {best:g}")


if __name__ == '__main__':
    main()
