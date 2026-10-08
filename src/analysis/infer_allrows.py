"""
re-run active conditional-CCD checkpoints on the FULL test split
(V=0 rows included) and write runs/<run>/pred_test_ccd_allrows.csv. No training.

    python src/analysis/infer_allrows.py --datasets CT22-EN CB
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.analysis.rsa import active_runs
from src.config import Config
from src.data.loader import ClaimDataset, load_split
from src.models.classifier import ClaimCheckWorthinessClassifier
from src.training.train import CKPT_ROOT, ENCODERS, RESULTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['CT22-EN', 'CB'])
    args = p.parse_args()
    cfg = Config.load_experiment_config(str(PROJECT_ROOT / 'src' / 'config' / 'experiment_config.json'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ds in args.datasets:
        records = load_split(ds, 'test')
        for enc in ENCODERS:
            runs = active_runs(ds, 'ccd', enc)
            if not runs:
                continue
            model_path = cfg['models'][enc]
            tok = AutoTokenizer.from_pretrained(model_path)
            loader = DataLoader(ClaimDataset(records, tok, 128), batch_size=64, shuffle=False)
            for r in runs:
                out = RESULTS / 'runs' / r['run'] / 'pred_test_ccd_allrows.csv'
                if out.exists():
                    continue
                model = ClaimCheckWorthinessClassifier(model_path, use_source_features=False)
                model.load_state_dict(torch.load(CKPT_ROOT / r['run'] / 'model.pt', map_location='cpu'))
                model.to(device).eval()
                probs, preds = [], []
                for b in loader:
                    logits = model(b['input_ids'].to(device), b['attention_mask'].to(device))
                    probs += torch.softmax(logits, -1)[:, 1].float().cpu().tolist()
                    preds += logits.argmax(-1).cpu().tolist()
                with open(out, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['id', 'verifiable', 'check_worthy', 'pred', 'prob_pos'])
                    for rec, pr, pb in zip(records, preds, probs):
                        w.writerow([rec['id'], rec['verifiable'], rec['check_worthy'], pr, f'{pb:.6f}'])
                # Consistency check: V=1 rows must reproduce the saved conditional predictions.
                with open(RESULTS / 'runs' / r['run'] / 'pred_test_ccd.csv') as f:
                    saved = {row['id']: int(row['pred']) for row in csv.DictReader(f)}
                mism = sum(1 for rec, pr in zip(records, preds) if rec['id'] in saved and saved[rec['id']] != pr)
                print(f"{r['run']}: {len(records)} rows -> {out.name}; V=1 pred mismatches vs saved: {mism}",
                      flush=True)
                del model
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
