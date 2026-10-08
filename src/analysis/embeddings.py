"""
[CLS] embedding export .

Fine-tuned runs call `dump_embeddings` at the end of training. The CLI dumps the
pretrained (not fine-tuned) encoders as the control baseline:

    python src/analysis/embeddings.py --encoder roberta --datasets CT22-EN CB
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.loader import load_split
from src.config import Config

RESULTS = Path(__file__).resolve().parent.parent.parent / 'results'
PAIRED = ('CT22-EN', 'CB')


def export_sets(dataset):
    """CT22-EN/CB models are exported on both datasets (enables cross-input RSA)."""
    sources = PAIRED if dataset in PAIRED else (dataset,)
    return [(ds, split) for ds in sources for split in ('dev', 'test')]


@torch.no_grad()
def extract_cls(encoder, tokenizer, records, device, batch_size=64, max_length=128):
    encoder.eval()
    out = []
    for i in range(0, len(records), batch_size):
        batch = tokenizer([r['text'] for r in records[i:i + batch_size]], truncation=True,
                          max_length=max_length, padding='longest', return_tensors='pt').to(device)
        h = encoder(input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']).last_hidden_state[:, 0, :]
        out.append(h.float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def dump_embeddings(encoder, tokenizer, sets, out_dir, device, max_length=128, limit=None):
    """sets: list of (dataset, split), e.g. export_sets(run_dataset)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds, split in sets:
        records = load_split(ds, split)[:limit]
        cls = extract_cls(encoder, tokenizer, records, device, max_length=max_length)
        np.savez_compressed(
            out_dir / f'{ds}_{split}.npz', cls=cls,
            ids=np.array([r['id'] for r in records]),
            verifiable=np.array([r['verifiable'] for r in records], dtype=np.int8),
            check_worthy=np.array([r['check_worthy'] for r in records], dtype=np.int8),
        )
        print(f"  embeddings {ds}/{split}: {cls.shape} -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', required=True)
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--config', default=str(Path(__file__).resolve().parent.parent
                                           / 'config' / 'experiment_config.json'))
    p.add_argument('--limit', type=int, default=None, help='Smoke test: rows per split')
    args = p.parse_args()

    cfg = Config.load_experiment_config(args.config)
    model_path = cfg['models'][args.encoder]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = AutoModel.from_pretrained(model_path).to(device)
    tag = 'smoke_' if args.limit else ''
    sets = list(dict.fromkeys(s for ds in args.datasets for s in export_sets(ds)))
    dump_embeddings(encoder, tokenizer, sets,
                    RESULTS / 'embeddings' / f'{tag}pretrained_{args.encoder}', device,
                    limit=args.limit)
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")


if __name__ == '__main__':
    main()
