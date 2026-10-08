"""
Training entry point.

One run = (dataset, task, encoder, lr, seed). Each run:
  1. trains with class-weighted CE, selecting the epoch on dev F1-macro
     (MTL: the CCD head's dev F1-macro);
  2. saves only the model state_dict to CKPT_ROOT (env var, default saved_models/);
  3. writes dev/test predictions, metrics and the dev-based collapse verdict to results/runs/;
  4. exports [CLS] embeddings (results/embeddings/) and, for single-task runs,
     per-token IG attributions on the full test split (results/attribution/).

Usage:
    python src/training/train.py --dataset CB --task ccd --encoder roberta --seed 42
    python src/training/train.py --dataset CT22-EN --task mtl --encoder roberta --seed 42 --smoke
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.attribution import attribute_records
from src.analysis.collapse import build_registry, collapse_check
from src.analysis.embeddings import dump_embeddings, export_sets
from src.config import Config
from src.data.loader import ClaimDataset, DATASET_PATHS, load_split, task_view
from src.evaluation.evaluator import Evaluator
from src.models.classifier import ClaimCheckWorthinessClassifier
from src.models.mtl_classifier import MTLClaimClassifier
from src.training.training import compute_class_weights, get_optimizer_and_scheduler, set_seed

CKPT_ROOT = Path(os.environ.get('CKPT_ROOT', PROJECT_ROOT / 'saved_models'))
RESULTS = PROJECT_ROOT / 'results'

BASE_HPARAMS = dict(lr=2e-5, epochs=5, weight_decay=0.01, batch_size=16, max_length=128,
                    warmup_ratio=0.1, max_grad_norm=1.0)
XLMR_HPARAMS = {
    'vcd': dict(lr=1.53e-5, epochs=5, weight_decay=0.092),
    'ccd': dict(lr=6.839497720163598e-06, epochs=9, weight_decay=0.002755502291134793),
}
ELECTRA_LR_GRID = (1e-5, 2e-5)
ENCODERS = ('roberta', 'bert-large', 'electra-large', 'xlm-roberta-large')

SMOKE_ROWS = {'train': 96, 'dev': 48, 'test': 48}


HEAD = {'vcd': 'vcd', 'ccd': 'ccd', 'ccd_full': 'ccd', 'ccd_seq': 'ccd'}


def resolve_hparams(encoder, task, lr):
    if task == 'mtl' and encoder != 'roberta':
        raise ValueError('MTL is only defined for RoBERTa-large in this setup')
    hp = dict(BASE_HPARAMS)
    if encoder == 'xlm-roberta-large':
        hp.update(XLMR_HPARAMS[HEAD[task]])
    if encoder == 'electra-large':
        if lr not in ELECTRA_LR_GRID:
            raise ValueError(f'ELECTRA requires --lr in {ELECTRA_LR_GRID}')
        hp['lr'] = lr
    elif lr is not None and lr != hp['lr']:
        raise ValueError(f'--lr override is only allowed for ELECTRA')
    return hp


def heads_for(task):
    return ('vcd', 'ccd') if task == 'mtl' else (HEAD[task],)


def compute_loss(task, outputs, batch, loss_fns, device):
    if task != 'mtl':
        return loss_fns[HEAD[task]](outputs, batch[f'{HEAD[task]}_label'].to(device))
    vcd_logits, ccd_logits = outputs
    loss = loss_fns['vcd'](vcd_logits, batch['vcd_label'].to(device))
    ccd_labels = batch['ccd_label'].to(device)
    if (ccd_labels != -1).any():
        loss = loss + loss_fns['ccd'](ccd_logits, ccd_labels)
    return loss


@torch.no_grad()
def predict(model, loader, task, device):
    """Returns {head: (labels, probs, preds, row_index)}; MTL CCD rows with label -1 are dropped."""
    model.eval()
    buf = {h: ([], [], []) for h in heads_for(task)}
    for batch in loader:
        out = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        logits = dict(zip(('vcd', 'ccd'), out)) if task == 'mtl' else {HEAD[task]: out}
        for h, lg in logits.items():
            buf[h][0].append(batch[f'{h}_label'].numpy())
            buf[h][1].append(torch.softmax(lg, dim=-1)[:, 1].float().cpu().numpy())
            buf[h][2].append(lg.argmax(dim=-1).cpu().numpy())
    res = {}
    for h, (lab, prob, pred) in buf.items():
        lab, prob, pred = map(np.concatenate, (lab, prob, pred))
        keep = np.flatnonzero(lab != -1)
        res[h] = (lab[keep], prob[keep], pred[keep], keep)
    return res


def metrics_of(labels, probs, preds):
    return Evaluator.compute_metrics(labels.tolist(), preds.tolist(), probs.tolist())


def write_predictions(path, records, labels, probs, preds, keep):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'label', 'pred', 'prob_pos'])
        for i, lab, pr, pb in zip(keep, labels, preds, probs):
            w.writerow([records[i]['id'], int(lab), int(pr), f'{pb:.6f}'])


def run_metadata():
    try:
        commit = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'],
                                         text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'status',
                                              '--porcelain', '--', 'src'], text=True).strip())
    except Exception:
        commit, dirty = 'unknown', None
    return {
        'git_commit': commit, 'src_dirty': dirty,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'host': os.uname().nodename,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE'),
    }


def main():
    p = argparse.ArgumentParser(description='Single-task / MTL training')
    p.add_argument('--dataset', required=True, choices=list(DATASET_PATHS))
    p.add_argument('--task', required=True, choices=['vcd', 'ccd', 'mtl', 'ccd_full', 'ccd_seq'])
    p.add_argument('--init_vcd_seed', type=int, default=None,
                   help='ccd_seq only: seed of the active VCD run whose encoder initialises this run')
    p.add_argument('--encoder', required=True, choices=ENCODERS)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--lr', type=float, default=None, help='ELECTRA only: 1e-5 or 2e-5')
    p.add_argument('--smoke', action='store_true', help='1 epoch on a few rows; separate output dirs')
    p.add_argument('--skip_attribution', action='store_true')
    p.add_argument('--config', default=str(PROJECT_ROOT / 'src' / 'config' / 'experiment_config.json'))
    args = p.parse_args()

    hp = resolve_hparams(args.encoder, args.task, args.lr)
    if args.smoke:
        hp['epochs'] = 1
    if (args.task == 'ccd_seq') != (args.init_vcd_seed is not None):
        p.error('--init_vcd_seed is required for, and only valid with, --task ccd_seq')
    run = f"{args.dataset}__{args.task}__{args.encoder}__lr{hp['lr']:g}__s{args.seed}"
    init_run = None
    if args.task == 'ccd_seq':
        vcd_lr = resolve_hparams(args.encoder, 'vcd', args.lr)['lr']
        init_run = f"{args.dataset}__vcd__{args.encoder}__lr{vcd_lr:g}__s{args.init_vcd_seed}"
        run += f"__init{args.init_vcd_seed}"
    sub = 'smoke' if args.smoke else ''
    run_dir = RESULTS / sub / 'runs' / run
    ckpt_path = CKPT_ROOT / sub / run / 'model.pt'
    if (run_dir / 'metrics.json').exists():
        sys.exit(f'{run_dir} already finished; refusing to overwrite')
    run_dir.mkdir(parents=True, exist_ok=True)
    # Twin jobs may be queued on two partitions; the first to start claims the run.
    try:
        os.close(os.open(run_dir / '.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        print(f'{run} is claimed by another job ({run_dir / ".lock"}); exiting')
        sys.exit(0)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== {run} | {json.dumps(hp)}")
    t0 = time.time()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Config.load_experiment_config(args.config)['models'][args.encoder]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    splits = {s: task_view(load_split(args.dataset, s), args.task) for s in ('train', 'dev', 'test')}
    if args.smoke:
        rng = np.random.RandomState(0)
        splits = {s: [recs[i] for i in sorted(rng.choice(len(recs), SMOKE_ROWS[s], replace=False))]
                  for s, recs in splits.items()}
    print('  sizes: ' + ', '.join(f'{s}={len(r)}' for s, r in splits.items()))

    gen = torch.Generator().manual_seed(args.seed)
    loaders = {
        s: DataLoader(ClaimDataset(recs, tokenizer, hp['max_length']), batch_size=hp['batch_size'],
                      shuffle=(s == 'train'), drop_last=(s == 'train'), num_workers=2,
                      pin_memory=True, generator=gen if s == 'train' else None)
        for s, recs in splits.items()
    }

    if args.task == 'mtl':
        model = MTLClaimClassifier(model_path, use_source_features=False)
    else:
        model = ClaimCheckWorthinessClassifier(model_path, use_source_features=False)
    if init_run is not None:
        vcd_state = torch.load(CKPT_ROOT / init_run / 'model.pt', map_location='cpu')
        enc_state = {k[len('encoder.'):]: v for k, v in vcd_state.items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(enc_state, strict=True)
        print(f"  encoder initialised from {init_run} ({len(enc_state)} tensors)")
    model.to(device)

    loss_fns = {}
    for h in heads_for(args.task):
        key = 'verifiable' if h == 'vcd' else 'check_worthy'
        labels = [r[key] for r in splits['train'] if r[key] != -1]
        w = compute_class_weights(labels, num_classes=2, device=device)
        loss_fns[h] = nn.CrossEntropyLoss(weight=w, ignore_index=-1)
        print(f"  class weights [{h}]: {w.tolist()}")

    optimizer, scheduler = get_optimizer_and_scheduler(
        model, loaders['train'], hp['lr'], hp['epochs'], hp['warmup_ratio'], hp['weight_decay'])
    select_head = 'ccd' if args.task == 'mtl' else HEAD[args.task]

    history, best_score, best_epoch = [], -1.0, None
    for epoch in range(hp['epochs']):
        model.train()
        total, n = 0.0, 0
        for batch in loaders['train']:
            optimizer.zero_grad()
            out = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            loss = compute_loss(args.task, out, batch, loss_fns, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            total += loss.item()
            n += 1
        dev = predict(model, loaders['dev'], args.task, device)
        dev_m = {h: metrics_of(*v[:3]) for h, v in dev.items()}
        score = dev_m[select_head]['f1_macro']
        history.append({'epoch': epoch, 'train_loss': total / max(n, 1),
                        **{f'dev_{h}_{k}': m[k] for h, m in dev_m.items()
                           for k in ('f1_macro', 'f1_score', 'auc')}})
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), ckpt_path)
        print(f"  epoch {epoch}: loss={total / max(n, 1):.4f} "
              + ' '.join(f"dev_{h}_f1m={m['f1_macro']:.4f} auc={m['auc']:.4f}" for h, m in dev_m.items())
              + (' *' if improved else ''), flush=True)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    heads = {}
    for split in ('dev', 'test'):
        for h, (lab, prob, pred, keep) in predict(model, loaders[split], args.task, device).items():
            heads.setdefault(h, {})[split] = metrics_of(lab, prob, pred)
            write_predictions(run_dir / f'pred_{split}_{h}.csv', splits[split], lab, prob, pred, keep)
            if split == 'dev':
                heads[h]['collapse'] = collapse_check(lab, prob, pred)
    collapsed = any(heads[h]['collapse']['collapsed'] for h in heads)

    metrics = {
        'run': run, 'dataset': args.dataset, 'task': args.task, 'encoder': args.encoder,
        'model_path': model_path, 'seed': args.seed, 'init_vcd_run': init_run, 'hparams': hp, 'smoke': args.smoke,
        'best_epoch': best_epoch, 'selection': f'dev {select_head} f1_macro',
        'sizes': {s: len(r) for s, r in splits.items()},
        'heads': heads, 'collapsed': collapsed, 'history': history,
        'checkpoint': str(ckpt_path), 'meta': run_metadata(),
    }
    for h, m in heads.items():
        print(f"  [{h}] dev f1m={m['dev']['f1_macro']:.4f} auc={m['dev']['auc']:.4f} | "
              f"test f1m={m['test']['f1_macro']:.4f} auc={m['test']['auc']:.4f} | "
              f"collapse={m['collapse']}")

    limit = 16 if args.smoke else None
    dump_embeddings(model.encoder, tokenizer, export_sets(args.dataset),
                    RESULTS / sub / 'embeddings' / run, device, hp['max_length'], limit=limit)
    if args.task != 'mtl' and not args.skip_attribution:
        test_all = load_split(args.dataset, 'test')[:4 if args.smoke else None]
        attribute_records(model, tokenizer, test_all, device,
                          RESULTS / sub / 'attribution' / f'{run}.npz', max_length=hp['max_length'])

    metrics['runtime_min'] = round((time.time() - t0) / 60, 1)
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    if not args.smoke:
        build_registry()
    print(f"=== done {run} in {metrics['runtime_min']} min; collapsed={collapsed}")


if __name__ == '__main__':
    main()
