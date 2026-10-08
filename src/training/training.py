"""
Training utilities: seeding, class weights, optimizer and scheduler.
"""
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def compute_class_weights(
        labels: Sequence[int],
        num_classes: int = 2,
        device: Optional[str] = None,
) -> torch.Tensor:
    """sklearn 'balanced' formula: w_c = N / (n_classes * N_c).

    Returns a tensor indexed by class id, suitable for nn.CrossEntropyLoss(weight=...).
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    if (counts == 0).any():
        raise ValueError(f"compute_class_weights: zero-count class in {counts.tolist()}")
    n_total = counts.sum()
    weights = n_total / (num_classes * counts)
    t = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def set_seed(seed: int):
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer_and_scheduler(
        model: nn.Module,
        train_dataloader: DataLoader,
        learning_rate: float,
        epochs: int,
        warmup_ratio: float,
        weight_decay: float
):
    """AdamW without weight decay on biases and LayerNorm weights, with linear warmup and decay."""
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return optimizer, scheduler
