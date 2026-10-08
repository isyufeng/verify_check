"""
Per-token Integrated Gradients for single-task classifiers.

Target: P(label=1). Baseline: every non-special token replaced by the pad token
([CLS]/[SEP] kept). Raw per-token attributions are stored so any top-k overlap
metric can be recomputed later without the checkpoint.
"""
from pathlib import Path

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients


def _forward_pos_prob(model):
    def fn(input_ids, attention_mask):
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.softmax(logits, dim=-1)[:, 1]
    return fn


def attribute_records(model, tokenizer, records, device, out_path, n_steps=50, max_length=128):
    model.eval()
    lig = LayerIntegratedGradients(_forward_pos_prob(model), model.encoder.embeddings)

    token_ids, attrs, special, offsets, probs = [], [], [], [0], []
    for r in records:
        enc = tokenizer(r['text'], truncation=True, max_length=max_length, return_tensors='pt',
                        return_special_tokens_mask=True)
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        spec = enc['special_tokens_mask'][0].bool()
        baseline = ids.clone()
        baseline[0, ~spec.to(device)] = tokenizer.pad_token_id

        a = lig.attribute(inputs=ids, baselines=baseline, additional_forward_args=(mask,),
                          n_steps=n_steps, internal_batch_size=n_steps)
        with torch.no_grad():
            probs.append(float(_forward_pos_prob(model)(ids, mask)[0]))

        token_ids.append(ids[0].cpu().numpy())
        attrs.append(a.sum(dim=-1)[0].detach().float().cpu().numpy())
        special.append(spec.numpy())
        offsets.append(offsets[-1] + ids.shape[1])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ids=np.array([r['id'] for r in records]),
        verifiable=np.array([r['verifiable'] for r in records], dtype=np.int8),
        check_worthy=np.array([r['check_worthy'] for r in records], dtype=np.int8),
        offsets=np.array(offsets, dtype=np.int64),
        token_ids=np.concatenate(token_ids).astype(np.int32),
        attr=np.concatenate(attrs).astype(np.float32),
        special=np.concatenate(special),
        pos_prob=np.array(probs, dtype=np.float32),
        n_steps=n_steps,
    )
    print(f"  attribution: {len(records)} samples -> {out_path}")
