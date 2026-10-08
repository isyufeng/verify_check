# Verifiability vs. check-worthiness: representation analysis

Code for training verifiable-claim detection (VCD) and check-worthy claim detection (CCD) encoders
and comparing them through representational similarity (RSA, linear CKA), linear probing,
transfer/gating experiments, seed stability and token attribution.

## Setup

```bash
pip install -r requirements.txt
```

Pretrained encoders are loaded from the Hugging Face Hub (`src/config/experiment_config.json`):
RoBERTa-large, BERT-large (uncased), ELECTRA-large (discriminator) and XLM-RoBERTa-large.

## Data

The processed splits used in all experiments are included under `data/raw/`:

- `data/raw/ct22_multilang/<lang>/ct22_<lang>_{train,dev,test}.jsonl`: CheckThat! 2022 Task 1
  (Arabic, Bulgarian, Dutch, English, Turkish), with Task 1B (`verifiable`) and Task 1A (`check_worthy`)
  joined on `tweet_id`. `check_worthy` is null where the check-worthiness question was not asked (V=0; some language
  releases label these rows CW=0 instead). CCD uses V=1 rows only.
- `data/raw/claimbuster/claimbuster_{train,dev,test}.jsonl`: ClaimBuster with V = factual (UFS or CFS)
  and CW = CFS, in a fixed stratified split.

`src/data/loader.py` checks every split against the expected sample counts.
Both datasets remain subject to the terms of their original releases.

To rebuild these files from the original releases, place the CheckThat! 2022 Task 1 data under
`data/raw/clef2022-checkthat-lab/` and `claimbuster_all.tsv` (Hugging Face `Nithiwat/claimbuster`)
under `data/raw/claimbuster/`, then run:

```bash
bash scripts/prepare_data.sh
```

## Training

One run is one (dataset, task, encoder, learning rate, seed):

```bash
bash scripts/train_one.sh CT22-EN vcd roberta 42
bash scripts/train_one.sh CB ccd roberta 42
bash scripts/train_one.sh CB ccd_seq roberta 42 --init_vcd_seed 42   # intermediate-task fine-tuning
bash scripts/train_one.sh CT22-EN mtl roberta 42                      # shared encoder, two heads
bash scripts/train_one.sh CB ccd_full roberta 42                      # CCD on all samples, V=0 as CW=0
```

Tasks: `vcd`; `ccd` (V=1 samples only); `ccd_full`; `ccd_seq`; `mtl` (RoBERTa only).
Hyperparameters are fixed in `src/training/train.py`; ELECTRA's learning rate is selected
from {1e-5, 2e-5} by `src/analysis/electra_lr.py`.
Each run selects its epoch by dev macro-F1 and writes predictions and metrics to `results/runs/`,
[CLS] embeddings to `results/embeddings/`, and, for single-task runs, Integrated Gradients
attributions to `results/attribution/`. Model weights go to `$CKPT_ROOT` (default `saved_models/`).
A run is marked collapsed if its dev predicted-positive rate is outside [5%, 95%] or its dev ROC-AUC
is below 0.55 (`src/analysis/collapse.py`); collapsed runs are replaced by an additional seed.

The full grid is listed in `scripts/train_all.sh`.

## Analysis

```bash
bash scripts/run_analysis.sh
```

| Script | Analysis |
|---|---|
| `embeddings.py` | [CLS] embeddings of pretrained encoders (RSA/CKA floor) |
| `infer_allrows.py` | conditional CCD scores on the full test split (for gating) |
| `collapse.py` | collapse rule and run registry |
| `rsa.py`, `rsa_suite.py` | cross-task RSA, same-task ceiling, pretrained floor, bootstrap CIs |
| `cka.py` | linear CKA counterpart of the RSA suite |
| `probe.py` | linear probes for CW and V on frozen [CLS] representations |
| `transfer.py` | intermediate-task FT, MTL, proxy scoring and gating vs. single-task CCD |
| `stats.py` | cell-level tests (cross-task vs. ceiling; CT22 vs. ClaimBuster) |
| `stability.py` | seed stability of representations and predictions |
| `attribution.py`, `attribution_overlap.py` | Integrated Gradients and length-controlled VCD-CCD overlap |

Outputs are written to `results/analysis/`.

## License

Apache 2.0 (see `LICENSE`).
