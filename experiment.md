# Experiment: does the PRAGMA foundation model add to a counts baseline?

The experiment that found a significant uplift, and how to reproduce it after applying
`changes_since_personas.diff`. The reproduction also validates the diff: every check below has an expected value.

## Question

A bank's propensity models usually start from **counts**: per event type, how often it happened in the history and
in the last 30 days, days since the last event, and the profile's numbers (33 features).

Does a customer embedding from a PRAGMA-style foundation model, pretrained on the raw event histories without
labels, predict what a customer does in the next 7 days better than these counts?

## Setup

| Part | Choice |
|---|---|
| Data | 20,000 synthetic customers from 8 personas (`pipelines/01_bronze.py 20000`, seed 0): 1,561,371 events, 78 per customer, from three messy raw exports (CRM, app, core banking) |
| Split | 16,000 train customers, 4,000 test customers (seeded). Tokenizer and model only see train customers |
| Tokenizer | PRAGMA key-value-time: 23 fields, 462 tokens, numbers as 64 percentile buckets plus zero (`pipelines/03_gold.py`) |
| Pretraining | Masked modelling (15% tokens, 10% events, 10% customer-field pairs), Muon + AdamW, bf16. Two sizes, 2,000 steps each, token budget 32,768 (`pipelines/04_train.py`): **nano** (d 64) and **small** = PRAGMA-S (d 192, about 10M parameters) |
| Snapshots | Every customer on 8 dates, 14 days apart. The history before the date goes into the model; the label is what happens in the next 7 days |
| Tasks | 9 tasks. Contact tasks are conditional, e.g. "opens the push it gets" only counts snapshots where a push arrives |
| Probe (pipeline 5) | Logistic regression per task (StandardScaler + lbfgs), fit on train customers' snapshots and scored on test customers'. Three inputs: counts, embedding, counts + embedding |
| LoRA (pipeline 6) | PRAGMA's main downstream method: rank 8, α 8 on QKV and both feed-forward layers, a head on the customer embedding, Adam lr 1e-3, 2 epochs, 4 snapshot dates. A fresh copy of the small model per task |
| Significance | Paired bootstrap over **test customers** (their snapshots stay together), 1,000 resamples, two-sided p, Holm correction over all comparisons. Significant = Holm p < 0.05 and the interval above zero |
| Hardware | 1× NVIDIA A10 (Lambda Cloud), torch 2.11.0+cu128, scikit-learn 1.9.1, numpy 2.4.6, Python 3.11 |

## Result: significant uplift (pipeline 5, logistic regression on the embedding)

ROC-AUC on 32,000 test snapshots (8 dates). Uplift = AUC(counts + embedding) − AUC(counts) [95% interval].

| Task | Test rows (positives) | Counts | small: + embedding | small: uplift | nano: uplift |
|---|---|---|---|---|---|
| opens the push it gets | 7,000 (607) | 0.667 | 0.698 | **+0.031 [+0.017, +0.044] significant** | **+0.018 [+0.006, +0.029] significant** |
| visits the app | 32,000 (8,937) | 0.719 | 0.724 | **+0.005 [+0.003, +0.007] significant** | **+0.004 [+0.002, +0.006] significant** |
| converts (contract signed) | 32,000 (607) | 0.765 | 0.775 | +0.010 [+0.001, +0.019], not after Holm | +0.004 [−0.003, +0.010] |
| applies for a product | 32,000 (704) | 0.682 | 0.682 | −0.000 [−0.011, +0.010] | +0.004 [−0.005, +0.013] |
| terminates a contract | 32,000 (65) | 0.855 | 0.882 | +0.027 [−0.003, +0.058] | +0.028 [+0.004, +0.056] |
| opens the email it gets | 3,849 (363) | 0.630 | 0.621 | −0.009 [−0.031, +0.012] | +0.015 [−0.003, +0.033] |
| answers the call it gets | 719 (173) | 0.707 | 0.717 | +0.010 [−0.013, +0.032] | +0.015 [+0.001, +0.032] |
| clicks the banner it sees | 3,039 (260) | 0.603 | 0.619 | +0.017 [−0.012, +0.044] | +0.021 [+0.003, +0.040] |
| unsubscribes after an email | too few positives | | | | |

- **Push opens:** on small, the embedding alone also beats counts: 0.698 vs 0.667, +0.030 [+0.013, +0.048], significant.
- **Model size:** the bigger model gives the bigger uplift on push opens, +0.031 (small) against +0.018 (nano), as PRAGMA reports for size.
- **Full tables:** `gpu_results/reports/downstream_small.md` and `downstream_nano.md`.

## LoRA fine-tuning per task (pipelines 6 and 7, small)

ROC-AUC on the test customers' snapshots (4 dates, so different rows and counts AUCs than above).

| Task | LoRA | Counts | Counts + embedding probe | LoRA − counts |
|---|---|---|---|---|
| opens the push it gets | 0.669 | 0.616 | 0.680 | +0.054 [+0.012, +0.097], not significant after Holm |
| answers the call it gets | 0.726 | 0.693 | 0.664 | +0.033 [−0.009, +0.074] |
| converts (contract signed) | 0.780 | 0.771 | 0.778 | +0.009 [−0.010, +0.033] |
| visits the app | 0.723 | 0.721 | 0.724 | +0.003 [−0.000, +0.005] |
| applies for a product | 0.690 | 0.692 | 0.696 | −0.002 |
| opens the email it gets | 0.629 | 0.630 | 0.622 | −0.002 |
| clicks the banner it sees | 0.625 | 0.626 | 0.623 | −0.001 |
| terminates a contract | 0.683 | 0.815 | 0.856 | **−0.132**: 153 train positives are too few to fine-tune |

Other KPIs (pipeline 7) show one more significant LoRA gain, on conversion:
- **PR-AUC:** 0.295 vs 0.186 for counts, +0.109 [+0.069, +0.145]; log-loss and best F1 are also better.
- **Where the gain sits:** all of it is in the top scores. Of the 80 highest-scored test snapshots (0.5%), 68 convert with LoRA against 37 with counts; from a 2% budget up, the methods tie.
- **Full tables:** `gpu_results/reports/finetune_small.md`, `compare_small.md` and `gains_small.png`.

## How to reproduce

### 1. Apply the diff

From `journeyModeller/` as it was before the change:

```bash
patch -p1 < changes_since_personas.diff
```

Or with git; switch off line-ending conversion, or every file gets CRLF and the checksums below fail:

```bash
git -c core.autocrlf=false apply changes_since_personas.diff
```

I tested both on the pre-change files (from Claude Code's file history). Both give files byte-identical to the
project.

### 2. Check that every file is exactly right

```bash
sha256sum -c <<'EOF'
148ca1aee99d6ab23bdd51e7226d7a88c9d66ba82626d7a88caa7a6b1ea10d91  experiments/ablation.py
5cc030d8dfee2dbf39abf9428c4484f52fdd112cc7ef72a6b9e5b0b3e244da15  experiments/gpu.sh
99e8d0b2d563f08804f59a3f68216110ea1a692fbf8f6bb47bf46b9947f280b3  pipelines/01_bronze.py
95a50fcd7c9397b604614c5280e210222333f13a8ab7a00190694316bd66da9e  pipelines/02_silver.py
fc66ec9d42ca4aeea27b2442aa0339ae5d0066de30c4cd62b008d30486e3f5fc  pipelines/03_gold.py
9b1260697b3084a31a172f1dca372733aa9b860f861a9b46a9d73fb26f4c19b9  pipelines/04_train.py
e7bf0c5850f932e419ed2c76293fa5dc36a5ce4eec215d6bc539f9deb0cc4711  pipelines/05_downstream.py
492d97507979b8d1f67af6ab2c834e8601905981d569403fd6b52375c8b8c971  pipelines/06_finetune.py
8ab605de5ce3237c041d992ab27243b4df4b460f61e7478a6333eafab509b13f  pipelines/07_compare.py
5c87f9a9bdf5a02acf197ba3d59ca899ff0c32d176a26b94cc5c97cf9bf6e9a3  pipelines/common.py
8d9217005bf40c59a92eb973583eeaf9e8cf06ec6fe4b89224c56dc9cd537544  pipelines/downstream.py
cae7d745a3b3599cd94a9e290b56ad94c5d8466f4fcbcb2952e1d1766b099503  pragmaLib/__init__.py
0696fe52537d6ef9f41ab39c7468ea53ce1dff4d3a953a89db7b8cd06a679770  pragmaLib/check.py
093d63a3530a8ef69f843d598972f6dd7123e5231581be8ce37e87b2a948a3c1  pragmaLib/lora.py
52222b97ec89f5ad7f6a59ef7fd401d349612e37e055aeb575961f04c7d88c89  pragmaLib/model.py
fe2f1bc0f53b457e318e68a0de510dbb62c826236eb876c7d52242b271339d83  pragmaLib/pretrain.py
ad72baa7072a5d2353dde307ca962bb9471918fe6645a333e25eaac0ee967b7a  pragmaLib/tokenizer.py
c00680353a04881e4a238e8fa968a553ee0fbbadaf76f1be6896ba5d415aada4  requirements.txt
EOF
```

All 18 lines should say `OK`. On macOS use `shasum -a 256 -c`.

### 3. CPU checks, a few minutes, no GPU needed

Needs Python ≥ 3.10 and `pip install -r requirements.txt`. torch must be ≥ 2.9, which has `torch.optim.Muon`.

```bash
python -m pragmaLib.check
```

This trains a tiny model for 150 steps on the CPU. Expected last line, about a minute on a laptop:

```
pragmaLib ok: hidden tokens right 64%, 27% of tokens scored per batch
```

The data is fully deterministic, and identical on Windows and Linux. These steps overwrite `data/`:

```bash
python pipelines/01_bronze.py 20000
python pipelines/02_silver.py
python pipelines/03_gold.py
```

The generator takes about 1.5 min. It must print these counts:

```
1,561,371 events of 20,000 customers (78 each) -> .../data/bronze: crm 869,225, app 652,239, core 39,907
1,568,016 raw rows -> 1,561,371 events (6,645 duplicates dropped) of 20,000 customers -> .../data/silver
train: 16,000 customers, 1,250,691 events, 6,296,229 tokens (0.0% [UNK]) -> .../data/gold/tokens/train.pkl
test: 4,000 customers, 310,680 events, 1,563,925 tokens (0.0% [UNK]) -> .../data/gold/tokens/test.pkl
tokenizer: 23 fields, a vocabulary of 462 tokens -> .../data/gold/tokenizer.json
```

Then check the raw exports byte for byte:

```bash
md5sum -c <<'EOF'
d08fac37029d1e313b20390e12932e7c  data/bronze/app_events.jsonl
9e49df9e6112b702f09464b7490dfa08  data/bronze/core_banking.csv
a5c51d5f3ed36b196c493493566d7679  data/bronze/crm_contacts.csv
91a290936ed0ee0eeb7d4d7d2c46191c  data/bronze/customers.csv
159766427140745ae4f25f7d55995b11  data/truth/personas.csv
EOF
```

### 4. The GPU experiment, about 20 min on an A10

On a fresh Ubuntu GPU machine, with the project copied to it:

```bash
bash experiments/gpu.sh
```

The script installs uv, a Python 3.11 venv and CUDA torch. It generates the data if `data/gold` is missing. It then
pretrains nano and small for 2,000 steps each, running pipeline 5 after each. Logs go to `logs/`, reports to
`reports/`.

The same steps by hand:

```bash
python pipelines/04_train.py --name small --size small --steps 2000 --token-budget 32768 --eval-every 250
python pipelines/05_downstream.py --name small
```

Times on the A10:

| Step | Time |
|---|---|
| Data | 2 min |
| nano pretraining | 1.5 min |
| nano pipeline 5 | 4 min |
| small pretraining | 5 min |
| small pipeline 5 | 5 min |

What must match in `reports/downstream_small.md`:
- **Counts column, exactly:** 0.765, 0.682, 0.719, 0.855, 0.630, 0.667, 0.707, 0.603. It depends only on the data and scikit-learn, so a mismatch means the data or the downstream code differs.
- **Training:** `logs/04_small.txt` ends with about 82% of hidden tokens right, at a loss near 1.47.
- **The uplift:** "opens the push it gets" and "visits the app" are **significant**, near +0.031 and +0.005.
- **Other embedding columns:** close to the table above, but not to the digit. GPU pretraining is seeded but not bit-for-bit deterministic (bf16, non-deterministic CUDA kernels). This was one run, so how far a rerun can move is not measured.

### 5. LoRA and all KPIs, about 17 min on an A10

```bash
python pipelines/06_finetune.py --name small
python pipelines/07_compare.py --name small --boot 500
```

Pipeline 6 fine-tunes each task, in about 2.5 min for the big tasks, and saves predictions to
`reports/predictions_small/`. Pipeline 7 turns them into `reports/compare_small.md` and `gains_small.png`.

Compare with the tables above. As in step 4, the counts AUCs must match exactly, and the LoRA AUCs only roughly.

## Caveats

- **Synthetic data:** the personas give each customer hidden tastes (topics, tone, hours, channels). The embedding
  can only find signal that the generator put in; the push uplift says the model reads those tastes from the
  history better than counts do. Real data has to confirm it.
- **One run:** each configuration ran once, with seed 0, and the variation between pretraining seeds on a GPU is
  not measured.
- **Embedding batches:** pipeline 5 of this run batched the embedding pass by the 24-token cap per event. The
  current `pragmaLib.model.embed` batches by real widths (`groups()`). It is the same computation in a different
  grouping, so the embeddings agree up to bf16 rounding.
- **Multiple comparisons:** pipelines 5 and 6 correct for testing many tasks at once (Holm). Pipeline 7's
  intervals are per KPI, without that correction.
