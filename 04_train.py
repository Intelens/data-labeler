"""Pipeline 4 · Train the foundation model, starting from data/gold.

PRAGMA's pretraining (pragmaLib/pretrain.py): masked modelling. The model reads every train customer's tokens with
some of them hidden (15% of tokens, 10% of whole events, 10% of a field across the history) and learns to guess them
back; no labels. Every --eval-every steps it pauses: the downstream probes of pipeline 5 are fit on its current
embeddings (on 4 snapshot dates, for speed) and their AUC on test customers is logged next to the loss.

Reads  data/gold/tokens/train.pkl + data/gold/tokenizer.json (pipeline 3); data/silver/ for the downstream probes
Writes models/foundation/   model.pt (weights + size), tokenizer.json, metrics.jsonl (loss, accuracy, downstream AUC)
       reports/training_progress.png   loss and downstream AUC against pretraining step

Run:  python pipelines/04_train.py [--steps 600] [--size nano] [--eval-every 100]
      nano: minutes on a laptop CPU; small / medium / large are PRAGMA-S / M / L (10M / 100M / 1B): use a GPU
"""
import argparse
import json
import pickle
import shutil

from common import GOLD, MODELS, REPORTS
from downstream import evaluator, plot_progress

from pragmaLib.model import SIZES
from pragmaLib.pretrain import pretrain
from pragmaLib.tokenizer import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=600)
p.add_argument("--size", default="nano", choices=list(SIZES))
p.add_argument("--token-budget", type=int, default=8192, help="tokens per batch")
p.add_argument("--eval-every", type=int, default=100, help="steps between downstream evaluations; 0 = only at the end")
a = p.parse_args()
if not (GOLD / "tokens" / "train.pkl").exists():
    raise SystemExit("no tokens yet: run pipelines/03_gold.py first")
with open(GOLD / "tokens" / "train.pkl", "rb") as fh:  # our own file, written by pipeline 3
    items = pickle.load(fh)
tok = Tokenizer.load(GOLD / "tokenizer.json")
evaluate, baseline = evaluator(tok)
print("counts baseline, AUC on test customers: " + ", ".join(f"{t} {v:.3f}" for t, v in baseline.items()))
run = MODELS / "foundation"
shutil.rmtree(run, ignore_errors=True)  # a fresh run, not a resumed one
pretrain(items, len(tok.vocab), run, size=a.size, steps=a.steps, token_budget=a.token_budget, evaluate=evaluate, eval_every=a.eval_every)
shutil.copy(GOLD / "tokenizer.json", run / "tokenizer.json")  # the model is only valid with its own vocabulary
REPORTS.mkdir(exist_ok=True)
plot_progress(run / "metrics.jsonl", baseline, REPORTS / "training_progress.png", f"Pretraining {a.size}: masked modelling and downstream probes")
last = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()][-1]
print(f"trained {a.size} for {a.steps} steps -> {run}; hidden tokens guessed right {last['mlm_acc']:.0%}; "
      f"plot -> {REPORTS / 'training_progress.png'}")
