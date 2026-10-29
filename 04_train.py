"""Pipeline 4 · Train the foundation model, starting from data/gold.

PRAGMA's pretraining, with pragmatiq: masked modelling. The model reads every train customer's tokens with some of
them hidden (single tokens, whole events, a field across the history) and learns to guess them back; no labels.
Its profile, event and history encoders sum a customer up in one embedding, which pipeline 5 uses.

Reads  data/gold/tokens/        (pipeline 3)
Writes models/foundation/       checkpoints/last.pt, the tokenizer, metrics.jsonl (loss and accuracy per step)

Run:  python pipelines/04_train.py [--steps 600] [--size nano]
      nano: ~10 minutes on a laptop CPU; small / medium / large are PRAGMA-S / M / L (10M / 100M / 1B): use a GPU
"""
import argparse
import json
import shutil
from pathlib import Path

from common import GOLD, MODELS, utf8

utf8()
from pragmatiq import api  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=600)
p.add_argument("--size", default="nano", choices=["nano", "small", "medium", "large"])
a = p.parse_args()
if not (GOLD / "tokens" / "shard_manifest.json").exists():
    raise SystemExit("no token shards yet: run pipelines/03_gold.py first")
shutil.rmtree(MODELS / "foundation", ignore_errors=True)  # a fresh run, not a resumed one
# A fixed schedule for a small book (token_budget = tokens per batch); on millions of customers config="auto"
# plans batch size and steps for the hardware it finds.
result = api.pretrain(GOLD / "tokens", "foundation", model_size=a.size, runs_root=MODELS,
                      config={"max_steps": a.steps, "token_budget": 8192, "warmup_steps": max(10, a.steps // 10), "devices": 1})
run = Path(result["run_dir"])
metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if '"mlm_acc"' in line]
print(f"trained {a.size} for {a.steps} steps -> {run}")
print(f"hidden tokens guessed right: {metrics[0]['mlm_acc']:.0%} at step {metrics[0]['step']}, {metrics[-1]['mlm_acc']:.0%} at step {metrics[-1]['step']}")
