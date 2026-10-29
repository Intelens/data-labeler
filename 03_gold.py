"""Pipeline 3 · Tokenize -> data/gold.

PRAGMA's key-value-time tokenization (pragmaLib/tokenizer.py): every field of an event becomes a key token (which
field) plus a value token (a category, or the percentile bucket of a number); the time stays a feature of the event,
not a token; the profile becomes tokens of its own. The vocabulary is fitted on the train customers only, so test
customers' unseen values become [UNK], as they would in production.

Reads  data/silver/   (pipeline 2)
Writes data/gold/split.json        which customers are train / test (seeded, 20% test)
       data/gold/tokenizer.json    the vocabulary: fields, categories, bucket edges
       data/gold/tokens/train.pkl, test.pkl   every customer's tokens, log-times and calendar, as numpy arrays

Run:  python pipelines/03_gold.py
"""
import json
import pickle
import shutil

import numpy as np
from common import GOLD, SEED, SILVER, TEST_SHARE, read_silver

from pragmaLib.tokenizer import Tokenizer

customers = {p.stem: read_silver(p) for p in sorted(SILVER.glob("*.txt"))}
ids = [c for c, (_, evs) in customers.items() if evs]
if not ids:
    raise SystemExit("no silver files yet: run pipelines/02_silver.py first")
test = set(np.random.default_rng(SEED).choice(ids, int(len(ids) * TEST_SHARE), replace=False).tolist())
train = [c for c in ids if c not in test]
shutil.rmtree(GOLD, ignore_errors=True)
(GOLD / "tokens").mkdir(parents=True, exist_ok=True)
(GOLD / "split.json").write_text(json.dumps({"seed": SEED, "train": train, "test": sorted(test)}), encoding="utf-8")

# Numbers with 16+ distinct values become percentile buckets (age, income, seconds, amounts); others are categories.
tok = Tokenizer.fit([customers[c] for c in train], numeric_min_cardinality=16)
tok.save(GOLD / "tokenizer.json")
for part, keep in (("train", train), ("test", sorted(test))):
    items = [{"user": c, **tok.encode(*customers[c])} for c in keep]  # ponytail: one pickle per split; shards when it outgrows memory
    with open(GOLD / "tokens" / f"{part}.pkl", "wb") as fh:
        pickle.dump(items, fh)
    unk = sum(int((it["vals"] == 2).sum()) for it in items) / max(1, sum(len(it["vals"]) for it in items))
    print(f"{part}: {len(items):,} customers, {sum(len(it['time']) for it in items):,} events, "
          f"{sum(len(it['keys']) for it in items):,} tokens ({unk:.1%} [UNK]) -> {GOLD / 'tokens' / f'{part}.pkl'}")
kinds = {kind: sorted(k for k, v in tok.kinds.items() if v == kind) for kind in ("categorical", "numeric")}
print(f"tokenizer: {len(tok.kinds)} fields, a vocabulary of {len(tok.vocab):,} tokens -> {GOLD / 'tokenizer.json'}")
for kind, keys in kinds.items():
    print(f"  {kind}: {', '.join(keys)}")
