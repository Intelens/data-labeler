"""Self-check: python -m pragmaLib.check   (from journeyModeller/)

Customers whose events follow a fixed pattern (an email, then its opening, then a page on the same topic); after a
short pretraining, hidden tokens must be guessed far better than chance, and the same pattern must give the same
customer similar embeddings.
"""
import tempfile
from pathlib import Path

import numpy as np

from .model import embed, load
from .pretrain import mask, pretrain
from .tokenizer import Tokenizer, log_time

rng = np.random.default_rng(0)
T0 = int(np.datetime64("2026-01-01", "us").astype(np.int64))


def customer(topic):
    evs, t = [], T0 + int(rng.integers(0, 300)) * 86_400_000_000
    for _ in range(int(rng.integers(3, 8))):
        for event, f in (("email_sent", {"channel": "email"}), ("email_opened", {"read_seconds": str(int(rng.integers(5, 90)))}),
                         ("page_visited", {"dwell_seconds": str(int(rng.integers(10, 300)))})):
            t += int(rng.integers(1, 600)) * 60_000_000
            evs.append((t, "communication" if event.startswith("email") else "app", {"event": event, "topic": topic, **f}))
    return {"age": str(int(rng.integers(18, 80)))}, evs


assert abs(log_time(0)) < 1e-9 and abs(log_time(8) - 8 * np.log(2)) < 1e-9
records = [customer(["savings", "mortgage"][i % 2]) for i in range(200)]
tok = Tokenizer.fit(records)
assert tok.kinds["read_seconds"] == "numeric" and tok.kinds["topic"] == "categorical"
enc = tok.encode(*records[0])
assert len(enc["keys"]) == len(enc["vals"]) == len(enc["evt"]) and enc["time"][-1] == 0 and enc["time"][0] > 0
assert tok.value_id("topic", "never seen") == 2  # [UNK]

with tempfile.TemporaryDirectory() as d:
    run = Path(d)
    pretrain([tok.encode(*r) for r in records], len(tok.vocab), run, steps=150, token_budget=4096, log_every=50)
    tok.save(run / "tokenizer.json")
    model, tok2 = load(run)
    last = [line for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line][-1]
    acc = float(last.split('"mlm_acc": ')[1].split(",")[0])
    assert acc > 0.3, f"hidden tokens right only {acc:.0%}"  # chance is about 1 / vocabulary size
    E = embed(model, tok2, [(a, e, None) for a, e in records[:40]])
    assert E.shape == (40, model.dim) and np.isfinite(E).all()
    Z = (E - E.mean(0)) / (E.std(0) + 1e-9)
    same = np.array([[i % 2 == j % 2 for j in range(40)] for i in range(40)])
    sim = np.corrcoef(Z)
    assert sim[same].mean() > sim[~same].mean(), "customers on the same topic should embed closer together"

import torch  # noqa: E402

from .model import collate  # noqa: E402

b = collate([tok.encode(*r) for r in records[:64]])
_, scored, _ = mask(b, torch.Generator().manual_seed(0))
share = scored.sum().item() / (b["tok_mask"][..., 1:].sum().item())
assert 0.2 < share < 0.45, share  # 15% tokens + 10% events + 10% fields, minus 10% [UNK]
print(f"pragmaLib ok: hidden tokens right {acc:.0%}, {share:.0%} of tokens scored per batch")
