"""PRAGMA's pretraining (paper §2.3.5, §2.4): masked modelling, Muon + AdamW, token-budget batches.

Masking, on the value tokens of events (keys stay visible): 15% of tokens, every token of 10% of events, every token
of 10% of (customer, field) pairs. Of the selected, 10% become [UNK] and are left out of the loss (a dropout); the
rest become [MASK] and are scored: cross-entropy against the whole shared table, label smoothing 0.1.
Optimiser: Muon for the transformer's weight matrices, AdamW for embeddings, norms, biases and heads; linear warmup
then cosine decay; gradients clipped at 1.0; bf16 on a GPU, fp32 on a CPU.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import PragmaModel, collate
from .tokenizer import MASK, UNK


def mask(b, gen, p_token=0.15, p_event=0.10, p_key=0.10, p_unk=0.10):
    """-> (input value ids, scored positions [B, E, T] bool, their true value ids)."""
    vals, keys = b["vals"], b["keys"]
    real = b["tok_mask"].clone()
    real[..., 0] = False  # the [EVT] slot is not a field
    shape, dev = real.shape, real.device
    sel = torch.rand(shape, generator=gen, device=dev) < p_token
    sel |= (torch.rand(shape[:2], generator=gen, device=dev) < p_event)[..., None]
    per_key = torch.rand(shape[0], int(keys.max()) + 1, generator=gen, device=dev) < p_key  # (customer, field) pairs
    sel |= per_key[torch.arange(shape[0], device=dev)[:, None, None], keys]
    sel &= real
    unk = sel & (torch.rand(shape, generator=gen, device=dev) < p_unk)
    scored = sel & ~unk
    return vals.masked_fill(scored, MASK).masked_fill(unk, UNK), scored, vals[scored]


def batches(items, token_budget, rng):
    """Shuffled customers, grouped until a batch holds token_budget tokens; forever."""
    sizes = np.array([len(it["keys"]) + len(it["time"]) + len(it["pkeys"]) + 1 for it in items])
    while True:
        batch, used = [], 0
        for i in rng.permutation(len(items)):
            if batch and used + sizes[i] > token_budget:
                yield batch
                batch, used = [], 0
            batch.append(items[i])
            used += sizes[i]
        if batch:
            yield batch


def pretrain(items, vocab_size, run_dir: Path, size="nano", steps=600, token_budget=8192, seed=0,
             lr_muon=3e-3, lr_adamw=3e-4, weight_decay=0.01, warmup=None, log_every=10, model_args=None, optim="enc",
             evaluate=None, eval_every=0):
    """items: Tokenizer.encode() outputs of the train customers. Writes run_dir/model.pt and metrics.jsonl.
    optim: "enc" = Muon on the encoders' matrices only, AdamW (default betas) with weight decay on the rest;
           "all" = Muon on every matrix but the embedding table, AdamW betas (0.9, 0.95), no decay on 1-D weights.
    evaluate(model) -> {name: number}, called every eval_every steps and at the end; logged in metrics.jsonl."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PragmaModel(vocab_size, size, **(model_args or {})).to(device).train()
    if optim == "all":
        matrices = [p for n, p in model.named_parameters() if p.ndim == 2 and not n.startswith("emb.")]
    else:
        matrices = [p for n, p in model.named_parameters() if p.ndim == 2 and "_enc." in n]  # the transformer's weights
    rest = [p for p in model.parameters() if all(p is not m for m in matrices)]
    if optim == "all":
        adamw = torch.optim.AdamW([{"params": [p for p in rest if p.ndim >= 2], "weight_decay": weight_decay},
                                   {"params": [p for p in rest if p.ndim < 2], "weight_decay": 0.0}], lr=lr_adamw, betas=(0.9, 0.95))
    else:
        adamw = torch.optim.AdamW(rest, lr=lr_adamw, weight_decay=weight_decay)
    opts = [torch.optim.Muon(matrices, lr=lr_muon, weight_decay=weight_decay), adamw]
    warmup = warmup or max(10, steps // 10)
    curve = lambda s: (s + 1) / warmup if s < warmup else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))  # noqa: E731
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, curve) for o in opts]
    gen, rng = torch.Generator(device=device).manual_seed(seed), np.random.default_rng(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    log, data, t0 = open(run_dir / "metrics.jsonl", "w", encoding="utf-8"), batches(items, token_budget, rng), time.time()
    for step in range(1, steps + 1):
        b = collate(next(data), device)
        b["vals"], scored, labels = mask(b, gen)
        if not labels.numel():  # nothing hidden in this batch (tiny batches only)
            continue
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(b)
            logits = model.mlm_logits(out, b, scored).float()
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o, s in zip(opts, scheds):
            o.step()
            s.step()
        if step % log_every == 0 or step == steps:
            m = {"step": step, "loss": round(loss.item(), 4), "mlm_acc": round((logits.argmax(-1) == labels).float().mean().item(), 4),
                 "seconds": round(time.time() - t0, 1)}
            if evaluate and (step == steps or (eval_every and step % eval_every == 0)):
                m["downstream"] = evaluate(model)
                model.train()
            log.write(json.dumps(m) + "\n")
            log.flush()
            print(f"step {step}/{steps}  loss {m['loss']:.3f}  hidden tokens right {m['mlm_acc']:.0%}  {m['seconds']:.0f} s"
                  + (f"  downstream {m['downstream']}" if "downstream" in m else ""), flush=True)
    log.close()
    torch.save({"config": model.config, "state": model.state_dict()}, run_dir / "model.pt")
    return model
