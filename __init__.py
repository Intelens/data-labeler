"""pragmaLib: the PRAGMA paper (Revolut, arXiv 2604.08649) in plain PyTorch, without pragmatiq.

  tokenizer.py   key-value-time tokenization: keys, categories, percentile buckets, log-time and calendar (§2.2)
  model.py       profile / event / history encoders, rotary time positions, MLM head, embedding, load (§2.3)
  pretrain.py    masked modelling (15% tokens, 10% events, 10% fields), Muon + AdamW, token-budget batches (§2.3.5, §2.4)
  check.py       a two-minute self-check: python -m pragmaLib.check
"""
