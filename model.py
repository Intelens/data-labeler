"""PRAGMA's model (paper §2.3): profile, event and history encoders, and the masked-modelling head.

  input position    E(key) + E(value), one embedding table E for keys and values (also the output layer)
  event encoder     each event on its own: [EVT] + its key-value positions -> the event's vector, plus a calendar MLP
                    (sin/cos of hour, weekday, day of month)
  profile encoder   [USR] + the profile's key-value positions -> the customer's state z_a
  history encoder   [z_a, event vectors] with rotary positions on the log-seconds before the evaluation point
                    -> z_h; the customer's embedding is z_h at [USR]
  MLM head          [token output, z_h at its event, z_h at USR] -> d -> logits against E (tied)
Pre-norm transformer blocks, GELU, dropout 0.1, bidirectional attention.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .tokenizer import EVT, PAD, USR

# paper Table 1 (d_model, d_ffn, depth profile / event / history, heads); nano is ours, for a CPU
SIZES = {"nano": (64, 256, 1, 2, 1, 2), "small": (192, 768, 1, 5, 2, 3),
         "medium": (512, 2048, 3, 16, 6, 8), "large": (1024, 4096, 9, 45, 18, 16)}


def rope(x, pos, base=10000.0):
    """Rotate channel pairs by pos · frequency: rotary positions for continuous positions (log-seconds)."""
    dh = x.shape[-1]
    freq = base ** (-torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)


class Block(nn.Module):
    """Pre-norm self-attention + feed-forward; pad: True where a position is padding; pos: rotary positions or None."""

    def __init__(self, d, ffn, heads, dropout):
        super().__init__()
        self.heads, self.p = heads, dropout
        self.n1, self.qkv, self.out = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.ff = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad, pos=None):
        n, length, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(n, length, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if pos is not None:
            q, k = rope(q, pos), rope(k, pos)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=~pad[:, None, None, :], dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(n, length, d)))
        return x + self.drop(self.ff(self.n2(x)))


class Encoder(nn.Module):
    def __init__(self, depth, d, ffn, heads, dropout):
        super().__init__()
        self.blocks, self.norm = nn.ModuleList(Block(d, ffn, heads, dropout) for _ in range(depth)), nn.LayerNorm(d)

    def forward(self, x, pad, pos=None):
        for b in self.blocks:
            x = b(x, pad, pos)
        return self.norm(x)


class PragmaModel(nn.Module):
    def __init__(self, vocab_size, size="nano", dropout=0.1, init_std=None, emb_dropout=0.0, head="mlp"):
        super().__init__()
        d, ffn, dp, de, dh, heads = SIZES[size]
        self.config = {"vocab_size": vocab_size, "size": size, "dropout": dropout, "init_std": init_std,
                       "emb_dropout": emb_dropout, "head": head}
        self.dim = d
        self.emb, self.emb_drop = nn.Embedding(vocab_size, d, padding_idx=PAD), nn.Dropout(emb_dropout)
        # ponytail: the paper's within-field position embedding (multi-valued fields) is left out: our fields have one value
        self.event_enc, self.profile_enc = Encoder(de, d, ffn, heads, dropout), Encoder(dp, d, ffn, heads, dropout)
        self.history_enc = Encoder(dh, d, ffn, heads, dropout)
        self.calendar = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d)) if head == "mlp" else nn.Linear(3 * d, d)
        nn.init.normal_(self.emb.weight, std=init_std or d ** -0.5)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()  # padding adds nothing (its gradient stays zero)

    def forward(self, b):
        """b: a batch from collate(). Returns the token outputs, z_h per event, and the customer embeddings."""
        n, e_max, t_max = b["keys"].shape
        x = self.emb_drop(self.emb(b["keys"]) + self.emb(b["vals"]))  # [B, E, T, d], position 0 of every event is [EVT]
        live = b["ev_mask"].view(-1)
        tok = self.event_enc(x.view(n * e_max, t_max, -1)[live], ~b["tok_mask"].view(n * e_max, t_max)[live])
        c = b["cal"].float() * torch.tensor([2 * math.pi / 24, 2 * math.pi / 7, 2 * math.pi / 31], device=x.device)
        z_e = torch.zeros(n * e_max, self.dim, device=x.device, dtype=tok.dtype)
        z_e[live] = tok[:, 0] + self.calendar(torch.cat([c.sin(), c.cos()], -1).view(n * e_max, 6)[live]).to(tok.dtype)
        p = self.profile_enc(self.emb_drop(self.emb(b["pkeys"]) + self.emb(b["pvals"])), ~b["p_mask"])
        h = torch.cat([p[:, :1], z_e.view(n, e_max, -1)], 1)  # [USR] (z_a) + the events
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=x.device), ~b["ev_mask"]], 1)
        pos = torch.cat([torch.zeros(n, 1, device=x.device), b["time"]], 1)  # the evaluation point is position 0
        z_h = self.history_enc(h, pad, pos)
        return {"tok": tok, "live": live, "z_h": z_h, "user": z_h[:, 0]}

    def mlm_logits(self, out, b, sel):
        """Logits for the selected event tokens (sel: [B, E, T] bool, never position 0)."""
        n, e_max, t_max = sel.shape
        rows = torch.full((n * e_max,), -1, dtype=torch.long, device=sel.device)
        rows[out["live"]] = torch.arange(int(out["live"].sum()), device=sel.device)
        bi, ei, ti = sel.nonzero(as_tuple=True)
        h = torch.cat([out["tok"][rows[bi * e_max + ei], ti], out["z_h"][bi, ei + 1], out["z_h"][bi, 0]], -1)
        return self.head(h) @ self.emb.weight.T


def collate(items, device="cpu"):
    """Tokenizer.encode() outputs -> padded tensors: keys / vals [B, E, 1+T] with [EVT] first, the profile [B, 1+P]
    with [USR] first, log-seconds and calendar per event, and masks (True = real)."""
    n = len(items)
    e_max = max(1, max(len(it["time"]) for it in items))
    t_max = 1 + max([int(np.bincount(it["evt"]).max()) for it in items if len(it["evt"])] or [0])
    p_max = 1 + max(len(it["pkeys"]) for it in items)
    keys, vals = np.zeros((n, e_max, t_max), np.int64), np.zeros((n, e_max, t_max), np.int64)
    pk, pv = np.zeros((n, p_max), np.int64), np.zeros((n, p_max), np.int64)
    time, cal = np.zeros((n, e_max), np.float32), np.zeros((n, e_max, 3), np.int64)
    ev_mask, p_mask = np.zeros((n, e_max), bool), np.zeros((n, p_max), bool)
    for i, it in enumerate(items):
        ne = len(it["time"])
        keys[i, :ne, 0], ev_mask[i, :ne], time[i, :ne], cal[i, :ne] = EVT, True, it["time"], it["cal"]
        slot = np.zeros(ne, np.int64)
        for k, v, e in zip(it["keys"], it["vals"], it["evt"]):  # ponytail: a Python loop per token; vectorise if it shows up
            slot[e] += 1
            keys[i, e, slot[e]], vals[i, e, slot[e]] = k, v
        npf = len(it["pkeys"])
        pk[i, 0], pk[i, 1:1 + npf], pv[i, 1:1 + npf], p_mask[i, :1 + npf] = USR, it["pkeys"], it["pvals"], True
    tok_mask = keys != PAD
    t = {"keys": keys, "vals": vals, "pkeys": pk, "pvals": pv, "time": time, "cal": cal, "ev_mask": ev_mask, "p_mask": p_mask,
         "tok_mask": tok_mask}
    return {k: torch.from_numpy(v).to(device) for k, v in t.items()}


@torch.no_grad()
def embed(model, tokenizer, records, batch_size=128):
    """records: [(profile attributes, events, evaluation point in µs or None)] -> [n, d] customer embeddings."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(records), batch_size):
        b = collate([tokenizer.encode(*r) for r in records[i:i + batch_size]], device)
        out.append(model(b)["user"].float().cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, model.dim), np.float32)


def load(run_dir):
    """A trained run (pretrain.py) -> (model, tokenizer)."""
    from .tokenizer import Tokenizer

    ck = torch.load(f"{run_dir}/model.pt", map_location="cpu", weights_only=True)
    model = PragmaModel(**ck["config"])
    model.load_state_dict(ck["state"])
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval(), Tokenizer.load(f"{run_dir}/tokenizer.json")
