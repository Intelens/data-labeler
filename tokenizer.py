"""PRAGMA's key-value-time tokenizer (paper §2.2).

Every field of an event, and of the profile, becomes one input position: a key token (which field) plus a value token.
  categorical value   one token per value seen in training (unseen -> [UNK])
  numeric value       a percentile bucket, edges learned from training data, plus one bucket for exactly zero
Time is not a token: each event carries the log-seconds back from the most recent event, 8·ln(1+Δt/8), and its
calendar (hour, weekday, day of month). Caps: 24 tokens per event, 200 profile tokens, 6,500 most recent events.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SPECIAL = ["[PAD]", "[MASK]", "[UNK]", "[USR]", "[EVT]"]
PAD, MASK, UNK, USR, EVT = range(len(SPECIAL))
MAX_EVENT_TOKENS, MAX_PROFILE_TOKENS, MAX_EVENTS = 24, 200, 6500


def log_time(seconds):
    """The paper's time transform: 8 · ln(1 + Δt / 8), Δt in seconds."""
    return 8.0 * np.log1p(np.maximum(np.asarray(seconds, dtype=np.float64), 0) / 8.0)


def _floats(values):
    try:
        return np.array([float(v) for v in values])
    except ValueError:
        return None


class Tokenizer:
    """vocab: token -> id, one shared table for keys ("k:age") and values ("v:age#3", "v:event=email_sent")."""

    def __init__(self, vocab: dict, kinds: dict, edges: dict):
        self.vocab, self.kinds, self.edges = vocab, kinds, {k: np.asarray(e) for k, e in edges.items()}

    @classmethod
    def fit(cls, records, n_buckets=64, numeric_min_cardinality=16, max_categories=1000):
        """records: [(profile attributes, [(time in µs, source, fields)])], train customers only.
        A field whose values all parse as numbers, with at least numeric_min_cardinality distinct ones, is numeric."""
        counts = defaultdict(Counter)
        for attrs, events in records:
            for k, v in attrs.items():
                counts[k][v] += 1
            for _, source, f in events:
                for k, v in {"source": source, **f}.items():
                    counts[k][v] += 1
        vocab, kinds, edges = {t: i for i, t in enumerate(SPECIAL)}, {}, {}
        for k, c in sorted(counts.items()):
            vocab[f"k:{k}"] = len(vocab)
            nums = _floats(c)
            if nums is not None and len(c) >= numeric_min_cardinality:
                sample = np.repeat(nums, list(c.values()))
                nonzero = sample[sample != 0]
                kinds[k] = "numeric"
                edges[k] = np.unique(np.percentile(nonzero, np.linspace(0, 100, n_buckets + 1)[1:-1])) if nonzero.size else np.zeros(0)
                for b in range(len(edges[k]) + 2):  # bucket 0: exactly zero
                    vocab[f"v:{k}#{b}"] = len(vocab)
            else:
                # ponytail: the paper sends high-cardinality text through BPE sub-words; here the most frequent
                # max_categories values are categories and the rest [UNK]. Add BPE when real free text arrives.
                kinds[k] = "categorical"
                for v, _ in c.most_common(max_categories):
                    vocab[f"v:{k}={v}"] = len(vocab)
        return cls(vocab, kinds, edges)

    def value_id(self, k, v) -> int:
        if self.kinds.get(k) == "numeric":
            try:
                x = float(v)
            except ValueError:
                return UNK
            return self.vocab[f"v:{k}#{0 if x == 0 else int(np.searchsorted(self.edges[k], x, side='right')) + 1}"]
        return self.vocab.get(f"v:{k}={v}", UNK)

    def pair(self, k, v) -> tuple[int, int]:
        return self.vocab.get(f"k:{k}", UNK), self.value_id(k, v)

    def encode(self, attrs, events, ref_us=None) -> dict:
        """One customer -> numpy arrays. ref_us: where time counts back from (default: the most recent event, as the paper)."""
        events = events[-MAX_EVENTS:]
        ref = ref_us if ref_us is not None else (events[-1][0] if events else 0)
        keys, vals, evt = [], [], []
        for e, (_, source, f) in enumerate(events):
            for k, v in list({"event": f.get("event"), "source": source, **f}.items())[:MAX_EVENT_TOKENS]:
                kk, vv = self.pair(k, v)
                keys.append(kk), vals.append(vv), evt.append(e)
        ts = np.array([t for t, _, _ in events], dtype=np.int64)
        day = ts.astype("datetime64[us]").astype("datetime64[D]")
        cal = np.stack([(ts // 3_600_000_000) % 24, (day.astype(np.int64) + 3) % 7,  # 1970-01-01 was a Thursday
                        (day - day.astype("datetime64[M]")).astype(np.int64) + 1], 1) if len(ts) else np.zeros((0, 3), np.int64)
        prof = [self.pair(k, v) for k, v in list(attrs.items())[:MAX_PROFILE_TOKENS]]
        return {"keys": np.array(keys, np.int32), "vals": np.array(vals, np.int32), "evt": np.array(evt, np.int32),
                "time": log_time((ref - ts) / 1e6).astype(np.float32), "cal": cal.astype(np.int16),
                "pkeys": np.array([p[0] for p in prof], np.int32), "pvals": np.array([p[1] for p in prof], np.int32)}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"vocab": self.vocab, "kinds": self.kinds, "edges": {k: e.tolist() for k, e in self.edges.items()}}),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], d["kinds"], d["edges"])
