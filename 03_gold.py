"""Pipeline 3 · Tokenize -> data/gold.

PRAGMA's key-value-time tokenization, with pragmatiq's tokenizer: every field of an event becomes a key token (which
field) plus a value token (a category, or the percentile bucket of a number); the time stays a feature of the event,
not a token; the profile becomes tokens of its own. The vocabulary is fitted on the train customers only, so test
customers' unseen values become [UNK], as they would in production.

Reads  data/silver/   (pipeline 2)
Writes data/gold/split.json           which customers are train / test (seeded, 20% test)
       data/gold/records/{train,test}/ events.parquet + profiles.parquet, pragmatiq's data contract (validated)
       data/gold/tokens/              the tokenizer + token shards of the train customers: what pretraining reads

Run:  python pipelines/03_gold.py       (needs: pip install "pragmatiq[train]==1.1.0")
"""
import json
import shutil

from common import GOLD, SEED, SILVER, TEST_SHARE, read_silver, utf8

utf8()
import lmdb  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from pragmatiq import api  # noqa: E402
from pragmatiq.core.schema import EVENTS_SCHEMA, PROFILES_SCHEMA  # noqa: E402
from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig  # noqa: E402

customers = {p.stem: read_silver(p) for p in sorted(SILVER.glob("*.txt"))}
ids = [c for c, (_, evs) in customers.items() if evs]
test = set(np.random.default_rng(SEED).choice(ids, int(len(ids) * TEST_SHARE), replace=False).tolist())
shutil.rmtree(GOLD, ignore_errors=True)
GOLD.mkdir(parents=True)
(GOLD / "split.json").write_text(json.dumps({"seed": SEED, "train": sorted(set(ids) - test), "test": sorted(test)}), encoding="utf-8")

for part in ("train", "test"):  # the records: one per customer, events + profile as of the last event
    keep = [c for c in ids if (c in test) == (part == "test")]
    ev = {"user_id": [], "ts": [], "source": [], "fields": []}
    for c in keep:
        for ts, source, fields in customers[c][1]:
            for k, v in zip(ev, (c, ts, source, list(fields.items()))):
                ev[k].append(v)
    prof = {"user_id": keep, "as_of": [customers[c][1][-1][0] for c in keep],
            "attributes": [list(customers[c][0].items()) for c in keep], "lifelong": [[] for _ in keep]}
    out = GOLD / "records" / part
    out.mkdir(parents=True)
    pq.write_table(pa.Table.from_pydict(ev, schema=EVENTS_SCHEMA), out / "events.parquet")
    pq.write_table(pa.Table.from_pydict(prof, schema=PROFILES_SCHEMA), out / "profiles.parquet")
    report = api.validate(out)  # pragmatiq's contract check: types, sources, time order, cardinality
    if not report["ok"]:
        raise SystemExit("\n".join(report["errors"]))
    print(f"{part}: {len(keep):,} customers, {len(ev['ts']):,} events -> {out} ({report['summary']})")

# Numbers with 16+ distinct values become percentile buckets (age, income, seconds, amounts); others are categories.
tok = PragmaTokenizer(TokenizerConfig.from_dict({"numeric_min_cardinality": 16}))
tok.fit(GOLD / "records" / "train")
tok.save(GOLD / "tokenizer")
shards = api.tokenize(GOLD / "records" / "train", GOLD / "tokens", tokenizer_dir=GOLD / "tokenizer")
shutil.rmtree(GOLD / "tokenizer")  # api.tokenize keeps a copy next to the shards
# ponytail: pragmatiq's LMDB user index reserves 16 GB, a sparse file on Windows that zip / copy read in full;
# a compact copy holds only the used pages
idx = GOLD / "tokens" / "user_index.lmdb"
(GOLD / "tokens" / "index.compact").mkdir()
with lmdb.open(str(idx), readonly=True, lock=False) as env:
    env.copy(str(GOLD / "tokens" / "index.compact"), compact=True)
shutil.rmtree(idx)
(GOLD / "tokens" / "index.compact").rename(idx)

kinds = {kind: sorted(k for k, v in tok.field_kind.items() if v == kind) for kind in ("categorical", "numeric", "text")}
print(f"tokenizer: {len(tok.key_vocab)} fields, a vocabulary of {tok.vocab_size:,} tokens")
for kind, keys in kinds.items():
    print(f"  {kind}: {', '.join(keys) or '-'}")
print(f"tokens -> {GOLD / 'tokens'}: " + ", ".join(f"{k} {v:,}" for k, v in shards.items() if isinstance(v, int)))
