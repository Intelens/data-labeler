"""Pipeline 5 · Downstream tasks: logistic regression on the foundation model's embeddings.

As PRAGMA evaluates its embeddings: freeze the model, embed customers, fit a small model per task. Every customer
is looked at on 8 snapshot dates, two weeks apart; the history before the date goes through the frozen foundation
model (one embedding, with the profile), and each task asks what happens in the 7 days after it: will they convert,
open an email, click a banner, ... One logistic regression per task, fit on the train customers of pipeline 3,
scored on the test customers, next to a baseline that knows only counts (pipelines/downstream.py), and both
together, as when embeddings are added to existing propensity models.

Reads  models/foundation/ (pipeline 4), data/silver/ (pipeline 2), data/gold/split.json (pipeline 3)
Writes reports/downstream.md + downstream.json

Run:  python pipelines/05_downstream.py
"""
import json

import numpy as np
from common import MODELS, REPORTS
from downstream import EVERY_DAYS, HORIZON_DAYS, TASKS, probe, snapshots

from pragmaLib.model import embed, load

if not (MODELS / "foundation" / "model.pt").exists():
    raise SystemExit("no foundation model yet: run pipelines/04_train.py first")
model, tok = load(MODELS / "foundation")
s = snapshots(8)
te, Y = s["te"], s["Y"]
print(f"{len(te):,} snapshots ({te.sum():,} of test customers); embedding them ...")
E = embed(model, tok, s["records"])
res = {name: probe(X, Y, te) for name, X in {"counts": s["base"], "embedding": E, "counts + embedding": np.hstack([s["base"], E])}.items()}

results = []
for j, (task, event) in enumerate(TASKS.items()):
    r = {"task": task, "event": event, "rate": round(float(Y[te, j].mean()), 4), "positives_test": int(Y[te, j].sum())}
    results.append({**r, **{n: res[n][task] for n in res}} if res["counts"][task] else {**r, "skipped": "too few positives"})
REPORTS.mkdir(exist_ok=True)
(REPORTS / "downstream.json").write_text(json.dumps({"horizon_days": HORIZON_DAYS, "snapshots": 8, "every_days": EVERY_DAYS,
                                                     "rows": len(te), "test_rows": int(te.sum()), "tasks": results}, indent=2), encoding="utf-8")
lines = [f"# Downstream tasks: what happens in the {HORIZON_DAYS} days after a snapshot", "",
         f"Logistic regression per task, fit on {int((~te).sum()):,} snapshots of train customers, scored on {int(te.sum()):,} snapshots "
         f"of test customers (8 snapshot dates, {EVERY_DAYS} days apart). ROC-AUC: 0.5 is a coin flip, 1 is perfect. "
         "Average precision (AP): compare it with the base rate. Gain: AUC of the embedding over the counts baseline.", "",
         "| Task | Base rate | AUC counts | AUC embedding | AUC counts + embedding | AP counts | AP embedding | AP counts + embedding | Gain |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in results:
    if "skipped" in r:
        lines.append(f"| {r['task']} | {r['rate']:.1%} | {r['skipped']} ({r['positives_test']} in test) | | | | | | |")
        continue
    c, e, b = r["counts"], r["embedding"], r["counts + embedding"]
    lines.append(f"| {r['task']} | {r['rate']:.1%} | {c['auc']:.3f} | {e['auc']:.3f} | {b['auc']:.3f} | {c['ap']:.3f} | {e['ap']:.3f} | {b['ap']:.3f} | {e['auc'] / c['auc'] - 1:+.1%} |")
(REPORTS / "downstream.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[6:]))
print(f"-> {REPORTS / 'downstream.md'}")
