"""Pipeline 5 · Downstream tasks: logistic regression on the foundation model's embeddings.

As PRAGMA evaluates its embeddings: freeze the model, embed customers, fit a small model per task. Every customer
is looked at on 8 snapshot dates, two weeks apart; the history before the date goes through the frozen foundation
model (one embedding, with the profile), and each task asks what happens in the 7 days after it: will they convert,
open an email, click a banner, ... One logistic regression per task, fit on the train customers of pipeline 3,
scored on the test customers, next to a baseline that knows only counts: each event type in the history and in
the last 30 days, days since the last event, and the profile's numbers. And both together, as when embeddings are
added to existing propensity models.

Reads  models/foundation/ (pipeline 4), data/silver/ (pipeline 2), data/gold/split.json (pipeline 3)
Writes reports/downstream.md + downstream.json

Run:  python pipelines/05_downstream.py       (needs: pip install "pragmatiq[train]==1.1.0" scikit-learn)
"""
import json

from common import DAY_US, GOLD, MODELS, REPORTS, SILVER, read_silver, utf8

utf8()
import numpy as np  # noqa: E402
from pragmatiq.models.pragmatiq import PragmaModel  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HORIZON_DAYS, SNAPSHOTS, EVERY_DAYS = 7, 8, 14
TASKS = {"converts (contract signed)": "contract_signed", "applies for a product": "application", "opens an email": "email_opened",
         "opens a push message": "push_opened", "clicks a banner": "banner_clicked", "answers a call": "call_answered",
         "visits a page": "page_visited", "unsubscribes": "unsubscribed", "terminates a contract": "contract_terminated"}
PROFILE_NUMBERS = ("age", "monthly_income", "customer_since_years")

if not (MODELS / "foundation" / "checkpoints" / "last.pt").exists():
    raise SystemExit("no foundation model yet: run pipelines/04_train.py first")
model = PragmaModel.from_pretrained(MODELS / "foundation")
split = json.loads((GOLD / "split.json").read_text(encoding="utf-8"))
test_ids = set(split["test"])
customers = {p.stem: read_silver(p) for p in sorted(SILVER.glob("*.txt"))}
names = sorted({f["event"] for _, evs in customers.values() for _, _, f in evs})
end = max(evs[-1][0] for _, evs in customers.values() if evs)
H = HORIZON_DAYS * DAY_US
cutoffs = [end - H - k * EVERY_DAYS * DAY_US for k in range(SNAPSHOTS)]


def counts(hist, t, attrs):  # the baseline's view: how often each event happened, how recently, the profile's numbers
    every = [f["event"] for _, _, f in hist]
    recent = [f["event"] for ts, _, f in hist if ts >= t - 30 * DAY_US]
    return [*(every.count(n) for n in names), *(recent.count(n) for n in names), np.log1p((t - hist[-1][0]) / DAY_US), len(hist),
            *(float(attrs.get(k, 0)) for k in PROFILE_NUMBERS)]


records, base, labels, is_test = [], [], [], []
for cid, (attrs, evs) in customers.items():
    for t in cutoffs:
        hist = [e for e in evs if e[0] < t]
        if not hist:
            continue
        coming = {f["event"] for ts, _, f in evs if t <= ts < t + H}
        records.append({"user_id": cid, "events": hist, "attributes": attrs, "lifelong": []})
        base.append(counts(hist, t, attrs))
        labels.append([e in coming for e in TASKS.values()])
        is_test.append(cid in test_ids)
print(f"{len(records):,} snapshots of {len(customers):,} customers ({sum(is_test):,} of test customers); embedding them ...")
E = np.vstack([model.embed_records(records[i:i + 1024]) for i in range(0, len(records), 1024)])
B, Y, te = np.array(base, dtype=float), np.array(labels), np.array(is_test)
inputs = {"counts": B, "embedding": E, "counts + embedding": np.hstack([B, E])}

results = []
for j, (task, event) in enumerate(TASKS.items()):
    y = Y[:, j]
    r = {"task": task, "event": event, "rate": round(float(y[te].mean()), 4), "positives_test": int(y[te].sum())}
    if y[~te].sum() < 20 or y[te].sum() < 10:
        results.append({**r, "skipped": "too few positives"})
        continue
    for name, X in inputs.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[~te], y[~te])
        p = clf.predict_proba(X[te])[:, 1]
        r[name] = {"auc": round(float(roc_auc_score(y[te], p)), 3), "ap": round(float(average_precision_score(y[te], p)), 3)}
    results.append(r)

REPORTS.mkdir(exist_ok=True)
(REPORTS / "downstream.json").write_text(json.dumps({"horizon_days": HORIZON_DAYS, "snapshots": SNAPSHOTS, "every_days": EVERY_DAYS,
                                                     "rows": len(records), "test_rows": int(te.sum()), "tasks": results}, indent=2), encoding="utf-8")
lines = [f"# Downstream tasks: what happens in the {HORIZON_DAYS} days after a snapshot", "",
         f"Logistic regression per task, fit on {int((~te).sum()):,} snapshots of train customers, scored on {int(te.sum()):,} snapshots "
         f"of test customers ({SNAPSHOTS} snapshot dates, {EVERY_DAYS} days apart). ROC-AUC: 0.5 is a coin flip, 1 is perfect. "
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
