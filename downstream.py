"""Downstream tasks, shared by pipeline 4 (during training), pipeline 5 (the report) and experiments/.

A snapshot is a customer's history before a date; each task asks what happens in the 7 days after it. One logistic
regression per task on standard-scaled inputs, as PRAGMA probes its embeddings (§3.1.1): fit on train customers,
scored (ROC-AUC, average precision) on test customers. The counts baseline knows each event type's count in the
history and in the last 30 days, the days since the last event, and the profile's numbers.
"""
import json

import numpy as np
from common import DAY_US, GOLD, SILVER, read_silver
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON_DAYS, EVERY_DAYS = 7, 14
TASKS = {"converts (contract signed)": "contract_signed", "applies for a product": "application", "opens an email": "email_opened",
         "opens a push message": "push_opened", "clicks a banner": "banner_clicked", "answers a call": "call_answered",
         "visits a page": "page_visited", "unsubscribes": "unsubscribed", "terminates a contract": "contract_terminated"}
PROFILE_NUMBERS = ("age", "monthly_income", "customer_since_years")


def snapshots(n_dates=8):
    """Every customer on n_dates snapshot dates, two weeks apart, ending a week before the data does.
    -> {records: [(profile, history, None)], base: counts features, Y: [rows, tasks] bool, te: is a test customer}"""
    test_ids = set(json.loads((GOLD / "split.json").read_text(encoding="utf-8"))["test"])
    customers = {p.stem: read_silver(p) for p in sorted(SILVER.glob("*.txt"))}
    names = sorted({f["event"] for _, evs in customers.values() for _, _, f in evs})
    end, h = max(evs[-1][0] for _, evs in customers.values() if evs), HORIZON_DAYS * DAY_US
    records, base, labels, is_test = [], [], [], []
    for cid, (attrs, evs) in customers.items():
        for t in (end - h - k * EVERY_DAYS * DAY_US for k in range(n_dates)):
            hist = [e for e in evs if e[0] < t]
            if not hist:
                continue
            coming = {f["event"] for ts, _, f in evs if t <= ts < t + h}
            every, recent = [f["event"] for _, _, f in hist], [f["event"] for ts, _, f in hist if ts >= t - 30 * DAY_US]
            records.append((attrs, hist, None))  # times count back from the latest event before the snapshot, as in pretraining
            base.append([*(every.count(n) for n in names), *(recent.count(n) for n in names), np.log1p((t - hist[-1][0]) / DAY_US),
                         len(hist), *(float(attrs.get(k, 0)) for k in PROFILE_NUMBERS)])
            labels.append([e in coming for e in TASKS.values()])
            is_test.append(cid in test_ids)
    return {"records": records, "base": np.array(base, dtype=float), "Y": np.array(labels), "te": np.array(is_test), "n_dates": n_dates}


def probe(X, Y, te) -> dict:
    """-> {task: {"auc", "ap"} or None when too few positives (under 20 in train or 10 in test)}."""
    out = {}
    for j, task in enumerate(TASKS):
        y = Y[:, j]
        if y[~te].sum() < 20 or y[te].sum() < 10:
            out[task] = None
            continue
        p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        out[task] = {"auc": round(float(roc_auc_score(y[te], p)), 3), "ap": round(float(average_precision_score(y[te], p)), 3)}
    return out


def evaluator(tok, n_dates=4):
    """For training: -> (evaluate(model) -> {task: AUC}, the counts baseline's {task: AUC}) on n_dates snapshot dates."""
    from pragmaLib.model import embed

    s = snapshots(n_dates)
    baseline = {t: r["auc"] for t, r in probe(s["base"], s["Y"], s["te"]).items() if r}

    def evaluate(model):
        res = probe(embed(model, tok, s["records"]), s["Y"], s["te"])
        return {t: r["auc"] for t, r in res.items() if r}
    return evaluate, baseline


def plot_progress(metrics_path, baseline, out_png, title=""):
    """Left: masked-modelling loss and accuracy per step. Right: each task's AUC at every evaluation, the counts
    baseline dashed in the same colour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in open(metrics_path, encoding="utf-8") if line.strip()]
    ev = [r for r in rows if "downstream" in r]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.plot([r["step"] for r in rows], [r["loss"] for r in rows], color="tab:blue", label="loss")
    a1.set_xlabel("pretraining step"), a1.set_ylabel("masked-modelling loss", color="tab:blue")
    acc = a1.twinx()
    acc.plot([r["step"] for r in rows], [r["mlm_acc"] for r in rows], color="tab:orange")
    acc.set_ylabel("hidden tokens guessed right", color="tab:orange")
    for i, task in enumerate(t for t in TASKS if any(t in r["downstream"] for r in ev)):
        c = f"C{i}"
        a2.plot([r["step"] for r in ev if task in r["downstream"]], [r["downstream"][task] for r in ev if task in r["downstream"]],
                marker="o", color=c, label=task)
        if task in baseline:
            a2.axhline(baseline[task], color=c, linestyle="--", linewidth=0.8)
    a2.axhline(0.5, color="grey", linewidth=0.6)
    a2.set_xlabel("pretraining step"), a2.set_ylabel("ROC-AUC on test customers (dashed: counts baseline)")
    a2.legend(fontsize=7, loc="lower right")
    fig.suptitle(title or "Pretraining progress: masked modelling and downstream probes")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
