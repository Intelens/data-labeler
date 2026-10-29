"""Pipeline 1 · Create raw events data -> data/bronze.

A year of fake bank customers: what the bank did (emails, push messages, calls, in-app banners) and what each
customer did (opened, visited a page, applied, signed a contract). Hidden traits steer every reaction: a favourite
topic, a favourite channel, the hours they are active, and a profile that leans on both.

Written the way three source systems would export it, each in its own format:
  bronze/crm_contacts.csv     contacts and reactions: ISO times with Z, raw codes (EMAIL_SENT, CALL_CONNECTED, ...),
                              the content text in one column, extra details as JSON
  bronze/app_events.jsonl     page views and applications: epoch milliseconds, nested props, dwell in ms; ~1% duplicates
  bronze/core_banking.csv     contracts and accounts: ';'-separated, dd-mm-yyyy times, amounts in euro
  bronze/customers.csv        one row per customer: age, income, tenure, plan, country

Run:  python pipelines/01_bronze.py [customers]       (default 2,000)
"""
import csv
import json
import shutil
import sys

import numpy as np
from common import BRONZE, CONTENT_FIELD, DAY_US, RAW, SEED

N_CUSTOMERS, DAYS, START = int(sys.argv[1]) if len(sys.argv) > 1 else 2000, 365, "2026-01-01"
HOUR_US = 3_600_000_000
PROFILE = {"age": (18, 80), "monthly_income": (1200, 9000), "customer_since_years": (0, 25),
           "plan": ["standard", "plus", "premium"], "country": ["NL", "DE", "GB"]}
TOPICS = ["savings", "credit_card", "investing", "mortgage"]
PROFILE_EFFECTS = {"age": {"call_made": 1.2, "email_sent": 0.5, "push_sent": -1.0, "banner_shown": -0.7},
                   "monthly_income": {"investing": 1.0, "mortgage": 0.8, "credit_card": -0.5}}
FIRST_EVENT = {"email_sent": 0.22, "push_sent": 0.20, "call_made": 0.10, "banner_shown": 0.13, "page_visited": 0.35}
PEAK_HOURS, HOUR_SPREAD, STRENGTH = [8, 12, 20], 2.0, 1.5
# event: who acts, its fields (fixed, a list to pick from, a (min, max) range, or content items whose appeal
# scales the response), the delay after the previous event in minutes, and what follows with which weight
EVENTS = {
    "email_sent": {"role": "action", "next": {"email_opened": 0.42, "end": 0.58}, "fields": {"channel": "email", "content": [
        {"subject": "Your savings could earn 4% more this year", "topic": "savings", "appeal": 1.4},
        {"subject": "Savings account: quarterly update", "topic": "savings", "appeal": 0.7},
        {"subject": "0% on balance transfers - ends Friday", "topic": "credit_card", "appeal": 1.3},
        {"subject": "Your new card benefits explained", "topic": "credit_card", "appeal": 0.8},
        {"subject": "Start investing from 1 euro", "topic": "investing", "appeal": 1.2},
        {"subject": "Market outlook for the next quarter", "topic": "investing", "appeal": 0.6},
        {"subject": "Mortgage rates just dropped", "topic": "mortgage", "appeal": 1.3},
        {"subject": "Thinking about buying a home?", "topic": "mortgage", "appeal": 0.9}]}},
    "email_opened": {"role": "reaction", "delay": (5, 720), "next": {"page_visited": 0.31, "unsubscribed": 0.02, "end": 0.67},
                     "fields": {"channel": "email", "read_seconds": (2, 120)}},
    "push_sent": {"role": "action", "next": {"push_opened": 0.30, "end": 0.70}, "fields": {"channel": "push", "content": [
        {"message": "Your savings rate just went up", "topic": "savings", "appeal": 1.3},
        {"message": "Round up purchases into savings", "topic": "savings", "appeal": 0.8},
        {"message": "0% balance transfer, today only", "topic": "credit_card", "appeal": 1.2},
        {"message": "Your card now gives cashback", "topic": "credit_card", "appeal": 0.9},
        {"message": "Markets moved: check your portfolio", "topic": "investing", "appeal": 1.1},
        {"message": "Invest your spare change", "topic": "investing", "appeal": 0.8},
        {"message": "Mortgage rates dropped today", "topic": "mortgage", "appeal": 1.3},
        {"message": "See what you could borrow", "topic": "mortgage", "appeal": 0.8}]}},
    "push_opened": {"role": "reaction", "delay": (0, 120), "next": {"page_visited": 0.55, "end": 0.45}, "fields": {"channel": "push"}},
    "call_made": {"role": "action", "next": {"call_answered": 0.45, "end": 0.55}, "fields": {"channel": "phone", "content": [
        {"script": "Savings review call", "topic": "savings", "appeal": 1.0},
        {"script": "Credit card upgrade offer", "topic": "credit_card", "appeal": 0.9},
        {"script": "Investment advice call", "topic": "investing", "appeal": 1.1},
        {"script": "Mortgage advice call", "topic": "mortgage", "appeal": 1.2}]}},
    "call_answered": {"role": "reaction", "delay": (0, 0), "next": {"application": 0.35, "page_visited": 0.20, "end": 0.45},
                      "fields": {"channel": "phone", "duration_minutes": (2, 30), "notes": [
                          {"notes": "customer interested in {topic}, follow-up planned", "appeal": 1.8},
                          {"notes": "reviewed finances, {topic} fits their goals - sent application link", "appeal": 2.0},
                          {"notes": "explained {topic} options, customer will think about it", "appeal": 1.0},
                          {"notes": "customer wants to compare {topic} offers first", "appeal": 0.6},
                          {"notes": "customer not interested in {topic} right now", "appeal": 0.2}]}},
    "banner_shown": {"role": "action", "next": {"banner_clicked": 0.28, "end": 0.72}, "fields": {
        "channel": "in_app", "slot": ["home_top", "home_mid", "product_list"], "content": [
            {"banner": "Earn 4% on your savings", "topic": "savings", "appeal": 1.2},
            {"banner": "Cashback on every card purchase", "topic": "credit_card", "appeal": 1.1},
            {"banner": "Invest your spare change", "topic": "investing", "appeal": 1.0},
            {"banner": "Check what you can borrow", "topic": "mortgage", "appeal": 0.9}]}},
    "banner_clicked": {"role": "reaction", "delay": (0, 2), "next": {"page_visited": 0.90, "end": 0.10}, "fields": {"channel": "in_app"}},
    "page_visited": {"role": "reaction", "delay": (0, 30), "next": {"page_visited": 0.20, "application": 0.33, "end": 0.47},
                     "fields": {"device": ["web", "app"], "dwell_seconds": (5, 300)}},
    "application": {"role": "reaction", "delay": (1, 60), "next": {"contract_signed": 0.61, "end": 0.39}, "fields": {"device": ["web", "app", "branch"]}},
    "contract_signed": {"role": "reaction", "delay": (60, 10080), "next": {"account_opened": 1.0},
                        "fields": {"device": ["web", "app", "branch"], "contract_value": (500, 50000), "term_months": ["12", "24", "36", "60"]}},
    "account_opened": {"role": "reaction", "delay": (60, 4320), "next": {"contract_terminated": 0.12, "end": 0.88}, "fields": {"product_status": "active"}},
    "unsubscribed": {"role": "reaction", "delay": (0, 5), "next": {}, "fields": {"channel": "email"}},
    "contract_terminated": {"role": "reaction", "delay": (43200, 345600), "next": {},
                            "fields": {"reason": ["switched provider", "no longer needed", "contract matured", "fees too high"]}},
}
ACTIONS = [e for e, v in EVENTS.items() if v["role"] == "action"]
rng = np.random.default_rng(SEED)
T0 = int(np.datetime64(START, "us").astype(np.int64))


def value(spec):
    if isinstance(spec, tuple):
        lo, hi = spec
        return int(rng.integers(lo, hi + 1))
    return spec[rng.integers(len(spec))] if isinstance(spec, list) else spec


def pick(weights, bias, lift=1.0):  # the next event; lift > 1 makes going on likelier
    keys = list(weights) or ["end"]
    w = np.array([weights.get(k, 1.0) * bias[k] * (1.0 if k == "end" else lift) for k in keys])
    return keys[rng.choice(len(keys), p=w / w.sum())]


rows, customers = [], []  # rows: (customer, time in microseconds, event, fields)
for i in range(N_CUSTOMERS):
    cid = f"c_{i:05d}"
    profile = {k: value(spec) for k, spec in PROFILE.items()}
    bias = dict(zip([*EVENTS, "end"], rng.lognormal(0, 0.5, len(EVENTS) + 1)))  # individual propensity per event
    taste = dict(zip(TOPICS, rng.dirichlet(np.ones(len(TOPICS)))))  # hidden: favourite topic
    likes = dict(zip(ACTIONS, rng.dirichlet(np.ones(len(ACTIONS)))))  # hidden: favourite channel
    fav_hour = float((rng.choice(PEAK_HOURS) + rng.normal(0, 1)) % 24)  # hidden: when they are active
    for attr, effects in PROFILE_EFFECTS.items():
        lo, hi = PROFILE[attr]
        z = 2 * (profile[attr] - lo) / (hi - lo) - 1
        for trait in (taste, likes):
            for k in trait.keys() & effects.keys():
                trait[k] *= np.exp(effects[k] * z)
    for trait in (taste, likes):
        total = sum(trait.values())
        for k in trait:
            trait[k] /= total
    customers.append({"customer_id": cid, **profile})
    for _ in range(int(rng.integers(4, 17))):  # journeys
        step = pick(FIRST_EVENT, bias)
        own = EVENTS[step]["role"] == "reaction"  # customers start their own journeys at their own hours
        hour = (fav_hour + rng.normal(0, HOUR_SPREAD)) % 24 if own else rng.uniform(0, 24)
        ts = T0 + int(rng.integers(DAYS)) * DAY_US + int(hour * HOUR_US)
        topic = TOPICS[rng.choice(len(TOPICS), p=list(taste.values()))]
        for n in range(50):
            if step == "end":
                break
            ev = EVENTS[step]
            if n:
                ts += int(value(ev.get("delay", (1, 1440))) * 60_000_000)
            if ts >= T0 + DAYS * DAY_US:
                break
            fields, appeal = {}, 1.0
            for k, spec in ev["fields"].items():
                if isinstance(spec, list) and isinstance(spec[0], dict):  # content: one item, its appeal counts
                    item = dict(spec[rng.integers(len(spec))])
                    appeal *= item.pop("appeal", 1.0)
                    fields.update({ck: cv.format(topic=topic) for ck, cv in item.items()})
                else:
                    fields[k] = value(spec)
            topic = fields.pop("topic", topic)
            rows.append((cid, ts, step, {"topic": topic, **fields}))
            lift = appeal * (len(TOPICS) * taste[topic]) ** STRENGTH
            if ev["role"] == "action":  # a good hour, a channel they like?
                off = abs((ts / HOUR_US - fav_hour + 12) % 24 - 12)
                lift *= (0.2 + 1.8 * np.exp(-off ** 2 / (2 * HOUR_SPREAD ** 2))) ** STRENGTH
                lift *= (len(ACTIONS) * likes[step]) ** STRENGTH
            step = pick(ev["next"], bias, lift)

# ---------------------------------------------------------------- export, the way each source system would
shutil.rmtree(BRONZE, ignore_errors=True)
BRONZE.mkdir(parents=True, exist_ok=True)  # a file held open elsewhere can keep the folder
by = {"crm": [], "app": [], "core": []}
for cid, ts, event, f in sorted(rows, key=lambda r: r[1]):  # exports come in time order, all customers mixed
    system, code = RAW[event]
    by[system].append((cid, ts, code, f))
content_of = {v: k for k, v in CONTENT_FIELD.items()}
with open(BRONZE / "crm_contacts.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["customer_id", "event_time", "channel", "event_type", "content", "topic", "details"])
    for cid, ts, code, f in by["crm"]:
        details = {k: f[k] for k in ("read_seconds", "duration_minutes", "notes", "slot") if k in f}
        w.writerow([cid, f"{np.datetime64(ts, 'us')}Z", f.get("channel", ""), code, f.get(CONTENT_FIELD.get(f.get("channel"), ""), ""),
                    f["topic"], json.dumps(details) if details else ""])
with open(BRONZE / "app_events.jsonl", "w", encoding="utf-8") as fh:
    for cid, ts, code, f in by["app"]:
        props = {"topic": f["topic"], "device": f["device"], **({"dwell_ms": f["dwell_seconds"] * 1000} if "dwell_seconds" in f else {})}
        line = json.dumps({"user": cid, "ts": ts // 1000, "name": code, "props": props}) + "\n"
        fh.write(line * (2 if rng.random() < 0.01 else 1))  # the app's retries send ~1% twice
with open(BRONZE / "core_banking.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, delimiter=";")
    w.writerow(["customer_id", "booked_at", "type", "product", "channel", "amount_eur", "term_months", "status", "reason"])
    for cid, ts, code, f in by["core"]:
        w.writerow([cid, np.datetime64(ts // 1_000_000, "s").astype(object).strftime("%d-%m-%Y %H:%M:%S"), code, f["topic"],
                    f.get("device", ""), f.get("contract_value", ""), f.get("term_months", ""), f.get("product_status", ""), f.get("reason", "")])
with open(BRONZE / "customers.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(customers[0]))
    w.writeheader()
    w.writerows(customers)
print(f"{len(rows):,} events of {N_CUSTOMERS:,} customers -> {BRONZE}: " + ", ".join(f"{s} {len(v):,}" for s, v in by.items()))
