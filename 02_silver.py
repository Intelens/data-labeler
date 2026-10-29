"""Pipeline 2 · Encode events -> data/silver, one text file per customer.

Every raw row, whatever system it came from, becomes one event as PRAGMA represents it: a time, a source and a set
of key=value fields (the event name is the field "event"). Times become UTC with microseconds, raw codes become event
names, the CRM's content column becomes subject / message / script / banner by channel, milliseconds become seconds,
the app's duplicate sends are dropped, and every customer's events are sorted in time.

Reads  data/bronze/   (pipeline 1)
Writes data/silver/c_00000.txt ...   first line:  profile <TAB> age=54 <TAB> plan=plus ...
                                     then one event per line, in time order:
                                     2026-03-02T08:14:00.123456 <TAB> communication <TAB> email_sent <TAB> topic=mortgage <TAB> ...

Run:  python pipelines/02_silver.py
"""
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np
from common import BRONZE, CONTENT_FIELD, RAW, SILVER, SOURCE

EVENT = {(system, code): event for event, (system, code) in RAW.items()}
events, unknown = defaultdict(set), defaultdict(int)  # customer -> {(time in microseconds, source, event, fields)}


def add(cid, ts_us, system, code, fields):
    event = EVENT.get((system, code))
    if event is None:
        unknown[(system, code)] += 1
        return
    clean = tuple((k, str(v).replace("\t", " ").replace("\n", " ")) for k, v in fields.items() if v not in ("", None))
    events[cid].add((ts_us, SOURCE[system], event, clean))  # a set: exact duplicates disappear


n_raw = 0
with open(BRONZE / "crm_contacts.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        n_raw += 1
        fields = {"topic": r["topic"], "channel": r["channel"]}
        if r["content"]:
            fields[CONTENT_FIELD[r["channel"]]] = r["content"]
        fields.update(json.loads(r["details"]) if r["details"] else {})
        add(r["customer_id"], int(np.datetime64(r["event_time"].rstrip("Z"), "us").astype(np.int64)), "crm", r["event_type"], fields)
with open(BRONZE / "app_events.jsonl", encoding="utf-8") as fh:
    for line in fh:
        n_raw += 1
        r = json.loads(line)
        p = r["props"]
        fields = {"topic": p.get("topic"), "device": p.get("device"),
                  "dwell_seconds": round(p["dwell_ms"] / 1000) if "dwell_ms" in p else None}
        add(r["user"], r["ts"] * 1000, "app", r["name"], fields)
with open(BRONZE / "core_banking.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter=";"):
        n_raw += 1
        ts_us = int(np.datetime64(datetime.strptime(r["booked_at"], "%d-%m-%Y %H:%M:%S"), "us").astype(np.int64))  # booked in UTC
        fields = {"topic": r["product"], "device": r["channel"], "contract_value": r["amount_eur"], "term_months": r["term_months"],
                  "product_status": r["status"], "reason": r["reason"]}
        add(r["customer_id"], ts_us, "core", r["type"], fields)
with open(BRONZE / "customers.csv", newline="", encoding="utf-8") as fh:
    profiles = {r.pop("customer_id"): r for r in csv.DictReader(fh)}

shutil.rmtree(SILVER, ignore_errors=True)
SILVER.mkdir(parents=True)
n_events = 0
for cid in sorted(set(events) | set(profiles)):
    lines = ["\t".join(["profile", *(f"{k}={v}" for k, v in profiles.get(cid, {}).items())])]
    for ts_us, source, event, fields in sorted(events[cid], key=lambda e: (e[0], e[2])):
        lines.append("\t".join([str(np.datetime64(ts_us, "us")), source, event, *(f"{k}={v}" for k, v in fields)]))
    (SILVER / f"{cid}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_events += len(events[cid])
print(f"{n_raw:,} raw rows -> {n_events:,} events ({n_raw - n_events:,} duplicates dropped) of {len(profiles):,} customers -> {SILVER}")
for (system, code), n in unknown.items():
    print(f"warning: {n} rows with unknown code {code!r} from {system}")
