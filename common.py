"""Shared by the journeyModeller pipelines: folders, the split, how raw source codes map to events, reading silver."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BRONZE, SILVER, GOLD = DATA / "bronze", DATA / "silver", DATA / "gold"
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
SEED, TEST_SHARE = 0, 0.2
DAY_US = 86_400_000_000

# event -> (source system, its raw code in that system's export). Silver maps the codes back.
RAW = {"email_sent": ("crm", "EMAIL_SENT"), "email_opened": ("crm", "EMAIL_OPEN"), "push_sent": ("crm", "PUSH_SENT"),
       "push_opened": ("crm", "PUSH_OPEN"), "call_made": ("crm", "CALL_OUT"), "call_answered": ("crm", "CALL_CONNECTED"),
       "banner_shown": ("crm", "BANNER_IMPRESSION"), "banner_clicked": ("crm", "BANNER_CLICK"), "unsubscribed": ("crm", "UNSUBSCRIBE"),
       "page_visited": ("app", "page_view"), "application": ("app", "application_submitted"),
       "contract_signed": ("core", "CONTRACT"), "account_opened": ("core", "ACCOUNT_OPEN"), "contract_terminated": ("core", "CONTRACT_END")}
SOURCE = {"crm": "communication", "app": "app", "core": "transaction"}  # pragmatiq's event sources, per system
CONTENT_FIELD = {"email": "subject", "push": "message", "phone": "script", "in_app": "banner"}  # the CRM's content column


def utf8() -> None:
    """ponytail: pragmatiq reads its files in the platform encoding; on Windows, re-run this script in UTF-8 mode."""
    if os.name == "nt" and not sys.flags.utf8_mode:
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))


def read_silver(path: Path) -> tuple[dict, list]:
    """One customer's silver file -> (profile attributes, [(time in microseconds, source, fields)])."""
    import numpy as np

    attrs, events = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        head, *rest = line.split("\t")
        if head == "profile":
            attrs = dict(x.split("=", 1) for x in rest)
        else:
            source, event, *kv = rest
            events.append((int(np.datetime64(head, "us").astype(np.int64)), source, {"event": event, **dict(x.split("=", 1) for x in kv)}))
    return attrs, events
