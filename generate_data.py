"""Generate fake CSV datasets for testing the active-learning widget.

Writes three files in the current directory:

- ``labeled_demo.csv``     — 50-row held-out evaluation set with gold labels.
- ``unlabeled_train.csv``  — 200-row pool to label with the active-learning widget.
- ``labeled_train_al.csv`` — empty stub; the widget will populate it.

Labels follow the same schema as the widget:
    {"sentiment": {...}, "topic": {...}}
A row's ``labels`` cell is a JSON-encoded list of "category::label" strings.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


LABELED_DEMO_PATH = Path("labeled_demo.csv")
UNLABELED_TRAIN_PATH = Path("unlabeled_train.csv")
LABELED_TRAIN_AL_PATH = Path("labeled_train_al.csv")

N_LABELED = 50
N_UNLABELED = 200
N_LABELED_TRAIN = 100


LABEL_DICT = {
    "sentiment": {
        "positive": "expresses approval, happiness, or satisfaction",
        "negative": "expresses disapproval, anger, or frustration",
        "neutral":  "factual or descriptive with no clear emotional valence",
    },
    "topic": {
        "pricing":  "mentions cost, price, fees, value for money, expensive or cheap",
        "support":  "mentions customer service, help, agents, response time",
        "quality":  "mentions product quality, defects, durability, materials",
        "shipping": "mentions delivery, shipping speed, packaging, arrival",
    },
}


# --- curated gold set ---------------------------------------------------------
# 50 hand-written (text, [labels]) pairs covering every category × label combo
# plus a few multi-topic and neutral rows.
LABELED_EVAL: list[tuple[str, list[str]]] = [
    # positive × quality
    ("Beautifully built, sturdy materials, feels premium in the hand.",         ["sentiment::positive", "topic::quality"]),
    ("Excellent craftsmanship, no scratch after months of daily use.",          ["sentiment::positive", "topic::quality"]),
    ("Top-notch finish and the metal frame feels rock solid.",                  ["sentiment::positive", "topic::quality"]),
    ("Genuinely well-made, you can see the attention to detail.",               ["sentiment::positive", "topic::quality"]),
    # positive × pricing
    ("Incredible value for the price, can't believe how affordable.",           ["sentiment::positive", "topic::pricing"]),
    ("Cheapest in its class and just as good as the premium brands.",           ["sentiment::positive", "topic::pricing"]),
    ("Best price I've seen for this kind of product, real bargain.",            ["sentiment::positive", "topic::pricing"]),
    ("Honestly a steal at this price point.",                                   ["sentiment::positive", "topic::pricing"]),
    # positive × support
    ("Customer service was outstanding, resolved my issue in minutes.",         ["sentiment::positive", "topic::support"]),
    ("The help desk team went above and beyond to fix my problem.",             ["sentiment::positive", "topic::support"]),
    ("Quick chat response, polite agent, problem solved on first contact.",     ["sentiment::positive", "topic::support"]),
    ("Support rep was patient and walked me through the setup.",                ["sentiment::positive", "topic::support"]),
    # positive × shipping
    ("Arrived a full day earlier than promised, packed perfectly.",             ["sentiment::positive", "topic::shipping"]),
    ("Lightning fast delivery and the box was pristine.",                       ["sentiment::positive", "topic::shipping"]),
    ("Got it next-day, beautifully wrapped, totally impressed.",                ["sentiment::positive", "topic::shipping"]),
    ("Free overnight shipping was a nice surprise.",                            ["sentiment::positive", "topic::shipping"]),

    # negative × quality
    ("Felt cheap and flimsy out of the box, plastic creaks badly.",             ["sentiment::negative", "topic::quality"]),
    ("Stopped working after two weeks, build quality is garbage.",              ["sentiment::negative", "topic::quality"]),
    ("Defective unit, screen flickers and buttons barely register.",            ["sentiment::negative", "topic::quality"]),
    ("Snapped in half on the first drop from desk height.",                     ["sentiment::negative", "topic::quality"]),
    # negative × pricing
    ("Outrageously overpriced for what it actually does, total ripoff.",        ["sentiment::negative", "topic::pricing"]),
    ("Way too expensive, you can get something better for half the cost.",      ["sentiment::negative", "topic::pricing"]),
    ("Hidden fees everywhere, the real cost is double the listed price.",       ["sentiment::negative", "topic::pricing"]),
    ("Charging this much for this is just gouging customers.",                  ["sentiment::negative", "topic::pricing"]),
    # negative × support
    ("Support never responded to my three emails, abandoned customers.",        ["sentiment::negative", "topic::support"]),
    ("The rep was rude, dismissive, refused to help with a simple return.",     ["sentiment::negative", "topic::support"]),
    ("Bounced between five agents, nobody could fix the issue.",                ["sentiment::negative", "topic::support"]),
    ("Customer service hung up on me twice and never called back.",             ["sentiment::negative", "topic::support"]),
    # negative × shipping
    ("Box arrived crushed, item scratched and missing parts.",                  ["sentiment::negative", "topic::shipping"]),
    ("Took three weeks to ship and arrived in a torn envelope.",                ["sentiment::negative", "topic::shipping"]),
    ("Lost in transit twice, finally arrived a month late and damaged.",        ["sentiment::negative", "topic::shipping"]),
    ("Delivery driver left the package outside in the rain.",                   ["sentiment::negative", "topic::shipping"]),

    # neutral (general, no specific topic)
    ("Does what it says on the tin, no surprises either way.",                  ["sentiment::neutral"]),
    ("Standard item, works as listed, nothing remarkable.",                     ["sentiment::neutral"]),
    ("Functions normally, no strong opinion after a week of use.",              ["sentiment::neutral"]),
    ("Works. That's about all I can say.",                                      ["sentiment::neutral"]),
    ("Average product, average everything.",                                    ["sentiment::neutral"]),

    # multi-topic combos
    ("Fast shipping, great build, fair price, and helpful support.",            ["sentiment::positive", "topic::shipping", "topic::quality", "topic::pricing", "topic::support"]),
    ("Defective, expensive, slow to arrive, and support ghosted me.",           ["sentiment::negative", "topic::quality", "topic::pricing", "topic::shipping", "topic::support"]),
    ("Solid value and the agent on chat was a delight.",                        ["sentiment::positive", "topic::pricing", "topic::support"]),
    ("Overpriced, fragile, and shipped late — three strikes.",                  ["sentiment::negative", "topic::pricing", "topic::quality", "topic::shipping"]),
    ("Reasonably priced and the materials hold up under daily use.",            ["sentiment::positive", "topic::pricing", "topic::quality"]),
    ("Charged me twice and the chat bot refused to escalate.",                  ["sentiment::negative", "topic::pricing", "topic::support"]),
    ("Beautifully packaged and the build feels premium too.",                   ["sentiment::positive", "topic::shipping", "topic::quality"]),
    ("Support refused a refund despite the obvious defect.",                    ["sentiment::negative", "topic::support", "topic::quality"]),

    # extra singletons
    ("Absolutely love it, best purchase of the year by a mile.",                ["sentiment::positive"]),
    ("Hands down the worst product I've ever bought, waste of money.",          ["sentiment::negative"]),
    ("Packaging was eco-friendly which I appreciate.",                          ["sentiment::positive", "topic::shipping"]),
    ("Loud, brittle, way too expensive — sending it back.",                     ["sentiment::negative", "topic::quality", "topic::pricing"]),
    ("Quiet, durable, well-priced — buying a second one.",                      ["sentiment::positive", "topic::quality", "topic::pricing"]),
]


# --- templated pool for unlabeled rows ----------------------------------------

POS_QUALITY = [
    "The {build} feels {pos_qadj}, very happy with it.",
    "{pos_qadv} {pos_qadj} {build}, would buy again.",
    "Honestly the {material} feels {pos_qadj_alt}.",
    "Solid {material}, {pos_qadj} construction, no complaints.",
    "After {time_long} of use the {build} still looks {pos_qadj_alt}.",
    "Premium feel for sure, the {material} is {pos_qadj}.",
]
NEG_QUALITY = [
    "The {build} feels {neg_qadj}, very disappointed.",
    "{neg_qadj} {build}, {neg_qact} after a {time_short}.",
    "Honestly the {material} feels {neg_qadj_alt}.",
    "Cheap {material}, {neg_qadj} build, won't last.",
    "The {build} {neg_qact} on day one.",
    "Wobbly {build}, the {material} is {neg_qadj}.",
]
POS_PRICING = [
    "Incredible {price_word} for what you get.",
    "Best {price_word} I've seen, total {pos_deal}.",
    "{pos_padj} {price_word}, lots of bang for your buck.",
    "Worth every penny, genuine {pos_deal}.",
    "Cheaper than I expected and still {pos_quick}.",
    "Hard to beat the {price_word} on this one.",
]
NEG_PRICING = [
    "{neg_padj} for what you actually get, feels like a {neg_deal}.",
    "Total {neg_deal}, the {price_word} is absurd.",
    "Way too {neg_padj}, you can find similar for half the {price_word}.",
    "Hidden {fee_word} push the {price_word} way up.",
    "{neg_padj} {price_word} and the value just isn't there.",
    "Felt {neg_padj}, especially after the {fee_word} kicked in.",
]
POS_SUPPORT = [
    "The {support} was {pos_sadj}, resolved my issue right away.",
    "{pos_sadj} {support}, response came in {time_short}.",
    "Got help from a {pos_sadj} {agent}, very impressed.",
    "The {support} team {pos_saction}, big thumbs up.",
    "Quick and {pos_sadj} {support}, problem solved.",
    "Talked to an {agent} who was {pos_sadj} and helpful.",
]
NEG_SUPPORT = [
    "The {support} was {neg_sadj}, no real help.",
    "{neg_sadj} {support}, I {neg_saction}.",
    "The {agent} was {neg_sadj} and {neg_saction_alt}.",
    "Tried {support_action} {ncount} times, no response.",
    "Useless {support}, {neg_saction}.",
    "After {ncount} attempts, the {support} {neg_saction_alt}.",
]
POS_SHIPPING = [
    "Arrived {pos_shadv}, {pos_shextra}.",
    "{pos_shadv} shipping and the {pkg} was {pos_pkadj}.",
    "Got it in {time_short}, {pos_pkadj} packaging.",
    "Fast delivery, {pos_pkadj} {pkg}, nothing to complain about.",
    "{pos_shadv} arrival, item in {pos_pkadj} condition.",
]
NEG_SHIPPING = [
    "Took {time_long} to arrive, {neg_shextra}.",
    "{neg_shadv} shipping, the {pkg} was {neg_pkadj}.",
    "Arrived {time_long} after the promised date, {neg_pkact}.",
    "The {pkg} was {neg_pkadj}, item {neg_arrival}.",
    "{neg_shadv} delivery, {neg_pkact}.",
]
NEUTRAL = [
    "Comes with {ncount} {accessory} in the box.",
    "Available in {ncount} {color_word}.",
    "About the size of a {compare_obj}.",
    "Manual is {manual_form}.",
    "Compatible with most {compat}.",
    "Weighs roughly {weight}.",
    "Includes a {warranty_dur} warranty.",
    "Battery lasts about a day on a charge.",
    "Comes pre-assembled in the box.",
    "Made of recyclable {material}.",
]
AMBIGUOUS = [
    "{ambig_pos} but {ambig_neg}.",
    "The {ambig_topic} was {ambig_adj}, though {ambig_caveat}.",
    "{ambig_pos}, however {ambig_neg}.",
    "Mixed feelings — {ambig_pos} but {ambig_neg}.",
    "{ambig_adj} {ambig_topic}, but {ambig_caveat}.",
]

BUCKETS = [
    POS_QUALITY, NEG_QUALITY,
    POS_PRICING, NEG_PRICING,
    POS_SUPPORT, NEG_SUPPORT,
    POS_SHIPPING, NEG_SHIPPING,
    NEUTRAL, AMBIGUOUS,
]


# Per-bucket label assignments — used to produce labeled rows from the same
# templates. Ambiguous templates are skipped because their labels aren't clean.
LABELED_BUCKETS: dict[str, tuple[list[str], list[str]]] = {
    "pos_quality":  (POS_QUALITY,  ["sentiment::positive", "topic::quality"]),
    "neg_quality":  (NEG_QUALITY,  ["sentiment::negative", "topic::quality"]),
    "pos_pricing":  (POS_PRICING,  ["sentiment::positive", "topic::pricing"]),
    "neg_pricing":  (NEG_PRICING,  ["sentiment::negative", "topic::pricing"]),
    "pos_support":  (POS_SUPPORT,  ["sentiment::positive", "topic::support"]),
    "neg_support":  (NEG_SUPPORT,  ["sentiment::negative", "topic::support"]),
    "pos_shipping": (POS_SHIPPING, ["sentiment::positive", "topic::shipping"]),
    "neg_shipping": (NEG_SHIPPING, ["sentiment::negative", "topic::shipping"]),
    "neutral":      (NEUTRAL,      ["sentiment::neutral"]),
}

SLOTS = {
    "build":          ["build", "construction", "design", "casing", "frame", "chassis"],
    "material":       ["plastic", "metal", "aluminum casing", "fabric trim", "rubber grip", "leather finish"],
    "pos_qadj":       ["sturdy", "robust", "solid", "premium", "well-made", "rock-solid"],
    "pos_qadj_alt":   ["surprisingly nice", "really substantial", "high-end", "rock solid"],
    "pos_qadv":       ["Genuinely", "Really", "Honestly", "Surprisingly"],
    "neg_qadj":       ["flimsy", "cheap", "fragile", "brittle", "shoddy", "rickety"],
    "neg_qadj_alt":   ["hollow", "weirdly light", "poorly assembled", "rough around the edges"],
    "neg_qact":       ["snapped", "fell apart", "stopped working", "broke", "started failing"],
    "time_short":     ["minutes", "an hour", "a day", "two days", "a week"],
    "time_long":      ["three weeks", "a month", "ten days", "two weeks", "six weeks"],
    "ncount":         ["three", "four", "five", "two"],
    "price_word":     ["price", "cost", "value"],
    "pos_deal":       ["bargain", "steal", "great deal", "winner"],
    "pos_padj":       ["affordable", "budget-friendly", "reasonable", "wallet-friendly"],
    "pos_quick":      ["solidly built", "well-made", "fast to ship"],
    "neg_padj":       ["overpriced", "expensive", "pricey", "outrageously expensive"],
    "neg_deal":       ["ripoff", "scam", "racket"],
    "fee_word":       ["fees", "charges", "surcharges", "processing fees"],
    "support":        ["customer service", "support team", "help desk", "chat support"],
    "agent":          ["rep", "agent", "representative", "operator"],
    "pos_sadj":       ["helpful", "knowledgeable", "friendly", "responsive", "attentive", "patient"],
    "pos_saction":    ["went above and beyond", "fixed my problem fast", "explained everything clearly", "called me back as promised"],
    "neg_sadj":       ["rude", "unhelpful", "useless", "dismissive", "abrasive"],
    "neg_saction":    ["was ignored", "got no response", "was disconnected"],
    "neg_saction_alt":["wouldn't help", "kept transferring me", "hung up on me"],
    "support_action": ["emailing", "calling", "messaging them"],
    "pos_shadv":      ["incredibly fast", "earlier than expected", "lightning fast", "ahead of schedule"],
    "pos_shextra":    ["packaging was great", "nothing damaged", "perfect condition", "very satisfied"],
    "pos_pkadj":      ["pristine", "intact", "perfect", "well-padded"],
    "pkg":            ["box", "package", "packaging"],
    "neg_shextra":    ["very frustrating", "no updates given", "no communication at all"],
    "neg_shadv":      ["Glacial", "Painfully slow", "Awful"],
    "neg_pkadj":      ["crushed", "torn", "soaked", "destroyed", "battered"],
    "neg_pkact":      ["item was scratched", "parts were missing", "item was damaged"],
    "neg_arrival":    ["was dented", "arrived broken", "was scratched"],
    "accessory":      ["cable", "manual", "adapter", "charger", "wall mount"],
    "color_word":     ["colors", "finishes"],
    "compare_obj":    ["paperback book", "wallet", "smartphone", "remote control"],
    "manual_form":    ["a single page", "available online", "in five languages", "QR-coded on the box"],
    "compat":         ["devices", "phones", "modern platforms"],
    "weight":         ["200 grams", "half a kilo", "a pound", "300 grams"],
    "warranty_dur":   ["one-year", "two-year", "lifetime", "limited two-year"],
    "ambig_pos":      ["The build is solid", "Cheap price", "Fast shipping", "Friendly support", "Packaging was nice"],
    "ambig_neg":      ["the price keeps going up", "the materials feel cheap", "support couldn't actually help", "delivery took forever", "the item was damaged"],
    "ambig_topic":    ["agent", "shipping", "product", "experience"],
    "ambig_adj":      ["friendly", "fast", "cheap", "decent"],
    "ambig_caveat":   ["I'm still unsure", "nothing was really resolved", "it didn't help much", "I'd hesitate to recommend"],
}


def _fill(template: str, rng: random.Random) -> str:
    out = template
    while "{" in out:
        start = out.find("{")
        end = out.find("}", start)
        slot = out[start + 1:end]
        if slot not in SLOTS:
            raise KeyError(f"unknown slot {{{slot}}} in template: {template}")
        out = out[:start] + rng.choice(SLOTS[slot]) + out[end + 1:]
    return out


def generate_unlabeled(n: int, seed: int = 1) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    safety = 0
    while len(out) < n and safety < n * 50:
        bucket = rng.choice(BUCKETS)
        tmpl = rng.choice(bucket)
        text = _fill(tmpl, rng)
        if text not in seen:
            seen.add(text)
            out.append(text)
        safety += 1
    if len(out) < n:
        raise RuntimeError(
            f"only produced {len(out)} unique texts after {safety} tries; "
            f"add more slot values or templates"
        )
    return out


def generate_labeled(
    n: int,
    seed: int = 2,
    avoid: set[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Produce ``n`` unique (text, [label_key]) pairs from LABELED_BUCKETS."""
    rng = random.Random(seed)
    bucket_names = list(LABELED_BUCKETS.keys())
    seen: set[str] = set(avoid or ())
    out: list[tuple[str, list[str]]] = []
    safety = 0
    while len(out) < n and safety < n * 80:
        name = rng.choice(bucket_names)
        templates, labels = LABELED_BUCKETS[name]
        text = _fill(rng.choice(templates), rng)
        if text not in seen:
            seen.add(text)
            out.append((text, list(labels)))
        safety += 1
    if len(out) < n:
        raise RuntimeError(
            f"only produced {len(out)} unique labeled texts after {safety} tries"
        )
    return out


def main() -> None:
    if len(LABELED_EVAL) < N_LABELED:
        raise RuntimeError(
            f"LABELED_EVAL has {len(LABELED_EVAL)} rows; want {N_LABELED}"
        )
    eval_rows = LABELED_EVAL[:N_LABELED]
    eval_df = pd.DataFrame(
        {
            "text": [t for t, _ in eval_rows],
            "labels": [json.dumps(lbls) for _, lbls in eval_rows],
        }
    ).sample(frac=1, random_state=0).reset_index(drop=True)
    eval_df.to_csv(LABELED_DEMO_PATH, index=False)

    pool_texts = generate_unlabeled(N_UNLABELED, seed=1)
    rng = random.Random(0)
    rng.shuffle(pool_texts)
    unlabeled_df = pd.DataFrame(
        {
            "text": pool_texts,
            "labels": [json.dumps([]) for _ in pool_texts],
        }
    )
    unlabeled_df.to_csv(UNLABELED_TRAIN_PATH, index=False)

    # Avoid duplicates against the eval and unlabeled sets so the labeled-train
    # rows occupy their own slice of the embedding space.
    forbidden = set(eval_df["text"]).union(set(pool_texts))
    labeled_train_rows = generate_labeled(
        N_LABELED_TRAIN, seed=2, avoid=forbidden
    )
    labeled_train_df = pd.DataFrame(
        {
            "text": [t for t, _ in labeled_train_rows],
            "labels": [json.dumps(lbls) for _, lbls in labeled_train_rows],
        }
    ).sample(frac=1, random_state=0).reset_index(drop=True)
    labeled_train_df.to_csv(LABELED_TRAIN_AL_PATH, index=False)

    print(f"wrote {LABELED_DEMO_PATH}    ({len(eval_df)} rows, labeled)")
    print(f"wrote {UNLABELED_TRAIN_PATH} ({len(unlabeled_df)} rows, unlabeled)")
    print(f"wrote {LABELED_TRAIN_AL_PATH} ({len(labeled_train_df)} rows, labeled)")


if __name__ == "__main__":
    main()
