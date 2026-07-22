"""Optimize a multi-label comment classifier with DSPy, LM served by local vLLM.

Start the vLLM OpenAI-compatible server first:
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
Then:
    python dspy_optimize.py

Reuses labeled_demo.csv (text + a stringified label list). The label set is
derived from the data; DSPy (MIPROv2) tunes the instruction + few-shot demos.
"""
import ast
import random
from types import SimpleNamespace

import dspy
import pandas as pd

# ── config ───────────────────────────────────────────────────────────────
MODEL = "openai/Qwen/Qwen2.5-7B-Instruct"     # any model your vLLM is serving
API_BASE = "http://localhost:8000/v1"
DATA_PATH = "labeled_demo.csv"

dspy.configure(lm=dspy.LM(MODEL, api_base=API_BASE, api_key="EMPTY", model_type="chat"))

# ── data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["labels"] = df["labels"].apply(ast.literal_eval)          # "['a','b']" -> ['a','b']
ALLOWED = sorted({l for labs in df["labels"] for l in labs})  # label vocab from the data

examples = [dspy.Example(comment=t, labels=labs).with_inputs("comment")
            for t, labs in zip(df["text"], df["labels"])]
random.Random(42).shuffle(examples)
trainset, testset = examples[:30], examples[30:]


# ── program ──────────────────────────────────────────────────────────────
class Classify(dspy.Signature):
    """Assign every applicable label to the customer comment."""
    comment: str = dspy.InputField()
    labels: list[str] = dspy.OutputField(desc=f"a subset of {ALLOWED}")


classify = dspy.Predict(Classify)


# ── metric: multi-label F1 ───────────────────────────────────────────────
def f1(example, pred, trace=None):
    gold, got = set(example.labels), set(getattr(pred, "labels", []) or [])
    if not gold and not got:
        return 1.0
    tp = len(gold & got)
    prec = tp / len(got) if got else 0.0
    rec = tp / len(gold) if gold else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _check_metric():
    ex = lambda l: dspy.Example(comment="x", labels=l)
    pr = lambda l: SimpleNamespace(labels=l)
    assert f1(ex(["a", "b"]), pr(["a", "b"])) == 1.0
    assert abs(f1(ex(["a", "b"]), pr(["a"])) - 2 / 3) < 1e-9     # prec 1, rec 0.5
    assert f1(ex(["a"]), pr(["b"])) == 0.0
    assert f1(ex([]), pr([])) == 1.0
    print("metric ok")


# ── optimize + evaluate ──────────────────────────────────────────────────
def main():
    _check_metric()
    from dspy.teleprompt import MIPROv2

    evaluate = dspy.Evaluate(devset=testset, metric=f1, display_progress=True)
    print("baseline:", evaluate(classify))

    optimizer = MIPROv2(metric=f1, auto="light")     # tunes instruction + demos
    optimized = optimizer.compile(classify, trainset=trainset,
                                  requires_permission_to_run=False)

    print("optimized:", evaluate(optimized))
    optimized.save("classify_optimized.json")
    # swap MIPROv2 for the lighter BootstrapFewShot(metric=f1) if you want fewer LM calls.


if __name__ == "__main__":
    main()
