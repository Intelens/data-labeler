"""Optimize a multi-label comment classifier with DSPy, vLLM loaded in-process.

No HTTP server: the vLLM engine runs inside this process behind a tiny custom
DSPy LM. (The server version is `dspy_optimize.py`.)

    pip install dspy vllm
    python dspy_optimize_offline.py

Reuses labeled_demo.csv (text + a stringified label list). The label set is
derived from the data; DSPy (MIPROv2) tunes the instruction + few-shot demos.
"""
import ast
import os
import random
from types import SimpleNamespace

import dspy
import pandas as pd
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DATA_PATH = "labeled_demo.csv"


# ── in-process vLLM behind DSPy's LM interface ───────────────────────────
class VLLMOffline(dspy.BaseLM):
    """Runs a vLLM engine in this process (no server); DSPy reads .choices[].message.content.

    ponytail: legacy forward contract; if DSPy warns, pass forward_contract='legacy'
    to super().__init__ or move to the typed LMRequest/LMResponse API.
    """

    def __init__(self, model, temperature=0.0, max_tokens=1000):
        super().__init__(model=model)
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model)              # weights load here, in-process
        self.SamplingParams = SamplingParams
        self.temperature, self.max_tokens = temperature, max_tokens

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        params = self.SamplingParams(
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        text = self.llm.chat(messages, params)[0].outputs[0].text
        return ChatCompletion(
            id="vllm", created=0, model=self.model, object="chat.completion",
            choices=[Choice(index=0, finish_reason="stop",
                            message=ChatCompletionMessage(role="assistant", content=text))],
        )


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
    dspy.configure(lm=VLLMOffline(MODEL))     # loads the model now
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
