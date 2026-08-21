"""QLoRA fine-tune a locally stored Qwen3-4B for structured extraction.

Input is one dataframe: a text column holds the document, every other column is a
field to extract. Column names are the schema -- no separate schema file to drift.

  python extract_qlora.py data/invoices.parquet --dry-run   # inspect one example, no GPU
  python extract_qlora.py data/invoices.parquet             # train
  python extract_qlora.py data/invoices.parquet --eval      # field-level score on held-out
"""
import json, os, sys

import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

load_dotenv()
BASE = os.environ.get("BASE_MODEL", r"models/Qwen3-4B")  # local path, not a hub id
ADAPTER = os.environ.get("ADAPTER_DIR", "runs/extract-qlora")
TEXT_COL = os.environ.get("TEXT_COL", "text")

LR = float(os.environ.get("LR", "1e-4"))
EPOCHS = float(os.environ.get("EPOCHS", "3"))
BATCH = int(os.environ.get("BATCH", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
EVAL_FRAC = float(os.environ.get("EVAL_FRAC", "0.1"))
SEED = int(os.environ.get("SEED", "42"))

PROMPT = """Extract the following fields from the document and return a single JSON object.

Fields: {fields}

Rules:
- Use exactly these keys, in this order.
- Use null when a field is not present in the document. Never guess.
- Return only the JSON object.

Document:
{text}"""


def load_frame(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    if TEXT_COL not in df.columns:
        sys.exit(f"no column {TEXT_COL!r} in {path}; columns are {list(df.columns)}")
    fields = [c for c in df.columns if c != TEXT_COL]
    if not fields:
        sys.exit("dataframe has no target columns besides the text column")
    return df, fields


def target_json(row, fields):
    # NaN -> null, so absent fields train the model to abstain instead of hallucinate.
    return json.dumps({f: (None if pd.isna(row[f]) else row[f]) for f in fields},
                      ensure_ascii=False, default=str)


def build(df, fields, tok):
    """-> Dataset of input_ids/labels, loss masked to the JSON answer only."""
    def encode(row):
        msgs = [{"role": "user", "content": PROMPT.format(fields=", ".join(fields),
                                                          text=row[TEXT_COL])}]
        # enable_thinking=False: extraction wants the JSON, not a reasoning trace.
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        ids = tok(prompt + target_json(row, fields) + tok.eos_token,
                  add_special_tokens=False).input_ids[:MAX_LEN]
        # Train on the answer only. Without this the model spends its capacity learning
        # to reproduce the prompt, and extraction accuracy drops hard.
        labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
        return {"input_ids": ids, "labels": labels}

    rows = [encode(r) for _, r in df.iterrows()]
    dropped = sum(1 for r in rows if all(l == -100 for l in r["labels"]))
    if dropped:
        print(f"warning: {dropped} rows truncated past their answer at MAX_LEN={MAX_LEN}")
    return Dataset.from_list(rows)


def split(df):
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = max(1, int(len(df) * EVAL_FRAC))
    return df.iloc[n:].reset_index(drop=True), df.iloc[:n].reset_index(drop=True)


def load_model(tok, adapter=None):
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map="auto",
    )
    return PeftModel.from_pretrained(model, adapter) if adapter else model


def train(df, fields, tok):
    tr, ev = split(df)
    model = prepare_model_for_kbit_training(load_model(tok))
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=ADAPTER, learning_rate=LR, num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
            per_device_eval_batch_size=1, lr_scheduler_type="cosine", warmup_ratio=0.03,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit", bf16=True, seed=SEED,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            logging_steps=10, report_to=[],
        ),
        train_dataset=build(tr, fields, tok), eval_dataset=build(ev, fields, tok),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    ).train()
    model.save_pretrained(ADAPTER)
    tok.save_pretrained(ADAPTER)
    print(f"adapter -> {ADAPTER}")


def evaluate(df, fields, tok):
    _, ev = split(df)
    model = load_model(tok, ADAPTER).eval()
    hits = {f: 0 for f in fields}
    parse_fails = 0

    for _, row in ev.iterrows():
        msgs = [{"role": "user", "content": PROMPT.format(fields=", ".join(fields),
                                                          text=row[TEXT_COL])}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        try:
            pred = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            parse_fails += 1
            continue
        gold = json.loads(target_json(row, fields))
        for f in fields:
            # ponytail: exact match on the stringified value. Add per-field normalisation
            # (dates, decimal commas, VAT ids) once a client's schema needs it.
            if str(pred.get(f)).strip() == str(gold[f]).strip():
                hits[f] += 1

    n = len(ev)
    print(f"held-out: {n} docs, {parse_fails} unparseable generations")
    for f in fields:
        print(f"  {f:<28} {hits[f] / n:.4f}")
    print(f"  {'ALL FIELDS (micro)':<28} {sum(hits.values()) / (n * len(fields)):.4f}")


if __name__ == "__main__":
    path = sys.argv[1]
    df, fields = load_frame(path)
    tok = AutoTokenizer.from_pretrained(BASE)
    print(f"{len(df)} rows, {len(fields)} fields: {fields}")

    if "--dry-run" in sys.argv:
        ds = build(df.head(1), fields, tok)
        ids, labels = ds[0]["input_ids"], ds[0]["labels"]
        supervised = tok.decode([i for i, l in zip(ids, labels) if l != -100])
        print(f"\n--- prompt+answer ({len(ids)} tokens) ---\n{tok.decode(ids)}")
        print(f"\n--- supervised span ---\n{supervised}")
        assert supervised.strip().startswith("{"), "loss mask does not start at the JSON"
        assert json.loads(supervised.replace(tok.eos_token, "")), "supervised span is not valid JSON"
        print("\nmask ok: loss covers exactly the JSON answer")
    elif "--eval" in sys.argv:
        evaluate(df, fields, tok)
    else:
        train(df, fields, tok)
