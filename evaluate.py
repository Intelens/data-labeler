# train_qwen3_cross_encoder.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
OUTPUT_DIR = Path("./qwen3-product-tag-cross-encoder")

MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

VALIDATION_FRACTION = 0.20
NEGATIVES_PER_POSITIVE = 3
RANDOM_SEED = 42

# Use descriptions instead of bare tag IDs where possible.
TAG_DESCRIPTIONS: dict[str, str] = {
    "clothing": "Clothing and wearable apparel",
    "outdoor": "Products intended for outdoor activities",
    "waterproof": "Products designed to resist or prevent water penetration",
    "electronics": "Electronic devices and accessories",
    "gaming": "Products intended for video gaming",
    "computer accessories": "Accessories and peripherals for computers",
    "food": "Food and edible grocery products",
    "coffee": "Coffee beans, ground coffee, or coffee products",
    "organic": "Products produced according to organic standards",
}


# ============================================================
# Example input data
# Replace these lists with your real data.
# ============================================================

train_texts = [
    "Lightweight waterproof hiking jacket",
    "Wireless gaming mouse with RGB lighting",
    "Organic dark roast coffee beans",
    "Insulated rain coat for mountain trekking",
    "Ergonomic mechanical keyboard for PC gaming",
    "Single-origin medium roast coffee",
    "Breathable running shirt for outdoor exercise",
    "Bluetooth computer mouse with programmable buttons",
    "Organic whole bean espresso blend",
    "Water-resistant winter jacket with hood",
]

train_tags = [
    ["clothing", "outdoor", "waterproof"],
    ["electronics", "gaming", "computer accessories"],
    ["food", "coffee", "organic"],
    ["clothing", "outdoor", "waterproof"],
    ["electronics", "gaming", "computer accessories"],
    ["food", "coffee"],
    ["clothing", "outdoor"],
    ["electronics", "computer accessories"],
    ["food", "coffee", "organic"],
    ["clothing", "outdoor", "waterproof"],
]


# ============================================================
# Validation helpers
# ============================================================

def validate_input_data(
    texts: Sequence[str],
    tags_per_text: Sequence[Sequence[str]],
) -> None:
    if len(texts) != len(tags_per_text):
        raise ValueError(
            "train_texts and train_tags must contain the same number "
            f"of elements. Received {len(texts)} texts and "
            f"{len(tags_per_text)} tag lists."
        )

    if len(texts) < 2:
        raise ValueError("At least two training texts are required.")

    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Text at index {index} must be a non-empty string."
            )

    for index, tags in enumerate(tags_per_text):
        if not tags:
            raise ValueError(
                f"The text at index {index} has no positive tags."
            )

        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError(
                    f"Tag {tag!r} at index {index} must be "
                    "a non-empty string."
                )


def normalize_training_data(
    texts: Sequence[str],
    tags_per_text: Sequence[Sequence[str]],
) -> tuple[list[str], list[list[str]]]:
    normalized_texts: list[str] = []
    normalized_tags: list[list[str]] = []

    for text, tags in zip(texts, tags_per_text):
        clean_text = text.strip()

        # Remove duplicate tags while preserving order.
        clean_tags = list(
            dict.fromkeys(tag.strip() for tag in tags if tag.strip())
        )

        normalized_texts.append(clean_text)
        normalized_tags.append(clean_tags)

    return normalized_texts, normalized_tags


# ============================================================
# Data splitting
# ============================================================

def split_original_examples(
    texts: Sequence[str],
    tags_per_text: Sequence[Sequence[str]],
    validation_fraction: float,
    seed: int,
) -> tuple[
    list[str],
    list[list[str]],
    list[str],
    list[list[str]],
]:
    """
    Split original product examples before generating text-tag pairs.

    This prevents pairs derived from the same product text from appearing
    in both the training and validation sets.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    indices = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    validation_size = max(
        1,
        int(round(len(indices) * validation_fraction)),
    )

    # Ensure that at least one example remains for training.
    validation_size = min(validation_size, len(indices) - 1)

    validation_indices = set(indices[:validation_size])
    training_indices = indices[validation_size:]

    training_texts = [texts[index] for index in training_indices]
    training_tags = [
        list(tags_per_text[index])
        for index in training_indices
    ]

    validation_texts = [
        texts[index]
        for index in indices
        if index in validation_indices
    ]
    validation_tags = [
        list(tags_per_text[index])
        for index in indices
        if index in validation_indices
    ]

    return (
        training_texts,
        training_tags,
        validation_texts,
        validation_tags,
    )


# ============================================================
# Pair generation
# ============================================================

def format_tag(tag: str) -> str:
    description = TAG_DESCRIPTIONS.get(tag)

    if description:
        return f"{tag}: {description}"

    return tag


def build_pair_dataset(
    texts: Sequence[str],
    tags_per_text: Sequence[Sequence[str]],
    all_tags: Sequence[str],
    negatives_per_positive: int,
    seed: int,
    include_all_negatives: bool = False,
) -> Dataset:
    """
    Convert multilabel examples into binary text-tag pairs.

    Positive pair:
        product text + one true tag -> label 1

    Negative pair:
        product text + one false tag -> label 0
    """
    rng = random.Random(seed)

    rows: dict[str, list] = {
        "product_text": [],
        "tag_text": [],
        "labels": [],
    }

    for product_text, positive_tag_list in zip(texts, tags_per_text):
        positive_tags = set(positive_tag_list)

        unknown_tags = positive_tags.difference(all_tags)
        if unknown_tags:
            raise ValueError(
                f"Unknown tags found: {sorted(unknown_tags)}"
            )

        # Add every positive pair.
        for tag in sorted(positive_tags):
            rows["product_text"].append(product_text)
            rows["tag_text"].append(format_tag(tag))
            rows["labels"].append(1.0)

        negative_pool = [
            tag
            for tag in all_tags
            if tag not in positive_tags
        ]

        if include_all_negatives:
            selected_negatives = negative_pool
        else:
            requested_count = (
                negatives_per_positive * len(positive_tags)
            )
            negative_count = min(
                requested_count,
                len(negative_pool),
            )

            selected_negatives = rng.sample(
                negative_pool,
                negative_count,
            )

        for tag in selected_negatives:
            rows["product_text"].append(product_text)
            rows["tag_text"].append(format_tag(tag))
            rows["labels"].append(0.0)

    return Dataset.from_dict(rows)


# ============================================================
# Custom collator
# ============================================================

@dataclass
class BinaryPairDataCollator:
    """
    Dynamically pads each batch and explicitly preserves correct dtypes.

    input_ids:      torch.int64
    attention_mask: torch.int64
    labels:         torch.float32
    """

    tokenizer: AutoTokenizer

    def __post_init__(self) -> None:
        self.padding_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )

    def __call__(
        self,
        features: list[dict],
    ) -> dict[str, torch.Tensor]:
        labels = [
            float(feature.pop("labels"))
            for feature in features
        ]

        batch = self.padding_collator(features)

        # Embedding indices must be integer tensors.
        batch["input_ids"] = batch["input_ids"].to(
            dtype=torch.long
        )

        if "attention_mask" in batch:
            batch["attention_mask"] = batch[
                "attention_mask"
            ].to(dtype=torch.long)

        if "position_ids" in batch:
            batch["position_ids"] = batch[
                "position_ids"
            ].to(dtype=torch.long)

        if "token_type_ids" in batch:
            batch["token_type_ids"] = batch[
                "token_type_ids"
            ].to(dtype=torch.long)

        # Shape: [batch_size, 1]
        batch["labels"] = torch.tensor(
            labels,
            dtype=torch.float32,
        ).reshape(-1, 1)

        return batch


# ============================================================
# Metrics
# ============================================================

def sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction

    # Some models return logits inside a tuple.
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    probabilities = sigmoid_numpy(logits)
    predictions = (probabilities >= 0.5).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    # ROC AUC is undefined when validation labels contain only one class.
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(labels, probabilities)
        )

    return metrics


# ============================================================
# Training
# ============================================================

def train_model(
    texts: Sequence[str],
    tags_per_text: Sequence[Sequence[str]],
) -> tuple[
    AutoModelForSequenceClassification,
    AutoTokenizer,
    list[str],
]:
    validate_input_data(texts, tags_per_text)

    texts, tags_per_text = normalize_training_data(
        texts,
        tags_per_text,
    )

    all_tags = sorted(
        {
            tag
            for tags in tags_per_text
            for tag in tags
        }
    )

    if len(all_tags) < 2:
        raise ValueError(
            "At least two distinct tags are needed to create "
            "negative pairs."
        )

    (
        split_train_texts,
        split_train_tags,
        validation_texts,
        validation_tags,
    ) = split_original_examples(
        texts=texts,
        tags_per_text=tags_per_text,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )

    training_dataset = build_pair_dataset(
        texts=split_train_texts,
        tags_per_text=split_train_tags,
        all_tags=all_tags,
        negatives_per_positive=NEGATIVES_PER_POSITIVE,
        seed=RANDOM_SEED,
        include_all_negatives=False,
    )

    # Using all negatives for validation gives more representative metrics.
    validation_dataset = build_pair_dataset(
        texts=validation_texts,
        tags_per_text=validation_tags,
        all_tags=all_tags,
        negatives_per_positive=NEGATIVES_PER_POSITIVE,
        seed=RANDOM_SEED + 1,
        include_all_negatives=True,
    )

    print(f"Number of tags: {len(all_tags)}")
    print(f"Training products: {len(split_train_texts)}")
    print(f"Validation products: {len(validation_texts)}")
    print(f"Training pairs: {len(training_dataset)}")
    print(f"Validation pairs: {len(validation_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens(
                {"pad_token": "<|pad|>"}
            )
        else:
            tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: dict) -> dict:
        # For decoder-only models, representing the two inputs explicitly
        # in one string is usually clearer than relying on token_type_ids.
        combined_inputs = [
            (
                "Product text:\n"
                f"{product_text}\n\n"
                "Candidate product tag:\n"
                f"{tag_text}\n\n"
                "Does this tag apply to the product?"
            )
            for product_text, tag_text in zip(
                batch["product_text"],
                batch["tag_text"],
            )
        ]

        tokenized = tokenizer(
            combined_inputs,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        # Keep these as ordinary floats. The collator converts them to
        # torch.float32 with shape [batch_size, 1].
        tokenized["labels"] = [
            float(label)
            for label in batch["labels"]
        ]

        return tokenized

    tokenized_training_dataset = training_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=training_dataset.column_names,
        desc="Tokenizing training pairs",
    )

    tokenized_validation_dataset = validation_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=validation_dataset.column_names,
        desc="Tokenizing validation pairs",
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = (
        use_cuda
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = use_cuda and not use_bf16

    if use_bf16:
        model_dtype = torch.bfloat16
    elif use_fp16:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    print(f"CUDA available: {use_cuda}")
    print(f"Model dtype: {model_dtype}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="multi_label_classification",
        torch_dtype=model_dtype,
    )

    # Required when a padding token was assigned.
    model.config.pad_token_id = tokenizer.pad_token_id

    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    # Required when gradient checkpointing is enabled.
    model.config.use_cache = False

    data_collator = BinaryPairDataCollator(
        tokenizer=tokenizer
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,

        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,

        dataloader_pin_memory=use_cuda,
        report_to="none",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,

        # Labels are already retained by the tokenization function.
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_training_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    # Inspect one batch before training. This confirms the error is fixed.
    first_batch = next(
        iter(trainer.get_train_dataloader())
    )

    print("\nBatch dtypes before training:")
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(
                f"  {key}: dtype={value.dtype}, "
                f"shape={tuple(value.shape)}"
            )

    assert first_batch["input_ids"].dtype == torch.long

    if "attention_mask" in first_batch:
        assert (
            first_batch["attention_mask"].dtype
            == torch.long
        )

    assert first_batch["labels"].dtype == torch.float32

    train_result = trainer.train()

    print("\nTraining metrics:")
    for key, value in train_result.metrics.items():
        print(f"  {key}: {value}")

    evaluation_metrics = trainer.evaluate()

    print("\nValidation metrics:")
    for key, value in evaluation_metrics.items():
        print(f"  {key}: {value}")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    tags_file = OUTPUT_DIR / "tags.txt"
    tags_file.write_text(
        "\n".join(all_tags),
        encoding="utf-8",
    )

    print(f"\nModel saved to: {OUTPUT_DIR.resolve()}")
    print(f"Tag vocabulary saved to: {tags_file.resolve()}")

    return trainer.model, tokenizer, all_tags


# ============================================================
# Inference
# ============================================================

@torch.inference_mode()
def score_candidate_tags(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    product_text: str,
    candidate_tags: Sequence[str],
    batch_size: int = 16,
) -> list[tuple[str, float]]:
    if not product_text.strip():
        raise ValueError("product_text cannot be empty.")

    if not candidate_tags:
        return []

    model.eval()
    device = next(model.parameters()).device

    scored_tags: list[tuple[str, float]] = []

    for start_index in range(
        0,
        len(candidate_tags),
        batch_size,
    ):
        tag_batch = list(
            candidate_tags[
                start_index:start_index + batch_size
            ]
        )

        combined_inputs = [
            (
                "Product text:\n"
                f"{product_text.strip()}\n\n"
                "Candidate product tag:\n"
                f"{format_tag(tag)}\n\n"
                "Does this tag apply to the product?"
            )
            for tag in tag_batch
        ]

        encoded = tokenizer(
            combined_inputs,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Correct:
        # Move tensors to the model device without applying model.dtype.
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        # Defensive checks. Token indices must remain integers.
        encoded["input_ids"] = encoded[
            "input_ids"
        ].long()

        if "attention_mask" in encoded:
            encoded["attention_mask"] = encoded[
                "attention_mask"
            ].long()

        if "position_ids" in encoded:
            encoded["position_ids"] = encoded[
                "position_ids"
            ].long()

        outputs = model(**encoded)

        logits = outputs.logits.float().reshape(-1)
        probabilities = torch.sigmoid(logits)

        for tag, probability in zip(
            tag_batch,
            probabilities.cpu().tolist(),
        ):
            scored_tags.append(
                (tag, float(probability))
            )

    return sorted(
        scored_tags,
        key=lambda item: item[1],
        reverse=True,
    )


def predict_tags(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    product_text: str,
    candidate_tags: Sequence[str],
    threshold: float = 0.5,
    top_k: int | None = None,
    batch_size: int = 16,
) -> list[tuple[str, float]]:
    scores = score_candidate_tags(
        model=model,
        tokenizer=tokenizer,
        product_text=product_text,
        candidate_tags=candidate_tags,
        batch_size=batch_size,
    )

    selected = [
        (tag, score)
        for tag, score in scores
        if score >= threshold
    ]

    if top_k is not None:
        selected = selected[:top_k]

    return selected


# ============================================================
# Load a saved model
# ============================================================

def load_saved_model(
    model_directory: str | Path,
) -> tuple[
    AutoModelForSequenceClassification,
    AutoTokenizer,
    list[str],
]:
    model_directory = Path(model_directory)

    tokenizer = AutoTokenizer.from_pretrained(
        model_directory
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = (
        use_cuda
        and torch.cuda.is_bf16_supported()
    )

    if use_bf16:
        model_dtype = torch.bfloat16
    elif use_cuda:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        model_directory,
        torch_dtype=model_dtype,
    )

    device = torch.device(
        "cuda" if use_cuda else "cpu"
    )
    model.to(device)
    model.eval()

    tags_file = model_directory / "tags.txt"

    if not tags_file.exists():
        raise FileNotFoundError(
            f"Missing tag vocabulary: {tags_file}"
        )

    all_tags = [
        line.strip()
        for line in tags_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    return model, tokenizer, all_tags


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed(RANDOM_SEED)

    model, tokenizer, all_tags = train_model(
        texts=train_texts,
        tags_per_text=train_tags,
    )

    test_product = (
        "Waterproof insulated coat for hiking in wet weather"
    )

    print(f"\nProduct: {test_product}")

    all_scores = score_candidate_tags(
        model=model,
        tokenizer=tokenizer,
        product_text=test_product,
        candidate_tags=all_tags,
        batch_size=8,
    )

    print("\nAll candidate scores:")
    for tag, score in all_scores:
        print(f"  {tag:25s} {score:.4f}")

    predictions = predict_tags(
        model=model,
        tokenizer=tokenizer,
        product_text=test_product,
        candidate_tags=all_tags,
        threshold=0.50,
        top_k=None,
        batch_size=8,
    )

    print("\nPredicted tags:")
    for tag, score in predictions:
        print(f"  {tag:25s} {score:.4f}")


if __name__ == "__main__":
    main()
