# qwen3_product_tag_cross_encoder.py

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
OUTPUT_DIR = Path("./qwen3-product-tag-cross-encoder")

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20
NEGATIVES_PER_POSITIVE = 3

MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

PREDICTION_THRESHOLD = 0.50


# ============================================================
# Replace these with your real data
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


# Optional descriptions make short or ambiguous tags clearer.
TAG_DESCRIPTIONS = {
    "clothing": "Clothing and wearable apparel",
    "outdoor": "Products intended for outdoor activities",
    "waterproof": "Products designed to resist water penetration",
    "electronics": "Electronic devices and accessories",
    "gaming": "Products intended for video gaming",
    "computer accessories": "Accessories and peripherals for computers",
    "food": "Food and edible grocery products",
    "coffee": "Coffee beans, ground coffee, or coffee products",
    "organic": "Products produced according to organic standards",
}


# ============================================================
# Reproducibility
# ============================================================

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data preparation
# ============================================================

def validate_data(
    texts: Sequence[str],
    tags_per_text: Sequence[Sequence[str]],
) -> None:
    if len(texts) != len(tags_per_text):
        raise ValueError(
            "train_texts and train_tags must have equal lengths."
        )

    if len(texts) < 2:
        raise ValueError("At least two training examples are required.")

    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Text at index {index} must be a non-empty string."
            )

    for index, tags in enumerate(tags_per_text):
        if not tags:
            raise ValueError(
                f"Example at index {index} must have at least one tag."
            )


def normalize_data(
    texts: Sequence[str],
    tags_per_text: Sequence[Sequence[str]],
) -> tuple[list[str], list[list[str]]]:
    normalized_texts = []
    normalized_tags = []

    for text, tags in zip(texts, tags_per_text):
        normalized_texts.append(text.strip())

        unique_tags = list(
            dict.fromkeys(
                tag.strip()
                for tag in tags
                if isinstance(tag, str) and tag.strip()
            )
        )

        normalized_tags.append(unique_tags)

    return normalized_texts, normalized_tags


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
    Split by original text before generating text-tag pairs.

    This prevents pairs from the same product from appearing in both
    training and validation data.
    """
    indices = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    validation_size = max(
        1,
        round(len(indices) * validation_fraction),
    )
    validation_size = min(validation_size, len(indices) - 1)

    validation_indices = set(indices[:validation_size])
    training_indices = indices[validation_size:]

    split_train_texts = [
        texts[index]
        for index in training_indices
    ]
    split_train_tags = [
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
        split_train_texts,
        split_train_tags,
        validation_texts,
        validation_tags,
    )


def format_tag(tag: str) -> str:
    description = TAG_DESCRIPTIONS.get(tag)

    if description:
        return f"{tag}: {description}"

    return tag


def format_pair(product_text: str, tag: str) -> str:
    return (
        "Product text:\n"
        f"{product_text}\n\n"
        "Candidate tag:\n"
        f"{format_tag(tag)}\n\n"
        "Determine whether the candidate tag applies to the product."
    )


def create_pairs(
    texts: Sequence[str],
    tags_per_text: Sequence[Sequence[str]],
    all_tags: Sequence[str],
    negatives_per_positive: int,
    seed: int,
    include_all_negatives: bool = False,
) -> list[dict]:
    rng = random.Random(seed)
    pairs = []

    for product_text, positive_tag_list in zip(
        texts,
        tags_per_text,
    ):
        positive_tags = set(positive_tag_list)

        for tag in sorted(positive_tags):
            pairs.append(
                {
                    "product_text": product_text,
                    "tag": tag,
                    "label": 1.0,
                }
            )

        negative_candidates = [
            tag
            for tag in all_tags
            if tag not in positive_tags
        ]

        if include_all_negatives:
            selected_negatives = negative_candidates
        else:
            number_of_negatives = min(
                len(negative_candidates),
                negatives_per_positive * len(positive_tags),
            )

            selected_negatives = rng.sample(
                negative_candidates,
                number_of_negatives,
            )

        for tag in selected_negatives:
            pairs.append(
                {
                    "product_text": product_text,
                    "tag": tag,
                    "label": 0.0,
                }
            )

    rng.shuffle(pairs)
    return pairs


# ============================================================
# Dataset and collator
# ============================================================

class ProductTagPairDataset(Dataset):
    def __init__(self, pairs: Sequence[dict]) -> None:
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        return self.pairs[index]


class ProductTagCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        formatted_inputs = [
            format_pair(
                example["product_text"],
                example["tag"],
            )
            for example in examples
        ]

        encoded = self.tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Enforce integer embedding indices here.
        encoded["input_ids"] = encoded["input_ids"].to(
            dtype=torch.long
        )

        if "attention_mask" in encoded:
            encoded["attention_mask"] = encoded[
                "attention_mask"
            ].to(dtype=torch.long)

        if "position_ids" in encoded:
            encoded["position_ids"] = encoded[
                "position_ids"
            ].to(dtype=torch.long)

        labels = torch.tensor(
            [example["label"] for example in examples],
            dtype=torch.float32,
        )

        encoded["labels"] = labels
        return encoded


# ============================================================
# Safe model wrapper
# ============================================================

class SafeQwenCrossEncoder(nn.Module):
    """
    Forces token indices to torch.long at the final possible point:
    directly inside forward(), before calling Qwen.
    """

    def __init__(self, model_name: str, model_dtype: torch.dtype):
        super().__init__()

        self.base_model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                torch_dtype=model_dtype,
            )
        )

        self.base_model.config.problem_type = "regression"
        self.base_model.config.use_cache = False

    @property
    def config(self):
        return self.base_model.config

    def gradient_checkpointing_enable(self) -> None:
        self.base_model.gradient_checkpointing_enable()

    def save_pretrained(self, output_directory: str | Path) -> None:
        self.base_model.save_pretrained(output_directory)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(self.base_model.parameters()).device

        # Final dtype enforcement immediately before Qwen.
        input_ids = input_ids.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )

        model_inputs = {
            "input_ids": input_ids,
        }

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

        if position_ids is not None:
            model_inputs["position_ids"] = position_ids.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

        if model_inputs["input_ids"].dtype != torch.long:
            raise TypeError(
                "input_ids must be torch.long immediately before "
                f"Qwen forward, but got {model_inputs['input_ids'].dtype}."
            )

        outputs = self.base_model(**model_inputs)

        return outputs.logits.reshape(-1)


# ============================================================
# Batch movement
# ============================================================

def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Do not pass dtype=model.dtype here.

    Integer fields stay integer. Labels stay float32.
    """
    moved_batch = {}

    for key, tensor in batch.items():
        if key in {
            "input_ids",
            "attention_mask",
            "position_ids",
            "token_type_ids",
        }:
            moved_batch[key] = tensor.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
        elif key == "labels":
            moved_batch[key] = tensor.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        else:
            moved_batch[key] = tensor.to(
                device=device,
                non_blocking=True,
            )

    return moved_batch


def print_batch_dtypes(
    batch: dict[str, torch.Tensor],
    heading: str,
) -> None:
    print(f"\n{heading}")

    for key, value in batch.items():
        print(
            f"{key:16s} "
            f"dtype={str(value.dtype):16s} "
            f"device={str(value.device):10s} "
            f"shape={tuple(value.shape)}"
        )


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate_model(
    model: SafeQwenCrossEncoder,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    all_probabilities = []
    all_labels = []

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            position_ids=batch.get("position_ids"),
        )

        labels = batch["labels"].reshape(-1)

        loss = F.binary_cross_entropy_with_logits(
            logits.float(),
            labels,
        )

        probabilities = torch.sigmoid(logits.float())

        total_loss += loss.item()
        all_probabilities.extend(
            probabilities.cpu().tolist()
        )
        all_labels.extend(labels.cpu().tolist())

    probabilities_array = np.asarray(all_probabilities)
    labels_array = np.asarray(all_labels).astype(np.int64)

    predictions_array = (
        probabilities_array >= threshold
    ).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        average="binary",
        zero_division=0,
    )

    accuracy = float(
        np.mean(predictions_array == labels_array)
    )

    return {
        "loss": total_loss / max(len(data_loader), 1),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# Training
# ============================================================

def train_cross_encoder(
    texts: Sequence[str],
    tags_per_text: Sequence[Sequence[str]],
) -> tuple[SafeQwenCrossEncoder, object, list[str]]:
    validate_data(texts, tags_per_text)

    texts, tags_per_text = normalize_data(
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
            "At least two different tags are required."
        )

    (
        split_train_texts,
        split_train_tags,
        validation_texts,
        validation_tags,
    ) = split_original_examples(
        texts,
        tags_per_text,
        VALIDATION_FRACTION,
        RANDOM_SEED,
    )

    training_pairs = create_pairs(
        texts=split_train_texts,
        tags_per_text=split_train_tags,
        all_tags=all_tags,
        negatives_per_positive=NEGATIVES_PER_POSITIVE,
        seed=RANDOM_SEED,
        include_all_negatives=False,
    )

    validation_pairs = create_pairs(
        texts=validation_texts,
        tags_per_text=validation_tags,
        all_tags=all_tags,
        negatives_per_positive=NEGATIVES_PER_POSITIVE,
        seed=RANDOM_SEED + 1,
        include_all_negatives=True,
    )

    print(f"Tags: {len(all_tags)}")
    print(f"Training products: {len(split_train_texts)}")
    print(f"Validation products: {len(validation_texts)}")
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Validation pairs: {len(validation_pairs)}")

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

    collator = ProductTagCollator(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    training_loader = DataLoader(
        ProductTagPairDataset(training_pairs),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    validation_loader = DataLoader(
        ProductTagPairDataset(validation_pairs),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Begin with float32 to remove mixed precision as a possible source
    # of debugging confusion.
    model_dtype = torch.float32

    model = SafeQwenCrossEncoder(
        model_name=MODEL_NAME,
        model_dtype=model_dtype,
    )

    model.base_model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    if (
        len(tokenizer)
        != model.base_model.get_input_embeddings().num_embeddings
    ):
        model.base_model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_steps_per_epoch = max(
        1,
        (
            len(training_loader)
            + GRADIENT_ACCUMULATION_STEPS
            - 1
        )
        // GRADIENT_ACCUMULATION_STEPS,
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch * NUM_EPOCHS
    )

    warmup_steps = round(
        total_optimizer_steps * WARMUP_RATIO
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    # Verify the first batch before training.
    test_batch = next(iter(training_loader))
    print_batch_dtypes(
        test_batch,
        "Batch dtypes before device movement:",
    )

    test_batch = move_batch_to_device(test_batch, device)
    print_batch_dtypes(
        test_batch,
        "Batch dtypes immediately before model forward:",
    )

    assert test_batch["input_ids"].dtype == torch.long
    assert test_batch["labels"].dtype == torch.float32

    with torch.no_grad():
        test_logits = model(
            input_ids=test_batch["input_ids"],
            attention_mask=test_batch.get("attention_mask"),
            position_ids=test_batch.get("position_ids"),
        )

    print(
        "\nInitial forward pass succeeded. "
        f"Logit shape: {tuple(test_logits.shape)}"
    )

    best_validation_f1 = -1.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(NUM_EPOCHS):
        model.train()

        epoch_loss = 0.0
        optimizer_step_count = 0

        for batch_index, batch in enumerate(training_loader):
            batch = move_batch_to_device(batch, device)

            # Assertions occur at every iteration.
            if batch["input_ids"].dtype != torch.long:
                raise TypeError(
                    "input_ids changed dtype before forward: "
                    f"{batch['input_ids'].dtype}"
                )

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                position_ids=batch.get("position_ids"),
            )

            labels = batch["labels"].reshape(-1)

            loss = F.binary_cross_entropy_with_logits(
                logits.float(),
                labels,
            )

            scaled_loss = (
                loss / GRADIENT_ACCUMULATION_STEPS
            )
            scaled_loss.backward()

            epoch_loss += loss.item()

            is_accumulation_boundary = (
                (batch_index + 1)
                % GRADIENT_ACCUMULATION_STEPS
                == 0
            )

            is_last_batch = (
                batch_index + 1 == len(training_loader)
            )

            if is_accumulation_boundary or is_last_batch:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                optimizer_step_count += 1

        average_train_loss = (
            epoch_loss / max(len(training_loader), 1)
        )

        validation_metrics = evaluate_model(
            model=model,
            data_loader=validation_loader,
            device=device,
            threshold=PREDICTION_THRESHOLD,
        )

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train loss: {average_train_loss:.4f}")
        print(
            "Validation: "
            f"loss={validation_metrics['loss']:.4f}, "
            f"accuracy={validation_metrics['accuracy']:.4f}, "
            f"precision={validation_metrics['precision']:.4f}, "
            f"recall={validation_metrics['recall']:.4f}, "
            f"f1={validation_metrics['f1']:.4f}"
        )

        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]

            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

            with open(
                OUTPUT_DIR / "tags.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    all_tags,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            print("Saved new best model.")

    return model, tokenizer, all_tags


# ============================================================
# Prediction
# ============================================================

@torch.inference_mode()
def score_tags(
    model: SafeQwenCrossEncoder,
    tokenizer,
    product_text: str,
    candidate_tags: Sequence[str],
    batch_size: int = 8,
) -> list[tuple[str, float]]:
    if not product_text.strip():
        raise ValueError("product_text cannot be empty.")

    device = next(model.parameters()).device
    model.eval()

    results = []

    for start in range(0, len(candidate_tags), batch_size):
        current_tags = list(
            candidate_tags[start:start + batch_size]
        )

        formatted_inputs = [
            format_pair(product_text, tag)
            for tag in current_tags
        ]

        encoded = tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Never use encoded.to(device, dtype=model.dtype).
        input_ids = encoded["input_ids"].to(
            device=device,
            dtype=torch.long,
        )

        attention_mask = encoded.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(
                device=device,
                dtype=torch.long,
            )

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.sigmoid(
            logits.float()
        ).cpu().tolist()

        results.extend(
            (tag, float(probability))
            for tag, probability in zip(
                current_tags,
                probabilities,
            )
        )

    return sorted(
        results,
        key=lambda item: item[1],
        reverse=True,
    )


def predict_tags(
    model: SafeQwenCrossEncoder,
    tokenizer,
    product_text: str,
    candidate_tags: Sequence[str],
    threshold: float = 0.5,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    scores = score_tags(
        model=model,
        tokenizer=tokenizer,
        product_text=product_text,
        candidate_tags=candidate_tags,
    )

    predictions = [
        (tag, score)
        for tag, score in scores
        if score >= threshold
    ]

    if top_k is not None:
        predictions = predictions[:top_k]

    return predictions


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_random_seed(RANDOM_SEED)

    model, tokenizer, all_tags = train_cross_encoder(
        texts=train_texts,
        tags_per_text=train_tags,
    )

    product = (
        "Waterproof insulated jacket for hiking "
        "in cold and wet weather"
    )

    scores = score_tags(
        model=model,
        tokenizer=tokenizer,
        product_text=product,
        candidate_tags=all_tags,
    )

    print(f"\nProduct: {product}")
    print("\nAll scores:")

    for tag, score in scores:
        print(f"{tag:24s} {score:.4f}")

    predictions = predict_tags(
        model=model,
        tokenizer=tokenizer,
        product_text=product,
        candidate_tags=all_tags,
        threshold=0.50,
    )

    print("\nSelected tags:")

    for tag, score in predictions:
        print(f"{tag:24s} {score:.4f}")


if __name__ == "__main__":
    main()
