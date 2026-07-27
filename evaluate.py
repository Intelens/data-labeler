import random
from collections.abc import Sequence

import numpy as np
from datasets import Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    losses,
)

# One list of texts
train_texts = [
    "Lightweight waterproof hiking jacket",
    "Wireless gaming mouse with RGB lighting",
    "Organic dark roast coffee beans",
]

# One list of tags for each text
train_tags = [
    ["clothing", "outdoor", "waterproof"],
    ["electronics", "gaming", "computer-accessories"],
    ["food", "coffee", "organic"],
]


def make_pair_dataset(
    texts: Sequence[str],
    tags_per_text: Sequence[Sequence[str]],
    negatives_per_positive: int = 3,
    seed: int = 42,
) -> tuple[Dataset, list[str]]:
    if len(texts) != len(tags_per_text):
        raise ValueError("texts and tags_per_text must have equal length")

    rng = random.Random(seed)
    all_tags = sorted({
        tag
        for tags in tags_per_text
        for tag in tags
    })

    rows = {
        "text": [],
        "tag": [],
        "label": [],
    }

    for text, positive_tags in zip(texts, tags_per_text):
        positives = set(positive_tags)
        negative_pool = [tag for tag in all_tags if tag not in positives]

        # Add every positive pair
        for tag in positives:
            rows["text"].append(text)
            rows["tag"].append(tag)
            rows["label"].append(1.0)

        # Sample negative pairs
        number_of_negatives = min(
            len(negative_pool),
            negatives_per_positive * len(positives),
        )

        for tag in rng.sample(negative_pool, number_of_negatives):
            rows["text"].append(text)
            rows["tag"].append(tag)
            rows["label"].append(0.0)

    return Dataset.from_dict(rows), all_tags


train_dataset, all_tags = make_pair_dataset(
    train_texts,
    train_tags,
    negatives_per_positive=3,
)

model = CrossEncoder(
    "distilbert/distilroberta-base",
    num_labels=1,
    max_length=256,
)

training_args = CrossEncoderTrainingArguments(
    output_dir="product-tag-cross-encoder",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    warmup_ratio=0.1,
    logging_steps=20,
    save_strategy="epoch",
    fp16=False,  # Set True on a compatible CUDA GPU
)

trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    loss=losses.BinaryCrossEntropyLoss(model),
)

trainer.train()
model.save_pretrained("product-tag-cross-encoder")
