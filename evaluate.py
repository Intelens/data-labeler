import torch
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.trainer import CrossEncoderTrainer
from sentence_transformers.training_args import CrossEncoderTrainingArguments

# ==========================================
# 1. SETUP RAW DATASET (Sentences + List Labels)
# ==========================================
# Example: Text items with multiple true tags from a predefined universe
ALL_LABELS = ["sports", "finance", "politics", "tech"]

raw_data = {
    "text": [
        "The central bank raised interest rates to combat inflation.",
        "The team won the championship match in overtime yesterday.",
        "The tech company released a new political discussion forum app."
    ],
    "labels": [
        ["finance"],              # Single label
        ["sports"],               # Single label
        ["tech", "politics"]      # Multi-label case
    ]
}

# ==========================================
# 2. TRANSFORM DATA FOR CROSS-ENCODER (NLI Framing)
# ==========================================
# Cross-Encoders evaluate pairs. We explode each sentence against ALL possible labels.
# Target is 1.0 if the label is in the true list, and 0.0 if it is not.
processed_text1 = []
processed_text2 = []
processed_labels = []

for text, true_labels in zip(raw_data["text"], raw_data["labels"]):
    for candidate_label in ALL_LABELS:
        processed_text1.append(text)
        # We format the candidate label as a natural hypothesis statement
        processed_text2.append(f"This text is about {candidate_label}.")
        # Binary target: 1.0 = True match, 0.0 = False match
        score = 1.0 if candidate_label in true_labels else 0.0
        processed_labels.append(score)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({
    "text1": processed_text1,
    "text2": processed_text2,
    "label": processed_labels
})

# ==========================================
# 3. INITIALIZE & TRAIN THE MODEL
# ==========================================
# Load your 0.6B parameter model. num_labels=1 since we output a single similarity probability.
model_name_or_path = "microsoft/deberta-v3-large"  # Replace with your 0.6B model path
model = CrossEncoder(model_name_or_path, num_labels=1)

# Configure arguments to maximize your 50GB VRAM
training_args = CrossEncoderTrainingArguments(
    output_dir="multilabel_cross_encoder",
    num_train_epochs=3,
    per_device_train_batch_size=128,   # High batch size for 50GB VRAM
    per_device_eval_batch_size=128,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="no",                # Change to 'epoch' or 'steps' if you pass an eval_dataset
    save_strategy="epoch"
)

trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Run training
trainer.train()
model.save_pretrained("final_multilabel_cross_encoder")

# ==========================================
# 4. MULTI-LABEL INFERENCE (PREDICTION)
# ==========================================
# Load the trained model
inference_model = CrossEncoder("final_multilabel_cross_encoder")

new_sentence = "The government is investing heavily in clean tech and green energy bonds."

# Build pairs of the new sentence against all possible target classes
inference_pairs = [(new_sentence, f"This text is about {label}.") for label in ALL_LABELS]

# Predict probabilities (Sigmoid activation maps output between 0 and 1)
probabilities = torch.sigmoid(torch.tensor(inference_model.predict(inference_pairs, batch_size=128))).tolist()

# Print predicted scores for each category
print(f"\nResults for sentence: '{new_sentence}'")
for label, prob in zip(ALL_LABELS, probabilities):
    print(f" - {label}: {prob:.4f}")

# Extract active labels using a classification threshold (e.g., > 0.5)
threshold = 0.5
predicted_labels = [label for label, prob in zip(ALL_LABELS, probabilities) if prob > threshold]
print(f"\nFinal Predicted Multi-Labels: {predicted_labels}")
