"""Shared bits for the multi-label text-classification pipelines.

Dataset convention: a `text` column plus one 0/1 column per label. The label
columns (everything except `text`) are the classes, in column order."""
import mlflow
import numpy as np

TEXT = "text"


def split_labeled(df):
    """(texts, Y, classes) from a dataset DataFrame. Y is an int 0/1 matrix."""
    if TEXT not in df.columns:
        raise ValueError(f"dataset needs a '{TEXT}' column")
    classes = [c for c in df.columns if c != TEXT]
    if not classes:
        raise ValueError("dataset needs at least one 0/1 label column besides 'text'")
    return df[TEXT].astype(str).tolist(), df[classes].astype(int).to_numpy(), classes


def label_pairs(texts, Y):
    """(anchor, positive) pairs of texts sharing >=1 label — training signal for
    contrastive fine-tuning.
    # ponytail: O(n^2) scan; batch/ANN it if the labeled set gets large."""
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if np.any(Y[i] & Y[j]):
                pairs.append((texts[i], texts[j]))
    return pairs


@mlflow.trace
def embed(model, texts):
    """Embed texts with a loaded sentence-transformers pyfunc model -> 2D array."""
    return np.asarray(model.predict(list(texts)))


def multilabel_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
    }
