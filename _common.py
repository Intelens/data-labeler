"""Shared bits for the multi-label text-classification pipelines.

Dataset convention: a `text` column and a `labels` column. `labels` is either a
list of label names or a comma/semicolon-separated string. Labels are multi-hot
encoded with sklearn's MultiLabelBinarizer (fit on train, applied to test)."""
import mlflow
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

TEXT, LABELS = "text", "labels"


def _rows(df):
    if TEXT not in df.columns or LABELS not in df.columns:
        raise ValueError(f"dataset needs '{TEXT}' and '{LABELS}' columns")
    texts = df[TEXT].astype(str).tolist()
    labels = []
    for v in df[LABELS]:
        items = v if isinstance(v, (list, tuple, set, np.ndarray)) else str(v).replace(";", ",").split(",")
        labels.append([s.strip() for s in items if str(s).strip()])
    return texts, labels


def encode_train_test(train_df, test_df):
    """Fit a MultiLabelBinarizer on train labels, transform both.
    Returns (train_texts, Ytr, test_texts, Yte, mlb)."""
    tr_texts, tr_labels = _rows(train_df)
    te_texts, te_labels = _rows(test_df)
    mlb = MultiLabelBinarizer()
    return tr_texts, mlb.fit_transform(tr_labels), te_texts, mlb.transform(te_labels), mlb


def encode_with(df, classes):
    """Encode df labels against a fixed class set (for evaluation). Unknown labels
    are ignored. Returns (texts, Y) with columns in `classes` order."""
    texts, labels = _rows(df)
    mlb = MultiLabelBinarizer(classes=list(classes))
    return texts, mlb.fit_transform(labels)


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
