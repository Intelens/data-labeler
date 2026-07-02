"""Shared bits for the multi-label text-classification pipelines.

Dataset convention: a `text` column and a `labels` column. `labels` is either a
list of label names or a comma/semicolon-separated string. Labels are multi-hot
encoded with sklearn's MultiLabelBinarizer (fit on train, applied to test)."""
import contextlib
import sys

import mlflow
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

TEXT, LABELS = "text", "labels"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def capture_console(path):
    """Tee stdout+stderr to `path` while still printing live.
    # ponytail: catches direct writes and tqdm; logging handlers bound to the old
    # stream before entry won't be captured — good enough for training output."""
    with open(path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)), \
             contextlib.redirect_stderr(_Tee(sys.stderr, f)):
            yield


def _rows(df, labels_col=LABELS):
    if TEXT not in df.columns or labels_col not in df.columns:
        raise ValueError(f"dataset needs '{TEXT}' and '{labels_col}' columns")
    texts = df[TEXT].astype(str).tolist()
    labels = []
    for v in df[labels_col]:
        items = v if isinstance(v, (list, tuple, set, np.ndarray)) else str(v).replace(";", ",").split(",")
        labels.append([s.strip() for s in items if str(s).strip()])
    return texts, labels


def encode_train_test(train_df, test_df, labels_col=LABELS):
    """Fit a MultiLabelBinarizer on train labels, transform both.
    Returns (train_texts, Ytr, test_texts, Yte, mlb)."""
    tr_texts, tr_labels = _rows(train_df, labels_col)
    te_texts, te_labels = _rows(test_df, labels_col)
    mlb = MultiLabelBinarizer()
    return tr_texts, mlb.fit_transform(tr_labels), te_texts, mlb.transform(te_labels), mlb


def encode_with(df, classes, labels_col=LABELS):
    """Encode df labels against a fixed class set (for evaluation). Unknown labels
    are ignored. Returns (texts, Y) with columns in `classes` order."""
    texts, labels = _rows(df, labels_col)
    mlb = MultiLabelBinarizer(classes=list(classes))
    return texts, mlb.fit_transform(labels)


def predictions_frame(texts, Y_true, Y_pred, mlb):
    """Table of text + true/predicted label sets (semicolon-joined) for logging as a
    UI-viewable CSV artifact."""
    import pandas as pd
    true = mlb.inverse_transform(np.asarray(Y_true))
    pred = mlb.inverse_transform(np.asarray(Y_pred))
    return pd.DataFrame({"text": list(texts),
                         "true": [";".join(t) for t in true],
                         "predicted": [";".join(p) for p in pred]})


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
