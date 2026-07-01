"""Self-check (no network): python pipelines/test_pipelines.py"""
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from typer.testing import CliRunner

import evaluate
import finetune_embedding
import train_logreg
from _common import label_pairs, multilabel_metrics, split_labeled


def test_split_and_pairs():
    df = pd.DataFrame({"text": ["a", "b", "c"], "sport": [1, 0, 1], "news": [0, 1, 1]})
    texts, Y, classes = split_labeled(df)
    assert texts == ["a", "b", "c"] and classes == ["sport", "news"]
    assert Y.tolist() == [[1, 0], [0, 1], [1, 1]]
    # a&c share 'sport', b&c share 'news', a&b share nothing
    pairs = label_pairs(texts, Y)
    assert ("a", "c") in pairs and ("b", "c") in pairs and ("a", "b") not in pairs


def test_train_and_metrics():
    rng = np.random.default_rng(0)
    # two linearly separable label directions so logreg actually learns
    Y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    X = np.array([[2.0, 0], [3, 0.1], [0, 2], [0.1, 3]]) + rng.normal(0, 0.01, (4, 2))
    clf = train_logreg.train(X, Y, C=1.0)         # @mlflow.trace-decorated
    pred = clf.predict(X)
    assert pred.shape == Y.shape
    m = multilabel_metrics(Y, Y)
    assert m == {"f1_micro": 1.0, "f1_macro": 1.0, "subset_accuracy": 1.0}


def test_traces_logged():
    with tempfile.TemporaryDirectory() as d:
        mlflow.set_tracking_uri((Path(d) / "mlruns").as_uri())
        mlflow.set_experiment("t")
        train_logreg.train(np.array([[1.0], [0.0]]), np.array([[1, 0], [0, 1]]), C=1.0)
        mlflow.flush_trace_async_logging()        # traces export asynchronously
        assert len(mlflow.search_traces()) >= 1   # the traced step was recorded


def test_clis_wire_up():
    r = CliRunner()
    for app in (finetune_embedding.app, train_logreg.app, evaluate.app):
        res = r.invoke(app, ["--help"])
        assert res.exit_code == 0, res.output


def main():
    test_split_and_pairs()
    test_train_and_metrics()
    test_traces_logged()
    test_clis_wire_up()
    print("OK")


if __name__ == "__main__":
    main()
