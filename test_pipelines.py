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
from _common import encode_train_test, encode_with, multilabel_metrics


def test_multilabel_encoding():
    train = pd.DataFrame({"text": ["a", "b"], "labels": ["sport,news", "news"]})
    test = pd.DataFrame({"text": ["c"], "labels": [["sport"]]})   # list form too
    tr_texts, Ytr, te_texts, Yte, mlb = encode_train_test(train, test)
    assert tr_texts == ["a", "b"] and te_texts == ["c"]
    assert list(mlb.classes_) == ["news", "sport"]          # MLB sorts
    assert Ytr.tolist() == [[1, 1], [1, 0]] and Yte.tolist() == [[0, 1]]
    # eval encoder uses a fixed class set/order; unknown labels are dropped
    _, Y = encode_with(pd.DataFrame({"text": ["x"], "labels": ["sport,weather"]}),
                       ["news", "sport"])
    assert Y.tolist() == [[0, 1]]


def test_train_loss_and_metrics():
    rng = np.random.default_rng(0)
    Y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    X = np.array([[2.0, 0], [3, 0.1], [0, 2], [0.1, 3]]) + rng.normal(0, 0.01, (4, 2))
    clf = train_logreg.train(X, Y, C=1.0, max_iter=1000)      # @mlflow.trace-decorated
    assert clf.predict(X).shape == Y.shape
    loss = train_logreg.mean_log_loss(clf, X, Y)
    assert np.isfinite(loss) and loss >= 0
    assert multilabel_metrics(Y, Y) == {"f1_micro": 1.0, "f1_macro": 1.0, "subset_accuracy": 1.0}


def test_setfit_loss_scan():
    hist = [{"embedding_loss": 0.5}, {"eval_embedding_loss": 0.4}, {"embedding_loss": 0.2}]
    assert finetune_embedding._losses(hist) == (0.2, 0.4)     # last train, last eval
    assert finetune_embedding._losses([]) == (None, None)


def test_traces_logged():
    with tempfile.TemporaryDirectory() as d:
        mlflow.set_tracking_uri((Path(d) / "mlruns").as_uri())
        mlflow.set_experiment("t")
        train_logreg.train(np.array([[1.0], [0.0]]), np.array([[1, 0], [0, 1]]), C=1.0, max_iter=200)
        mlflow.flush_trace_async_logging()        # traces export asynchronously
        assert len(mlflow.search_traces()) >= 1   # the traced step was recorded


def test_clis_wire_up():
    r = CliRunner()
    for app in (finetune_embedding.app, train_logreg.app, evaluate.app):
        res = r.invoke(app, ["--help"])
        assert res.exit_code == 0, res.output


def main():
    test_multilabel_encoding()
    test_train_loss_and_metrics()
    test_setfit_loss_scan()
    test_traces_logged()
    test_clis_wire_up()
    print("OK")


if __name__ == "__main__":
    main()
