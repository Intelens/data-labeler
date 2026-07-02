"""Train a multi-label logistic-regression classifier on embeddings, then submit it.

Loads a train and a test dataset, multi-hot-encodes labels with sklearn's
MultiLabelBinarizer, embeds the text with a submitted embedding model, fits one-vs-rest
logistic regression, then fully evaluates it: train/val log-loss + train/test
classification scores as metrics, the output schema (embeddings in, multi-hot out),
train/test prediction CSVs and the console output as artifacts. Records the embedding
model+version used as lineage.

    python pipelines/train_logreg.py   # prompts for the rest
"""
import pathlib
import tempfile

import mlflow
import numpy as np
import typer
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

from mlflow_wrapper import Store
from _common import (capture_console, embed, encode_train_test, multilabel_metrics,
                     predictions_frame)

app = typer.Typer(add_completion=False)


@mlflow.trace
def train(embeddings, Y, C, max_iter):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    clf = OneVsRestClassifier(LogisticRegression(max_iter=max_iter, C=C))
    clf.fit(embeddings, Y)
    return clf


@mlflow.trace
def grid_search_train(embeddings, Y, c_grid, max_iter, cv):
    """Cross-validated grid search over C (scoring f1_micro).
    Returns (best_estimator, best_C, best_cv_score)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.multiclass import OneVsRestClassifier

    gs = GridSearchCV(OneVsRestClassifier(LogisticRegression(max_iter=max_iter)),
                      {"estimator__C": list(c_grid)}, scoring="f1_micro", cv=cv)
    gs.fit(embeddings, Y)
    return gs.best_estimator_, gs.best_params_["estimator__C"], float(gs.best_score_)


def mean_log_loss(clf, X, Y):
    """Mean per-label binary log loss (a.k.a. binary cross-entropy)."""
    from sklearn.metrics import log_loss

    proba = clf.predict_proba(X)
    return float(np.mean([log_loss(Y[:, j], proba[:, j], labels=[0, 1])
                          for j in range(Y.shape[1])]))


@app.command()
def main(
    use_case: str = typer.Option(..., prompt=True, help="Use case / experiment (slug)."),
    train_dataset: str = typer.Option(..., prompt=True, help="Train dataset name."),
    test_dataset: str = typer.Option(..., prompt=True, help="Test/validation dataset name."),
    embedding_model: str = typer.Option("embedder", prompt=True, help="Embedding model name."),
    embedding_version: int = typer.Option(None, help="Embedding version; else latest."),
    model_name: str = typer.Option("classifier", prompt=True, help="Name to submit under."),
    target_column: str = typer.Option("labels", prompt=True, help="Label column in the datasets."),
    c: float = typer.Option(1.0, prompt="Inverse regularization (C)", help="LogReg C (ignored if --grid-search)."),
    max_iter: int = typer.Option(1000, help="LogReg max_iter."),
    grid_search: bool = typer.Option(False, prompt=True, help="Grid-search C instead of using --c."),
    c_grid: str = typer.Option("0.01,0.1,1,10", help="Comma-separated C values for grid search."),
    cv: int = typer.Option(3, help="Cross-validation folds for grid search."),
    tracking_uri: str = typer.Option(None, help="MLflow tracking URI; else env/default."),
):
    store = Store(use_case, tracking_uri=tracking_uri)
    store.set_experiment()  # traces below land in the use_case experiment

    with tempfile.TemporaryDirectory() as d:
        console = pathlib.Path(d) / "console.log"
        train_csv = pathlib.Path(d) / "train_predictions.csv"
        test_csv = pathlib.Path(d) / "test_predictions.csv"
        with capture_console(console):     # capture the whole run's console output
            tr_v = store.list_dataset_versions(train_dataset)[-1]   # versions actually used
            te_v = store.list_dataset_versions(test_dataset)[-1]
            tr_texts, Ytr, te_texts, Yte, mlb = encode_train_test(
                store.get_dataset(train_dataset, tr_v), store.get_dataset(test_dataset, te_v),
                labels_col=target_column)
            version_used = embedding_version or store.latest_model_version(embedding_model)
            embedder = store.get_model(embedding_model, version_used)
            typer.echo(f"embedding {len(tr_texts)} train / {len(te_texts)} test texts "
                       f"with {embedding_model} v{version_used}...")

            Xtr, Xte = embed(embedder, tr_texts), embed(embedder, te_texts)
            cv_score = None
            if grid_search:
                grid = [float(x) for x in c_grid.split(",") if x.strip()]
                clf, c, cv_score = grid_search_train(Xtr, Ytr, grid, max_iter, cv)
                typer.echo(f"grid search C={grid}: best C={c}, cv f1_micro={cv_score:.4f}")
            else:
                clf = train(Xtr, Ytr, c, max_iter)

            # full evaluation on both splits: loss + classification scores + predictions
            pred_tr, pred_te = np.asarray(clf.predict(Xtr)), np.asarray(clf.predict(Xte))
            scores = {**{f"train_{k}": v for k, v in multilabel_metrics(Ytr, pred_tr).items()},
                      **{f"test_{k}": v for k, v in multilabel_metrics(Yte, pred_te).items()}}
            metrics = {"train_loss": mean_log_loss(clf, Xtr, Ytr),
                       "val_loss": mean_log_loss(clf, Xte, Yte), **scores}
            if cv_score is not None:
                metrics["cv_f1_micro"] = cv_score
            predictions_frame(tr_texts, Ytr, pred_tr, mlb).to_csv(train_csv, index=False)
            predictions_frame(te_texts, Yte, pred_te, mlb).to_csv(test_csv, index=False)
            typer.echo(f"done: scores={scores}")

        dim = int(Xtr.shape[1])
        # schema: embedding vectors in, multi-hot labels out
        signature = ModelSignature(
            inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, dim))]),
            outputs=Schema([TensorSpec(np.dtype(np.int64), (-1, len(mlb.classes_)))]))
        hyper = {"classes": ",".join(mlb.classes_), "C": c, "max_iter": max_iter,
                 "grid_search": grid_search,
                 "embedding_model": embedding_model, "embedding_version": version_used,
                 "embedding_dim": dim,
                 "train_dataset": f"{train_dataset}:v{tr_v}",
                 "test_dataset": f"{test_dataset}:v{te_v}",
                 "train_size": len(tr_texts), "test_size": len(te_texts)}
        if grid_search:
            hyper.update({"c_grid": c_grid, "cv": cv})
        version = store.submit_model(
            clf, model_name, base_model=embedding_model, base_version=version_used,
            params=hyper, metrics=metrics, signature=signature,
            run_artifacts={"logs": [str(console)], "predictions": [str(train_csv), str(test_csv)]},
            code_files=[__file__, str(pathlib.Path(__file__).with_name("_common.py"))],
            log_requirements=True)
    typer.echo(f"submitted {use_case}/{model_name} v{version}  "
               f"train_loss={metrics['train_loss']:.4f} val_loss={metrics['val_loss']:.4f} scores={scores}")


if __name__ == "__main__":
    app()
