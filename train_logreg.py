"""Train a multi-label logistic-regression classifier on embeddings, then submit it.

Loads a train and a test dataset, multi-hot-encodes labels with sklearn's
MultiLabelBinarizer, embeds the text with a submitted embedding model, fits
one-vs-rest logistic regression, and submits it with its hyperparameters plus
train/validation loss. Records the embedding model+version used as lineage.

    python pipelines/train_logreg.py   # prompts for the rest
"""
import mlflow
import numpy as np
import typer

from mlflow_wrapper import Store
from _common import embed, encode_train_test

app = typer.Typer(add_completion=False)


@mlflow.trace
def train(embeddings, Y, C, max_iter):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    clf = OneVsRestClassifier(LogisticRegression(max_iter=max_iter, C=C))
    clf.fit(embeddings, Y)
    return clf


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
    c: float = typer.Option(1.0, prompt="Inverse regularization (C)", help="LogReg C."),
    max_iter: int = typer.Option(1000, help="LogReg max_iter."),
    tracking_uri: str = typer.Option(None, help="MLflow tracking URI; else env/default."),
):
    store = Store(use_case, tracking_uri=tracking_uri)
    store.set_experiment()  # traces below land in the use_case experiment
    tr_v = store.list_dataset_versions(train_dataset)[-1]   # versions actually used
    te_v = store.list_dataset_versions(test_dataset)[-1]
    tr_texts, Ytr, te_texts, Yte, mlb = encode_train_test(
        store.get_dataset(train_dataset, tr_v), store.get_dataset(test_dataset, te_v))
    version_used = embedding_version or store.latest_model_version(embedding_model)
    embedder = store.get_model(embedding_model, version_used)
    typer.echo(f"embedding {len(tr_texts)} train / {len(te_texts)} test texts "
               f"with {embedding_model} v{version_used}...")

    Xtr, Xte = embed(embedder, tr_texts), embed(embedder, te_texts)
    clf = train(Xtr, Ytr, c, max_iter)
    metrics = {"train_loss": mean_log_loss(clf, Xtr, Ytr),
               "val_loss": mean_log_loss(clf, Xte, Yte)}
    hyper = {"classes": ",".join(mlb.classes_), "C": c, "max_iter": max_iter,
             "embedding_model": embedding_model, "embedding_version": version_used,
             "train_dataset": f"{train_dataset}:v{tr_v}",
             "test_dataset": f"{test_dataset}:v{te_v}",
             "train_size": len(tr_texts), "test_size": len(te_texts)}
    version = store.submit_model(clf, model_name, base_model=embedding_model,
                                 base_version=version_used, params=hyper, metrics=metrics)
    typer.echo(f"submitted {use_case}/{model_name} v{version}  "
               f"train_loss={metrics['train_loss']:.4f} val_loss={metrics['val_loss']:.4f}")


if __name__ == "__main__":
    app()
