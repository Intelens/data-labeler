"""Train a multi-label logistic-regression classifier on embeddings, then submit it.

Embeds the dataset text with a submitted embedding model and fits one-vs-rest
logistic regression. Records the embedding model+version it used as lineage.

    python pipelines/train_logreg.py   # prompts for the rest
"""
import mlflow
import typer

from mlflow_wrapper import Store
from _common import embed, split_labeled

app = typer.Typer(add_completion=False)


@mlflow.trace
def train(embeddings, Y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=C))
    clf.fit(embeddings, Y)
    return clf


@app.command()
def main(
    use_case: str = typer.Option(..., prompt=True, help="Use case / experiment (slug)."),
    dataset: str = typer.Option(..., prompt=True, help="Labeled dataset name."),
    embedding_model: str = typer.Option("embedder", prompt=True, help="Embedding model name."),
    embedding_version: int = typer.Option(None, help="Embedding version; else latest."),
    model_name: str = typer.Option("classifier", prompt=True, help="Name to submit under."),
    c: float = typer.Option(1.0, prompt="Inverse regularization (C)", help="LogReg C."),
    tracking_uri: str = typer.Option(None, help="MLflow tracking URI; else env/default."),
):
    store = Store(use_case, tracking_uri=tracking_uri)
    store.set_experiment()  # traces below land in the use_case experiment
    texts, Y, classes = split_labeled(store.get_dataset(dataset))
    version_used = embedding_version or store.latest_model_version(embedding_model)
    embedder = store.get_model(embedding_model, version_used)
    typer.echo(f"embedding {len(texts)} texts with {embedding_model} v{version_used}...")
    clf = train(embed(embedder, texts), Y, c)
    version = store.submit_model(
        clf, model_name, base_model=embedding_model, base_version=version_used,
        params={"classes": ",".join(classes), "C": c})
    typer.echo(f"submitted {use_case}/{model_name} v{version} ({len(classes)} labels)")


if __name__ == "__main__":
    app()
