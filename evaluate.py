"""Evaluate a submitted classifier on a labeled dataset; log metrics as a run.

Reads the classifier's lineage (base embedding model + version, trained classes)
from its run params, so you only name the classifier and the eval dataset.

    python pipelines/evaluate.py   # prompts for the rest
"""
import mlflow
import numpy as np
import typer

from mlflow_wrapper import Store
from _common import embed, encode_with, multilabel_metrics

app = typer.Typer(add_completion=False)


@mlflow.trace
def classify(clf, embeddings):
    return np.asarray(clf.predict(embeddings))


@app.command()
def main(
    use_case: str = typer.Option(..., prompt=True, help="Use case / experiment (slug)."),
    dataset: str = typer.Option(..., prompt=True, help="Labeled eval dataset name."),
    classifier: str = typer.Option("classifier", prompt=True, help="Classifier model name."),
    classifier_version: int = typer.Option(None, help="Classifier version; else latest."),
    target_column: str = typer.Option("labels", prompt=True, help="Label column in the dataset."),
    run_name: str = typer.Option("eval", prompt=True, help="Name for the eval run."),
    tracking_uri: str = typer.Option(None, help="MLflow tracking URI; else env/default."),
):
    store = Store(use_case, tracking_uri=tracking_uri)
    store.set_experiment()  # traces below land in the use_case experiment
    version = classifier_version or store.latest_model_version(classifier)
    lineage = store.model_run_params(classifier, version)
    embed_model, embed_version = lineage["base_model"], int(lineage["base_version"])
    trained = lineage["classes"].split(",")

    # encode eval labels against the classifier's trained classes (same columns/order)
    texts, Y_true = encode_with(store.get_dataset(dataset), trained, labels_col=target_column)

    embedder = store.get_model(embed_model, embed_version)
    clf = store.get_model(classifier, version)
    Y_pred = classify(clf, embed(embedder, texts))
    metrics = multilabel_metrics(Y_true, Y_pred)

    store.submit_run(run_name, params={"classifier": classifier, "classifier_version": version},
                     metrics=metrics)
    typer.echo(f"{classifier} v{version} on {dataset}: " +
               ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    app()
