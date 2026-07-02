"""Fine-tune an embedding model on labeled text with SetFit, then submit its body.

Loads a train and a test dataset, multi-hot-encodes labels with sklearn's
MultiLabelBinarizer, and SetFit-fine-tunes the sentence-transformer. Submits the
fine-tuned body (a custom pyfunc EmbeddingModel) as the "embedder", together with all
hyperparameters/training args, train/val loss, and — using SetFit's own classification
head — train/test classification scores. Train and test predictions are uploaded as
CSV artifacts (viewable in the UI). Downstream train_logreg can fit its own head.

    python pipelines/finetune_embedding.py   # prompts for the rest
"""
import pathlib
import tempfile

import mlflow
import numpy as np
import typer
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema, TensorSpec

from mlflow_wrapper import Store
from _common import capture_console, encode_train_test, multilabel_metrics, predictions_frame

app = typer.Typer(add_completion=False)


class EmbeddingModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow model wrapping the fine-tuned SentenceTransformer body. Takes a
    list/Series of texts and returns their embeddings; the body is stored as the
    `body` artifact and reloaded in load_context."""

    def load_context(self, context):
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(context.artifacts["body"])

    def predict(self, context, model_input, params=None):
        if hasattr(model_input, "columns"):        # DataFrame -> first column
            texts = model_input.iloc[:, 0].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        return np.asarray(self._st.encode(texts))


def _losses(log_history):
    """Pull the last train and eval loss out of a SetFit/transformers log history.
    # ponytail: log_history keys drift across versions; scan for known aliases, None if absent."""
    def last(keys):
        for h in reversed(log_history or []):
            for k in keys:
                if k in h:
                    return float(h[k])
        return None
    return last(["embedding_loss", "loss"]), last(["eval_embedding_loss", "eval_loss"])


@mlflow.trace
def finetune(base_model, tr_texts, Ytr, te_texts, Yte, train_args):
    """SetFit-fine-tune on multi-label data with an eval set. train_args is a dict of
    TrainingArguments kwargs. Returns (trained SetFitModel, train_loss, val_loss);
    the model keeps both its fine-tuned body and its classification head."""
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    model = SetFitModel.from_pretrained(base_model, multi_target_strategy="one-vs-rest")
    train_ds = Dataset.from_dict({"text": list(tr_texts), "label": Ytr.tolist()})
    eval_ds = Dataset.from_dict({"text": list(te_texts), "label": Yte.tolist()})
    trainer = Trainer(model=model, train_dataset=train_ds, eval_dataset=eval_ds,
                      args=TrainingArguments(**train_args))
    trainer.train()
    train_loss, val_loss = _losses(getattr(trainer.state, "log_history", None))
    return model, train_loss, val_loss


@app.command()
def main(
    use_case: str = typer.Option(..., prompt=True, help="Use case / experiment (slug)."),
    train_dataset: str = typer.Option(..., prompt=True, help="Train dataset name."),
    test_dataset: str = typer.Option(..., prompt=True, help="Test/validation dataset name."),
    base_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", prompt=True,
                                    help="Base model to fine-tune."),
    model_name: str = typer.Option("embedder", prompt=True, help="Name to submit under."),
    target_column: str = typer.Option("labels", prompt=True, help="Label column in the datasets."),
    epochs: int = typer.Option(1, prompt=True, help="Training epochs."),
    batch_size: int = typer.Option(16, prompt=True, help="Batch size."),
    body_learning_rate: float = typer.Option(2e-5, prompt=True, help="Embedding body learning rate."),
    head_learning_rate: float = typer.Option(1e-2, prompt=True, help="Classification head learning rate."),
    num_iterations: int = typer.Option(20, help="Contrastive pairs generated per example."),
    end_to_end: bool = typer.Option(False, prompt=True,
                                    help="Train the whole model end-to-end (else freeze body in head phase)."),
    tracking_uri: str = typer.Option(None, help="MLflow tracking URI; else env/default."),
):
    store = Store(use_case, tracking_uri=tracking_uri)
    store.set_experiment()  # traces below land in the use_case experiment
    train_args = {"num_epochs": epochs, "batch_size": batch_size,
                  "num_iterations": num_iterations, "end_to_end": end_to_end,
                  "body_learning_rate": body_learning_rate,
                  "head_learning_rate": head_learning_rate}

    with tempfile.TemporaryDirectory() as d:
        console = pathlib.Path(d) / "console.log"
        bodydir = pathlib.Path(d) / "body"
        train_csv = pathlib.Path(d) / "train_predictions.csv"
        test_csv = pathlib.Path(d) / "test_predictions.csv"
        with capture_console(console):     # capture the whole run's console output
            tr_v = store.list_dataset_versions(train_dataset)[-1]   # versions actually used
            te_v = store.list_dataset_versions(test_dataset)[-1]
            tr_texts, Ytr, te_texts, Yte, mlb = encode_train_test(
                store.get_dataset(train_dataset, tr_v), store.get_dataset(test_dataset, te_v),
                labels_col=target_column)
            typer.echo(f"SetFit fine-tuning {base_model} on {len(tr_texts)} train / "
                       f"{len(te_texts)} test texts, {len(mlb.classes_)} labels...")
            model, train_loss, val_loss = finetune(base_model, tr_texts, Ytr, te_texts, Yte, train_args)
            body = model.model_body
            body.save(str(bodydir))
            dim = body.get_sentence_embedding_dimension()

            # score with the trained classification head, on both splits
            # ponytail: assumes SetFit multi-target predict returns a multi-hot matrix in mlb order
            pred_tr = np.asarray(model.predict(list(tr_texts)))
            pred_te = np.asarray(model.predict(list(te_texts)))
            scores = {**{f"train_{k}": v for k, v in multilabel_metrics(Ytr, pred_tr).items()},
                      **{f"test_{k}": v for k, v in multilabel_metrics(Yte, pred_te).items()}}
            predictions_frame(tr_texts, Ytr, pred_tr, mlb).to_csv(train_csv, index=False)
            predictions_frame(te_texts, Yte, pred_te, mlb).to_csv(test_csv, index=False)
            typer.echo(f"done: dim={dim}, losses=({train_loss},{val_loss}), scores={scores}")

        # schema: takes a string column, returns a float32 embedding vector
        signature = ModelSignature(inputs=Schema([ColSpec("string")]),
                                   outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, dim))]))
        hyper = {"trainer": "setfit", "base_model": base_model, "target_column": target_column,
                 "multi_target_strategy": "one-vs-rest", "classes": ",".join(mlb.classes_),
                 "embedding_dim": dim, **train_args,
                 "train_dataset": f"{train_dataset}:v{tr_v}",
                 "test_dataset": f"{test_dataset}:v{te_v}",
                 "train_size": len(tr_texts), "test_size": len(te_texts)}
        metrics = {**scores,
                   **{k: v for k, v in {"train_loss": train_loss, "val_loss": val_loss}.items()
                      if v is not None}}
        # submit with schema, datasets used, console output, and train/test prediction tables
        version = store.submit_model(
            EmbeddingModel(), model_name, artifacts={"body": str(bodydir)},
            params=hyper, metrics=metrics, signature=signature,
            run_artifacts={"logs": [str(console)], "predictions": [str(train_csv), str(test_csv)]},
            code_files=[__file__, str(pathlib.Path(__file__).with_name("_common.py"))],
            log_requirements=True)
    typer.echo(f"submitted {use_case}/{model_name} v{version}  scores={scores}")


if __name__ == "__main__":
    app()
