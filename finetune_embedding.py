"""Fine-tune an embedding model on labeled text with SetFit, then submit its body.

Loads a train and a test dataset, multi-hot-encodes labels with sklearn's
MultiLabelBinarizer, and SetFit-fine-tunes the sentence-transformer (SetFit builds
the contrastive pairs internally). Submits the fine-tuned body (a SentenceTransformer)
as the "embedder", together with the training arguments, hyperparameters, and
train/validation loss. Downstream train_logreg fits the classification head.

    python pipelines/finetune_embedding.py   # prompts for the rest
"""
import tempfile

import mlflow
import numpy as np
import typer

from mlflow_wrapper import Store
from _common import encode_train_test

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
def finetune(base_model, tr_texts, Ytr, te_texts, Yte, epochs, batch_size):
    """SetFit-fine-tune on multi-label data with an eval set.
    Returns (SentenceTransformer body, train_loss, val_loss)."""
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    model = SetFitModel.from_pretrained(base_model, multi_target_strategy="one-vs-rest")
    train_ds = Dataset.from_dict({"text": list(tr_texts), "label": Ytr.tolist()})
    eval_ds = Dataset.from_dict({"text": list(te_texts), "label": Yte.tolist()})
    trainer = Trainer(model=model, train_dataset=train_ds, eval_dataset=eval_ds,
                      args=TrainingArguments(num_epochs=epochs, batch_size=batch_size))
    trainer.train()
    train_loss, val_loss = _losses(getattr(trainer.state, "log_history", None))
    return model.model_body, train_loss, val_loss


@app.command()
def main(
    use_case: str = typer.Option(..., prompt=True, help="Use case / experiment (slug)."),
    train_dataset: str = typer.Option(..., prompt=True, help="Train dataset name."),
    test_dataset: str = typer.Option(..., prompt=True, help="Test/validation dataset name."),
    base_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", prompt=True,
                                    help="Base model to fine-tune."),
    model_name: str = typer.Option("embedder", prompt=True, help="Name to submit under."),
    epochs: int = typer.Option(1, prompt=True, help="Training epochs."),
    batch_size: int = typer.Option(16, help="Batch size."),
    tracking_uri: str = typer.Option(None, help="MLflow tracking URI; else env/default."),
):
    store = Store(use_case, tracking_uri=tracking_uri)
    store.set_experiment()  # traces below land in the use_case experiment
    tr_texts, Ytr, te_texts, Yte, mlb = encode_train_test(
        store.get_dataset(train_dataset), store.get_dataset(test_dataset))
    typer.echo(f"SetFit fine-tuning {base_model} on {len(tr_texts)} train / "
               f"{len(te_texts)} test texts, {len(mlb.classes_)} labels...")

    body, train_loss, val_loss = finetune(base_model, tr_texts, Ytr, te_texts, Yte,
                                           epochs, batch_size)
    hyper = {"trainer": "setfit", "base_model": base_model, "epochs": epochs,
             "batch_size": batch_size, "multi_target_strategy": "one-vs-rest",
             "classes": ",".join(mlb.classes_),
             "train_size": len(tr_texts), "test_size": len(te_texts)}
    metrics = {k: v for k, v in {"train_loss": train_loss, "val_loss": val_loss}.items()
               if v is not None}
    # wrap the fine-tuned body in the custom pyfunc model and submit that
    with tempfile.TemporaryDirectory() as d:
        body.save(d)
        version = store.submit_model(EmbeddingModel(), model_name, artifacts={"body": d},
                                     params=hyper, metrics=metrics)
    typer.echo(f"submitted {use_case}/{model_name} v{version} "
               f"(train_loss={train_loss}, val_loss={val_loss})")


if __name__ == "__main__":
    app()
