"""Fine-tune a sentence-transformers embedding model on labeled text, then submit it.

    python pipelines/finetune_embedding.py   # prompts for the rest
"""
import mlflow
import typer

from mlflow_wrapper import Store
from _common import label_pairs, split_labeled

app = typer.Typer(add_completion=False)


@mlflow.trace
def finetune(base_model, texts, Y, epochs, batch_size):
    """Contrastive fine-tune on same-label pairs. Returns the SentenceTransformer.
    # ponytail: MultipleNegativesRankingLoss on shared-label pairs, a few epochs —
    # the standard lazy fine-tune. Swap in a task-specific loss if it underperforms."""
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(base_model)
    pairs = label_pairs(texts, Y)
    if not pairs:
        typer.echo("no same-label pairs to train on; returning base model unchanged")
        return model
    loader = DataLoader([InputExample(texts=[a, b]) for a, b in pairs],
                        shuffle=True, batch_size=max(1, min(batch_size, len(pairs))))
    model.fit(train_objectives=[(loader, losses.MultipleNegativesRankingLoss(model))],
              epochs=epochs, show_progress_bar=False)
    return model


@app.command()
def main(
    use_case: str = typer.Option(..., prompt=True, help="Use case / experiment (slug)."),
    dataset: str = typer.Option(..., prompt=True, help="Labeled dataset name."),
    base_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", prompt=True,
                                    help="Base model to fine-tune."),
    model_name: str = typer.Option("embedder", prompt=True, help="Name to submit under."),
    epochs: int = typer.Option(1, prompt=True, help="Training epochs."),
    batch_size: int = typer.Option(16, help="Batch size."),
    tracking_uri: str = typer.Option(None, help="MLflow tracking URI; else env/default."),
):
    store = Store(use_case, tracking_uri=tracking_uri)
    store.set_experiment()  # traces below land in the use_case experiment
    texts, Y, classes = split_labeled(store.get_dataset(dataset))
    typer.echo(f"fine-tuning {base_model} on {len(texts)} texts, {len(classes)} labels...")
    model = finetune(base_model, texts, Y, epochs, batch_size)
    version = store.submit_model(model, model_name, params={"base_embedding": base_model})
    typer.echo(f"submitted {use_case}/{model_name} v{version}")


if __name__ == "__main__":
    app()
