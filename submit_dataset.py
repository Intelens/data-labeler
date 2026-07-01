"""Insert a dataset from a file into the store. Prompts for anything not passed.

    python scripts/submit_dataset.py
    python scripts/submit_dataset.py data.csv --name sales --use-case forecasting
"""
from pathlib import Path

import pandas as pd
import typer

from mlflow_wrapper import Store

app = typer.Typer(add_completion=False)

_READERS = {".csv": pd.read_csv, ".parquet": pd.read_parquet}


@app.command()
def main(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True,
                                help="Dataset file (.csv or .parquet)."),
    name: str = typer.Option(..., prompt=True, help="Dataset name (slug)."),
    use_case: str = typer.Option(..., prompt=True, help="Use case / experiment (slug)."),
    tracking_uri: str = typer.Option(None, help="MLflow tracking URI; else env/default."),
):
    reader = _READERS.get(file.suffix.lower())
    if reader is None:
        raise typer.BadParameter(f"unsupported extension {file.suffix!r}; use {list(_READERS)}")
    df = reader(file)
    version = Store(use_case, tracking_uri=tracking_uri).submit_dataset(df, name)
    typer.echo(f"submitted {use_case}/{name} v{version} ({len(df)} rows) from {file}")


if __name__ == "__main__":
    app()
