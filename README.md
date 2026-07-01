# mlflow_wrapper

Thin MLflow wrapper. A `Store` is scoped to one `use_case` and gives you versioned
parquet datasets, auto-flavored models, and runs — all under a standardized name.

## Install

```bash
pip install -e .          # deps: mlflow>=3, pandas, pyarrow
```

## Config

`Store(use_case, tracking_uri=None, registry_uri=None)` resolves each URI by
**arg > env var > default**:

| setting       | env var                 | default        |
|---------------|-------------------------|----------------|
| tracking_uri  | `MLFLOW_TRACKING_URI`   | `file:./mlruns`|
| registry_uri  | `MLFLOW_REGISTRY_URI`   | tracking_uri   |

`use_case` and every `name` must match `^[a-z0-9][a-z0-9_-]*$` (no `__`).

## Usage

```python
import pandas as pd
from mlflow_wrapper import Store

store = Store("forecasting")
```

### Datasets — versioned parquet

```python
v1 = store.submit_dataset(pd.DataFrame({"x": [1, 2]}), "sales")   # -> 1
v2 = store.submit_dataset(pd.DataFrame({"x": [1, 2, 3]}), "sales") # -> 2
store.list_dataset_versions("sales")        # [1, 2]
store.get_dataset("sales")                  # latest DataFrame
store.get_dataset("sales", version=1)       # pinned
```

### Models — auto-detected flavor

`submit_model` picks the flavor from the object: **xgboost**, **sklearn**, and
**sentence-transformers** log with their native flavor; anything else (a
`mlflow.pyfunc.PythonModel`) goes through pyfunc. Versioning is MLflow's registry.

```python
store.submit_model(sklearn_estimator, "ranker")               # -> version 1
store.submit_model(xgb_regressor, "ranker")                   # -> version 2
store.get_model("ranker")                                     # latest, pyfunc-loaded
store.get_model("ranker", version=1)

# lineage: record the base model + version this one builds on
store.submit_model(model, "reranker", base_model="ranker", base_version=2)
```

### Runs

```python
rid = store.submit_run("baseline", params={"lr": 0.1}, metrics={"rmse": 2.5})
store.get_run("baseline")     # most recent matching run
store.list_runs()             # all runs in this use_case (DataFrame)
```

## Layout

Everything is stored in MLflow: `use_case` is the experiment, objects are tagged
`type`/`use_case`/`name`, datasets are parquet run-artifacts, models use the registry
as `use_case__name`.

## UI

```bash
mlflow ui --backend-store-uri ./mlruns    # then open http://localhost:5000
```

Point `--backend-store-uri` at whatever your `MLFLOW_TRACKING_URI` is (default
`./mlruns`). Add `--port 5001` if 5000 is taken. Browse experiments (per use_case),
runs, dataset artifacts, and the model registry there.

## Test / example

```bash
python test_mlflow_wrapper.py   # assert-based self-check
```

See [example.ipynb](example.ipynb) for a runnable walkthrough.
