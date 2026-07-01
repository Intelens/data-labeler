"""Thin MLflow wrapper. A `Store` is scoped to one `use_case` and gives you
versioned parquet datasets, pyfunc models, and runs under a standardized name.

    store = Store("forecasting")            # tracking uri: arg > env > file:./mlruns
    v = store.submit_dataset(df, "sales")   # -> version int
    df = store.get_dataset("sales")         # latest
"""
import os
import re
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate(label, val):
    if not isinstance(val, str) or not _SLUG.match(val) or "__" in val:
        raise ValueError(f"{label} must match {_SLUG.pattern} and contain no '__': {val!r}")


def _flavor(model):
    """Pick the mlflow flavor module by inspecting the model's class, or None for
    the generic pyfunc path. No imports of the model libs — just their module names."""
    top = type(model).__module__.split(".")[0]
    if top == "xgboost":                       # check before sklearn: XGBRegressor is both
        return mlflow.xgboost
    if top == "sentence_transformers":
        return mlflow.sentence_transformers
    if any(c.__module__.split(".")[0] == "sklearn" for c in type(model).__mro__):
        return mlflow.sklearn
    return None


class Store:
    """MLflow-backed store scoped to a single use_case."""

    def __init__(self, use_case, tracking_uri=None, registry_uri=None):
        _validate("use_case", use_case)
        self.use_case = use_case
        # Precedence: arg > env var > default.
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "file:./mlruns"
        self.registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI") or self.tracking_uri
        # ponytail: MLflow tracking/registry uri is process-global; last Store constructed wins
        # if two point at different backends. Fine for the one-backend case; split processes if not.
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)

    def _full_name(self, name):
        _validate("name", name)
        return f"{self.use_case}__{name}"

    def _find_runs(self, name, obj_type, extra=""):
        """Runs tagged for this object, newest first. Empty if experiment absent."""
        exp = mlflow.get_experiment_by_name(self.use_case)
        if exp is None:
            return []
        filt = (f"tags.type = '{obj_type}' and tags.use_case = '{self.use_case}' "
                f"and tags.name = '{name}'")
        if extra:
            filt += f" and {extra}"
        return mlflow.search_runs([exp.experiment_id], filter_string=filt, output_format="list")

    # --- datasets ---------------------------------------------------------

    def submit_dataset(self, df, name, tags=None):
        """Store df as a versioned parquet artifact. Returns the new version (int)."""
        _validate("name", name)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        # ponytail: version=max+1, single-writer assumption; add a registry/lock if concurrent writers
        version = max(self.list_dataset_versions(name), default=0) + 1
        mlflow.set_experiment(self.use_case)
        run_tags = {"type": "dataset", "use_case": self.use_case, "name": name,
                    "version": str(version), **(tags or {})}
        with mlflow.start_run(run_name=f"{name}-v{version}", tags=run_tags):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "data.parquet"
                df.to_parquet(p, engine="pyarrow", index=False)
                mlflow.log_artifact(str(p), artifact_path="dataset")
            # Register as a first-class MLflow dataset too, so it shows in the UI's
            # Datasets view (the parquet above is what get_dataset actually reads back).
            mlflow.log_input(mlflow.data.from_pandas(df, name=name, digest=str(version)),
                             context=f"v{version}")
        return version

    def list_dataset_versions(self, name):
        """Sorted list of existing dataset versions (ints)."""
        return sorted(int(r.data.tags["version"]) for r in self._find_runs(name, "dataset"))

    def get_dataset(self, name, version=None):
        """Load a dataset as a DataFrame. version=None loads the latest."""
        extra = f"tags.version = '{version}'" if version is not None else ""
        runs = self._find_runs(name, "dataset", extra)
        if not runs:
            raise KeyError(f"no dataset {self.use_case}/{name} version={version}")
        run = max(runs, key=lambda r: int(r.data.tags["version"]))
        local = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="dataset/data.parquet")
        return pd.read_parquet(local, engine="pyarrow")

    # --- models -----------------------------------------------------------

    def submit_model(self, model, name, artifacts=None, pip_requirements=None,
                     base_model=None, base_version=None):
        """Log and register a model. xgboost / sklearn / sentence-transformers are
        auto-detected and logged with their native flavor; anything else goes through
        pyfunc (pass a mlflow PythonModel). Returns the registry version (int).

        base_model/base_version: if this model uses another registered model, they're
        recorded as run params for lineage."""
        full = self._full_name(name)
        flavor = _flavor(model)
        mlflow.set_experiment(self.use_case)
        with mlflow.start_run(run_name=name,
                              tags={"type": "model", "use_case": self.use_case, "name": name}):
            if base_model is not None:
                mlflow.log_params({"base_model": base_model,
                                   "base_version": "" if base_version is None else base_version})
            if flavor is None:
                info = mlflow.pyfunc.log_model(
                    name="model", python_model=model, artifacts=artifacts,
                    pip_requirements=pip_requirements, registered_model_name=full)
            else:
                info = flavor.log_model(model, name="model",
                                        pip_requirements=pip_requirements,
                                        registered_model_name=full)
        client = MlflowClient()
        client.set_registered_model_tag(full, "use_case", self.use_case)
        client.set_model_version_tag(full, info.registered_model_version, "use_case", self.use_case)
        return int(info.registered_model_version)

    def get_model(self, name, version=None):
        """Load a registered pyfunc model. version=None loads the highest version."""
        full = self._full_name(name)
        if version is None:
            versions = MlflowClient().search_model_versions(f"name = '{full}'")
            if not versions:
                raise KeyError(f"no model {full}")
            version = max(int(v.version) for v in versions)
        return mlflow.pyfunc.load_model(f"models:/{full}/{version}")

    # --- runs -------------------------------------------------------------

    def submit_run(self, name, params=None, metrics=None, tags=None):
        """Create a run with standardized name/use_case tags. Returns run_id."""
        _validate("name", name)
        mlflow.set_experiment(self.use_case)
        run_tags = {"type": "run", "use_case": self.use_case, "name": name, **(tags or {})}
        with mlflow.start_run(run_name=name, tags=run_tags) as run:
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            return run.info.run_id

    def get_run(self, name):
        """Most recent run matching name, or None."""
        runs = self._find_runs(name, "run")
        return runs[0] if runs else None

    def list_runs(self):
        """All runs in this use_case as a DataFrame."""
        exp = mlflow.get_experiment_by_name(self.use_case)
        if exp is None:
            return pd.DataFrame()
        return mlflow.search_runs([exp.experiment_id])
