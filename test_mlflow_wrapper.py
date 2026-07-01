"""Self-check: python test_mlflow_wrapper.py — points MLflow at a temp file store."""
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from mlflow_wrapper import Store, _flavor


class Echo(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


def _fake(module):  # object whose class reports the given __module__
    return type("Fake", (object,), {"__module__": module})()


def test_flavor_detection():
    import xgboost
    from sklearn.linear_model import LinearRegression
    assert _flavor(xgboost.XGBRegressor()) is mlflow.xgboost            # both xgb+sklearn -> xgb
    assert _flavor(LinearRegression()) is mlflow.sklearn
    assert _flavor(_fake("sentence_transformers.X")) is mlflow.sentence_transformers
    assert _flavor(Echo()) is None                                     # generic pyfunc


def main():
    test_flavor_detection()
    with tempfile.TemporaryDirectory() as d:
        uri = (Path(d) / "mlruns").as_uri()
        store = Store("forecasting", tracking_uri=uri, registry_uri=uri)
        assert store.tracking_uri == uri and store.registry_uri == uri

        # datasets: version increments + version-pinned + latest retrieval
        v1 = store.submit_dataset(pd.DataFrame({"x": [1, 2]}), "sales")
        v2 = store.submit_dataset(pd.DataFrame({"x": [1, 2, 3]}), "sales")
        assert (v1, v2) == (1, 2), (v1, v2)
        assert store.list_dataset_versions("sales") == [1, 2]
        assert list(store.get_dataset("sales").x) == [1, 2, 3]          # latest
        assert list(store.get_dataset("sales", version=1).x) == [1, 2]

        # runs
        rid = store.submit_run("baseline", params={"lr": 0.1}, metrics={"rmse": 2.5})
        got = store.get_run("baseline")
        assert got is not None and got.info.run_id == rid
        assert got.data.params["lr"] == "0.1" and got.data.metrics["rmse"] == 2.5

        # pyfunc model: two versions, latest resolves, round-trips
        assert store.submit_model(Echo(), "ranker") == 1
        assert store.submit_model(Echo(), "ranker") == 2
        assert list(pd.DataFrame(store.get_model("ranker").predict(pd.DataFrame({"a": [7]}))).iloc[:, 0]) == [7]

        # native flavors auto-detected + loadable via pyfunc
        from sklearn.linear_model import LinearRegression
        import xgboost
        X, y = [[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0]
        store.submit_model(LinearRegression().fit(X, y), "linreg")
        store.submit_model(xgboost.XGBRegressor(n_estimators=3).fit(X, y), "xgb")
        assert len(store.get_model("linreg").predict(pd.DataFrame({"f": [1.0]}))) == 1
        assert len(store.get_model("xgb").predict(pd.DataFrame({"f": [1.0]}))) == 1

        # base_model lineage recorded as run params
        store.submit_model(Echo(), "downstream", base_model="ranker", base_version=2)
        run = store._find_runs("downstream", "model")[0]
        assert run.data.params["base_model"] == "ranker" and run.data.params["base_version"] == "2"

        # validation rejects junk at the boundary (ctor + methods)
        for bad in ("Bad Name", "a__b", "", "-x"):
            try:
                store.submit_run(bad)
                assert False, f"accepted bad name {bad!r}"
            except ValueError:
                pass
        try:
            Store("Bad Case")
            assert False, "accepted bad use_case"
        except ValueError:
            pass

    print("OK")


if __name__ == "__main__":
    main()
