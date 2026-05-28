"""Active-learning fine-tuning widget for sentence-transformer text classifiers.

Subclasses ``LabelingWidget``: keeps all the labeling UI, adds a training
panel, an uncertainty-based queue for the next-unlabeled button, and an
evaluation plot.

Takes a :class:`mlflow_model.SentenceTransformerClassifier` as input; all
training, prediction, and evaluation are delegated to that model.

Updated evaluation behavior:
- Shows one binary confusion matrix per category.
- Shows one binary confusion matrix per label.
- Every individual 2x2 matrix sums to n_eval.

Usage:
    from IPython.display import display
    from mlflow_model import SentenceTransformerClassifier
    from active_learning import ActiveLearner

    clf = SentenceTransformerClassifier(
        label_dict=labels,
        model_name_or_path="all-MiniLM-L6-v2",
        threshold=0.45,
    )

    w = ActiveLearner(
        model=clf,
        labeled_test_path="labeled_demo.csv",
        labeled_train_path="labeled_train_al.csv",
        unlabeled_train_path="unlabeled_train.csv",
        retrain_every=10,
        epochs=1,
        batch_size=16,
        query_strategy="margin",
    )
    display(w)
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Literal

import ipywidgets as widgets
import numpy as np
import pandas as pd

from labeler import LABEL_SEP, LabelingWidget
from mlflow_model import SentenceTransformerClassifier


QueryStrategy = Literal["margin", "least_confidence", "random"]


class ActiveLearner(LabelingWidget):
    def __init__(
        self,
        model: SentenceTransformerClassifier,
        labeled_test_path: str | Path,
        labeled_train_path: str | Path,
        unlabeled_train_path: str | Path,
        text_column: str = "text",
        labels_column: str = "labels",
        top_k_highlight: int = 3,
        retrain_every: int = 10,
        val_fraction: float = 0.0,
        epochs: int = 1,
        batch_size: int = 16,
        threshold: float | None = None,
        query_strategy: QueryStrategy = "margin",
        random_state: int = 42,
        model_save_path: str | Path | None = None,
        mlflow_experiment: str | None = None,
    ):
        self.model = model
        self.labeled_test_path = Path(labeled_test_path)
        self.labeled_train_path = Path(labeled_train_path)
        self.unlabeled_train_path = Path(unlabeled_train_path)

        self._retrain_every = retrain_every
        self._val_fraction = val_fraction
        self._epochs = epochs
        self._batch_size = batch_size
        self._threshold = float(model.threshold if threshold is None else threshold)
        self._query_strategy = query_strategy
        self._random_state = random_state
        self._model_save_path = Path(model_save_path) if model_save_path else None
        self._mlflow_experiment = mlflow_experiment

        self._labels_at_last_train: int = 0
        self._queue: list[int] = []
        self._train_history: list[dict] = []
        self._eval_df: pd.DataFrame | None = None
        self._eval_texts: list[str] = []
        self._eval_truth: np.ndarray | None = None

        pool_df, n_labeled_loaded, n_unlabeled_loaded = self._build_pool(
            text_column,
            labels_column,
        )
        test_df = self._load_path(
            self.labeled_test_path,
            labels_column,
            require_labels=True,
        )

        super().__init__(
            embed_model=model,
            label_dict=model.label_dict,
            df=pool_df,
            save_path=self.labeled_train_path,
            text_column=text_column,
            labels_column=labels_column,
            top_k_highlight=top_k_highlight,
        )

        self.set_eval_df(test_df)

        print(
            f"[ActiveLearner] pool ready: {len(pool_df)} rows total "
            f"({n_labeled_loaded} pre-labeled from {self.labeled_train_path.name}, "
            f"{n_unlabeled_loaded} new from {self.unlabeled_train_path.name})"
        )

        self._labels_at_last_train = self._count_labels()
        self._build_train_panel()
        self._update_train_status()

        try:
            self._initial_snapshot = self._compute_eval_snapshot(
                float(self.threshold_input.value)
            )
            self._render_eval(
                self.initial_plot_out,
                self.initial_metrics_html,
                self._initial_snapshot,
                source_label="Zero-shot",
            )
        except Exception as e:
            self._initial_snapshot = None
            print(f"[ActiveLearner] initial zero-shot eval skipped: {e}")

        if self._mlflow_experiment is not None:
            try:
                import mlflow

                mlflow.set_experiment(self._mlflow_experiment)
            except Exception as e:
                print(f"[ActiveLearner] mlflow experiment setup failed: {e}")

    def _build_pool(
        self,
        text_column: str,
        labels_column: str,
    ) -> tuple[pd.DataFrame, int, int]:
        """Combine ``labeled_train`` + ``unlabeled_train`` into one pool."""
        unlabeled_df = self._load_path(
            self.unlabeled_train_path,
            labels_column,
            require_labels=False,
        )

        if self.labeled_train_path.exists():
            try:
                labeled_df = self._load_path(
                    self.labeled_train_path,
                    labels_column,
                    require_labels=False,
                )
            except Exception as e:
                print(
                    f"[ActiveLearner] could not read {self.labeled_train_path}: "
                    f"{e} — starting with no pre-labeled rows"
                )
                labeled_df = pd.DataFrame(
                    {text_column: [], labels_column: [[] for _ in range(0)]}
                )
        else:
            labeled_df = pd.DataFrame(
                {text_column: [], labels_column: [[] for _ in range(0)]}
            )

        labeled_df = labeled_df[
            labeled_df[labels_column].apply(bool)
        ].reset_index(drop=True)

        seen = set(labeled_df[text_column].astype(str))
        unlabeled_new = unlabeled_df[
            ~unlabeled_df[text_column].astype(str).isin(seen)
        ].reset_index(drop=True)

        combined = pd.concat([labeled_df, unlabeled_new], ignore_index=True)
        if labels_column not in combined.columns:
            combined[labels_column] = [[] for _ in range(len(combined))]
        return combined, len(labeled_df), len(unlabeled_new)

    def _save(self) -> None:
        """Persist only labeled rows back to ``labeled_train_path``."""
        path = self.labeled_train_path
        path.parent.mkdir(parents=True, exist_ok=True)
        labeled = self.df[
            self.df[self.labels_column].apply(bool)
        ].reset_index(drop=True)

        ext = path.suffix.lower()
        if ext == ".csv":
            tmp = labeled.copy()
            tmp[self.labels_column] = tmp[self.labels_column].apply(json.dumps)
            tmp.to_csv(path, index=False)
        elif ext == ".parquet":
            labeled.to_parquet(path, index=False)
        elif ext == ".json":
            labeled.to_json(path, orient="records", indent=2)
        elif ext in (".pkl", ".pickle"):
            labeled.to_pickle(path)
        else:
            raise ValueError(f"unsupported save_path extension: {ext}")

    @staticmethod
    def _load_path(
        path: Path,
        labels_column: str,
        require_labels: bool,
    ) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"input file not found: {path}")

        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".json":
            df = pd.read_json(path, orient="records")
        elif ext in (".pkl", ".pickle"):
            df = pd.read_pickle(path)
        else:
            raise ValueError(f"unsupported extension for {path}: {ext}")

        if labels_column in df.columns:
            df[labels_column] = df[labels_column].apply(
                LabelingWidget._coerce_label_list
            )
        elif require_labels:
            raise ValueError(
                f"{path} is missing the '{labels_column}' column"
            )
        else:
            df[labels_column] = [[] for _ in range(len(df))]
        return df

    def set_eval_df(self, eval_df: pd.DataFrame) -> None:
        if self.text_column not in eval_df.columns:
            raise ValueError(f"eval_df is missing text column '{self.text_column}'")
        if self.labels_column not in eval_df.columns:
            raise ValueError(f"eval_df is missing labels column '{self.labels_column}'")

        df = eval_df.copy().reset_index(drop=True)
        df[self.labels_column] = df[self.labels_column].apply(self._coerce_label_list)
        df = df[df[self.labels_column].apply(bool)].reset_index(drop=True)

        self._eval_df = df
        self._eval_texts = df[self.text_column].astype(str).tolist()

        all_keys = [f"{cat}{LABEL_SEP}{lab}" for cat, lab, _ in self.flat_labels]
        truth = np.zeros((len(df), len(all_keys)), dtype=bool)
        for r, labs in enumerate(df[self.labels_column]):
            label_set = set(labs)
            for c, key in enumerate(all_keys):
                truth[r, c] = key in label_set
        self._eval_truth = truth

    def _build_train_panel(self) -> None:
        self.retrain_input = widgets.BoundedIntText(
            value=self._retrain_every,
            min=0,
            max=10_000,
            description="Retrain every:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="180px"),
        )
        self.threshold_input = widgets.FloatSlider(
            value=self._threshold,
            min=0.0,
            max=1.0,
            step=0.01,
            description="Threshold:",
            readout_format=".2f",
            continuous_update=False,
            layout=widgets.Layout(width="320px"),
        )
        self.val_input = widgets.BoundedFloatText(
            value=self._val_fraction,
            min=0.0,
            max=0.9,
            step=0.05,
            description="Val frac:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="140px"),
        )

        (
            self._model_hp_specs,
            self._model_hp_widgets,
            self._pso_select_widgets,
        ) = self._build_model_hp_controls()
        # Back-compat aliases so the rest of the widget can still read
        # widget.epochs_input.value / widget.batch_input.value.
        self.epochs_input = self._model_hp_widgets.get("epochs")
        self.batch_input = self._model_hp_widgets.get("batch_size")
        self.strategy_input = widgets.Dropdown(
            options=["margin", "least_confidence", "random"],
            value=self._query_strategy,
            description="Query:",
            layout=widgets.Layout(width="220px"),
        )

        self.train_btn = widgets.Button(
            description="Train now",
            button_style="primary",
            layout=widgets.Layout(width="120px"),
        )
        self.eval_btn = widgets.Button(
            description="Evaluate",
            layout=widgets.Layout(width="120px"),
        )
        self.refresh_queue_btn = widgets.Button(
            description="Refresh queue",
            layout=widgets.Layout(width="140px"),
        )
        self.save_model_btn = widgets.Button(
            description="Save model",
            layout=widgets.Layout(width="120px"),
            disabled=self._model_save_path is None,
        )

        # ── PSO hyperparameter tuning controls ──
        self.pso_particles_input = widgets.BoundedIntText(
            value=5, min=2, max=30, description="Particles:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )
        self.pso_iter_input = widgets.BoundedIntText(
            value=4, min=1, max=30, description="Iter:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="130px"),
        )
        self.pso_metric_input = widgets.Dropdown(
            options=["macro_f1", "micro_f1", "val_loss"],
            value="macro_f1",
            description="Metric:",
            layout=widgets.Layout(width="220px"),
        )
        self.pso_tune_thr_input = widgets.Checkbox(
            value=True,
            description="Tune thr",
            indent=False,
            layout=widgets.Layout(width="140px"),
            tooltip=(
                "When metric is macro/micro F1, also search the threshold "
                "per particle by scanning [0.01, 0.99] on the trained model."
            ),
        )
        self.tune_btn = widgets.Button(
            description="Tune (PSO)",
            button_style="warning",
            layout=widgets.Layout(width="140px"),
            tooltip="Particle-swarm search over the model hyperparameters",
        )

        self.train_btn.on_click(lambda b: self._train())
        self.eval_btn.on_click(lambda b: self._evaluate_and_plot())
        self.refresh_queue_btn.on_click(lambda b: self._refresh_queue(verbose=True))
        self.save_model_btn.on_click(lambda b: self._save_model())
        self.tune_btn.on_click(lambda b: self._tune_pso())
        self.threshold_input.observe(self._on_threshold_change, names="value")

        self.train_status = widgets.HTML()
        self._log_lines: list[str] = []
        self.train_log_html = widgets.HTML(value=self._render_terminal([]))

        out_layout = widgets.Layout(border="1px solid #eee", margin="6px 0")
        self.current_plot_out = widgets.Output(layout=out_layout)
        self.current_metrics_html = widgets.HTML()
        self.initial_plot_out = widgets.Output(layout=out_layout)
        self.initial_metrics_html = widgets.HTML()

        self.plot_out = self.current_plot_out
        self.metrics_html = self.current_metrics_html

        self.plot_tabs = widgets.Tab(
            children=[
                widgets.VBox([self.current_plot_out, self.current_metrics_html]),
                widgets.VBox([self.initial_plot_out, self.initial_metrics_html]),
            ]
        )
        self.plot_tabs.set_title(0, "Current")
        self.plot_tabs.set_title(1, "Initial (zero-shot)")

        # Pair each value control with its "PSO" checkbox so they line up.
        hp_pairs = [
            widgets.HBox([
                self._pso_select_widgets[name],
                self._model_hp_widgets[name],
            ])
            for name in self._model_hp_widgets
        ]
        # Chunk the pairs into rows of 3 so they don't overflow horizontally.
        hp_rows = [
            widgets.HBox(hp_pairs[i:i + 3])
            for i in range(0, len(hp_pairs), 3)
        ]
        model_section = [
            widgets.HTML(
                f"<div style='margin-top:6px;font-size:12px;color:#666'>"
                f"Model hyperparameters · "
                f"<code>{type(self.model).__name__}</code></div>"
            ),
            *hp_rows,
        ]

        pso_section = [
            widgets.HTML(
                "<div style='margin-top:6px;font-size:12px;color:#666'>"
                "PSO hyperparameter tuning</div>"
            ),
            widgets.HBox([
                self.pso_particles_input,
                self.pso_iter_input,
                self.pso_metric_input,
                self.pso_tune_thr_input,
                self.tune_btn,
            ]),
        ]

        panel = widgets.VBox(
            [
                widgets.HTML(
                    "<hr style='margin:14px 0 6px 0'>"
                    "<h4 style='margin:4px 0'>Active learning trainer</h4>"
                ),
                widgets.HBox([self.retrain_input, self.val_input]),
                widgets.HBox([self.threshold_input, self.strategy_input]),
                *model_section,
                *pso_section,
                widgets.HBox(
                    [
                        self.train_btn,
                        self.eval_btn,
                        self.refresh_queue_btn,
                        self.save_model_btn,
                    ]
                ),
                self.train_status,
                self.train_log_html,
                self.plot_tabs,
            ]
        )

        self.children = (*self.children, panel)

    def _count_labels(self) -> int:
        return int(sum(1 for x in self.df[self.labels_column] if x))

    def _build_model_hp_controls(
        self,
    ) -> tuple[list[dict], dict[str, widgets.Widget], dict[str, widgets.Widget]]:
        """Construct ipywidgets controls from ``model.train_hyperparams()``.

        Returns ``(specs, name_to_widget, name_to_pso_select)``.

        - ``name_to_widget`` maps the hyperparam name to the value-editing
          widget (BoundedIntText, FloatLogSlider, Checkbox, Dropdown, …).
        - ``name_to_pso_select`` maps the same name to a tiny ``Checkbox``
          that decides whether PSO should search this hyperparam. Its
          initial value comes from ``spec.get('pso', True)``.

        Models without ``train_hyperparams`` get a minimal fallback
        (epochs + batch_size, both PSO-off).
        """
        if hasattr(self.model, "train_hyperparams"):
            try:
                specs = list(self.model.train_hyperparams())
            except Exception as e:
                print(f"[ActiveLearner] train_hyperparams() raised: {e}")
                specs = []
        else:
            specs = []

        if not specs:
            # Fallback when a model doesn't declare anything — batch size
            # is owned by the model, so it's deliberately absent here.
            specs = [
                {"name": "epochs", "label": "Epochs", "kind": "int",
                 "default": self._epochs, "min": 1, "max": 100, "pso": False},
            ]

        # Constructor-time overrides for common knobs so the widget honors
        # values the user already passed to ActiveLearner(...).
        overrides = {
            "epochs": self._epochs,
            "batch_size": self._batch_size,
        }

        name_to_widget: dict[str, widgets.Widget] = {}
        name_to_pso: dict[str, widgets.Widget] = {}
        style = {"description_width": "initial"}
        for spec in specs:
            name = spec["name"]
            label = spec.get("label", name) + ":"
            kind = spec.get("kind", "float")
            default = overrides.get(name, spec.get("default"))
            desc = spec.get("description", "")
            layout = widgets.Layout(width=spec.get("width", "170px"))

            w: widgets.Widget
            if kind == "int":
                w = widgets.BoundedIntText(
                    value=int(default),
                    min=int(spec.get("min", 0)),
                    max=int(spec.get("max", 10**6)),
                    step=int(spec.get("step", 1)),
                    description=label, style=style, layout=layout,
                )
            elif kind == "float":
                w = widgets.BoundedFloatText(
                    value=float(default),
                    min=float(spec.get("min", 0.0)),
                    max=float(spec.get("max", 1.0)),
                    step=float(spec.get("step", 0.01)),
                    description=label, style=style, layout=layout,
                )
            elif kind == "log_float":
                lo = float(spec.get("min", 1e-6))
                hi = float(spec.get("max", 1.0))
                base = 10.0
                # FloatLogSlider takes log-base exponents.
                w = widgets.FloatLogSlider(
                    value=float(default),
                    min=np.log10(lo), max=np.log10(hi),
                    step=0.1, base=base, readout_format=".1e",
                    description=label, style=style,
                    layout=widgets.Layout(width=spec.get("width", "280px")),
                    continuous_update=False,
                )
            elif kind == "bool":
                w = widgets.Checkbox(
                    value=bool(default),
                    description=spec.get("label", name),
                    indent=False,
                    layout=widgets.Layout(width=spec.get("width", "150px")),
                )
            elif kind == "choice":
                w = widgets.Dropdown(
                    options=list(spec["choices"]),
                    value=default,
                    description=label, style=style,
                    layout=widgets.Layout(width=spec.get("width", "240px")),
                )
            else:
                print(f"[ActiveLearner] unknown hyperparam kind: {kind!r}")
                continue
            if desc:
                try:
                    w.tooltip = desc
                except Exception:
                    pass
            name_to_widget[name] = w

            pso_default = bool(spec.get("pso", True))
            chk = widgets.Checkbox(
                value=pso_default,
                description="PSO",
                indent=False,
                layout=widgets.Layout(width="62px"),
                tooltip=f"Include {name!r} in PSO search",
            )
            name_to_pso[name] = chk
        return specs, name_to_widget, name_to_pso

    def _pso_active_specs(self) -> tuple[list[dict], dict[str, object]]:
        """Split declared HPs into PSO-searched specs and fixed kwargs.

        Returns ``(active_specs, fixed_hp_values)`` — the first is fed to PSO,
        the second is merged into every particle's ``train()`` call.
        """
        active: list[dict] = []
        fixed: dict[str, object] = {}
        for spec in self._model_hp_specs:
            name = spec["name"]
            chk = self._pso_select_widgets.get(name)
            if chk is not None and bool(chk.value):
                active.append(spec)
            else:
                w = self._model_hp_widgets.get(name)
                if w is not None:
                    fixed[name] = w.value
        return active, fixed

    def _collect_model_hp_values(self) -> dict[str, object]:
        """Read current values from the dynamic hyperparameter widgets."""
        return {name: w.value for name, w in self._model_hp_widgets.items()}

    # ── Particle-swarm hyperparameter search ────────────────────────────
    @staticmethod
    def _specs_to_pso_bounds(specs: list[dict]) -> list[tuple[float, float]]:
        """Continuous bounds for each spec; log/categorical encoded numerically."""
        bounds = []
        for spec in specs:
            kind = spec["kind"]
            if kind in ("int", "float"):
                bounds.append((float(spec["min"]), float(spec["max"])))
            elif kind == "log_float":
                bounds.append((
                    float(np.log10(spec["min"])),
                    float(np.log10(spec["max"])),
                ))
            elif kind == "bool":
                bounds.append((0.0, 1.0))
            elif kind == "choice":
                bounds.append((0.0, float(len(spec["choices"]) - 1)))
            else:
                bounds.append((0.0, 1.0))
        return bounds

    @staticmethod
    def _decode_pso_position(specs: list[dict], position: np.ndarray) -> dict:
        out: dict[str, object] = {}
        for spec, val in zip(specs, position):
            name = spec["name"]
            kind = spec["kind"]
            if kind == "int":
                out[name] = int(round(float(val)))
            elif kind == "float":
                out[name] = float(val)
            elif kind == "log_float":
                out[name] = float(10 ** float(val))
            elif kind == "bool":
                out[name] = bool(round(float(val)))
            elif kind == "choice":
                idx = int(round(float(val)))
                idx = max(0, min(idx, len(spec["choices"]) - 1))
                out[name] = spec["choices"][idx]
        return out

    @staticmethod
    def _best_threshold_for_metric(
        sims: np.ndarray,
        truth: np.ndarray,
        metric: str,
        grid: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Sweep thresholds, return ``(best_threshold, best_score)``.

        Works for ``macro_f1`` and ``micro_f1``. The grid defaults to
        ``arange(0.01, 1.00, 0.01)``.
        """
        if grid is None:
            grid = np.arange(0.01, 1.00, 0.01)
        best_score = -1.0
        best_thr = 0.5
        truth_b = truth.astype(bool)
        for thr in grid:
            preds = sims >= float(thr)
            tp = (preds & truth_b).sum(axis=0).astype(float)
            fp = (preds & ~truth_b).sum(axis=0).astype(float)
            fn = (~preds & truth_b).sum(axis=0).astype(float)
            if metric == "macro_f1":
                with np.errstate(divide="ignore", invalid="ignore"):
                    p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
                    r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
                    f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
                score = float(f1.mean())
            elif metric == "micro_f1":
                tp_s, fp_s, fn_s = float(tp.sum()), float(fp.sum()), float(fn.sum())
                pp = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
                rr = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
                score = (2 * pp * rr / (pp + rr)) if (pp + rr) > 0 else 0.0
            else:
                score = 0.0
            if score > best_score:
                best_score = float(score)
                best_thr = float(thr)
        return best_thr, best_score

    def _format_hp_for_log(self, hp: dict) -> str:
        def fmt(v):
            if isinstance(v, float):
                if 0 < abs(v) < 0.01 or abs(v) >= 1000:
                    return f"{v:.2e}"
                return f"{v:.4g}"
            return str(v)
        return ", ".join(f"{k}={fmt(v)}" for k, v in hp.items())

    def _pso_evaluate_particle(
        self,
        position: np.ndarray,
        specs: list[dict],
        train_pairs,
        eval_pairs,
        initial_state,
        metric: str,
        thr: float,
        tune_threshold: bool,
        fixed_hp: dict[str, object] | None = None,
    ) -> tuple[float, dict, dict]:
        """Train + evaluate one particle. Returns (fitness, hp_decoded, info).

        ``specs`` is the PSO-active subset of the model's hyperparameters.
        ``fixed_hp`` are the values for hyperparameters NOT in ``specs`` —
        they are merged into every ``train()`` call so the particle only
        varies the selected dimensions.

        When ``tune_threshold`` is true and the metric is an F1, the
        threshold is selected per particle by scanning ``[0.01, 0.99]`` on
        the model's scores. The chosen threshold is returned in ``info``
        as ``best_threshold``.
        """
        if hasattr(self.model, "restore_state"):
            try:
                self.model.restore_state(initial_state)
            except Exception as e:
                print(f"[PSO] restore_state failed: {e}")

        decoded = self._decode_pso_position(specs, position)
        merged = {**(fixed_hp or {}), **decoded}
        epochs = int(merged.pop("epochs", 1)) if "epochs" in merged else 1
        if "batch_size" in merged:
            batch_size: int | None = int(merged.pop("batch_size"))
        else:
            batch_size = None  # let the model use its own default
        hp_values = merged

        info: dict[str, object] = {}
        try:
            metrics = self.model.train(
                train_pairs,
                epochs=epochs,
                batch_size=batch_size,
                log_to_mlflow=False,
                progress_callback=None,
                eval_text_label_pairs=eval_pairs,
                **hp_values,
            )
        except Exception as e:
            return (
                float("inf"),
                _hp_for_return(),
                {"error": str(e)},
            )

        def _hp_for_return() -> dict:
            out = {**hp_values, "epochs": epochs}
            if batch_size is not None:
                out["batch_size"] = batch_size
            return out

        if metric == "val_loss":
            score = float(metrics.get("mean_val_loss", float("inf")))
            if np.isnan(score):
                score = float(metrics.get("mean_loss", float("inf")))
            info["val_loss"] = score
            info["best_threshold"] = thr
        else:
            try:
                sims = self.model.predict_scores(self._eval_texts)
            except Exception as e:
                return (
                    float("inf"),
                    _hp_for_return(),
                    {"error": str(e)},
                )
            if tune_threshold:
                best_thr, best_score = self._best_threshold_for_metric(
                    sims, self._eval_truth, metric,
                )
                info["best_threshold"] = best_thr
            else:
                best_thr = thr
                _, best_score = self._best_threshold_for_metric(
                    sims, self._eval_truth, metric, grid=np.array([thr]),
                )
                info["best_threshold"] = thr
            info[metric] = best_score
            score = -float(best_score)

        return (
            score,
            _hp_for_return(),
            info,
        )

    def _tune_pso(self) -> None:
        """Run PSO over the user-selected subset of model hyperparameters."""
        if not hasattr(self.model, "train_hyperparams"):
            self._log_train("PSO: model has no train_hyperparams(); aborting")
            return

        all_specs = list(self.model.train_hyperparams())
        if not all_specs:
            self._log_train("PSO: empty train_hyperparams(); aborting")
            return

        specs, fixed_hp = self._pso_active_specs()
        if not specs:
            self._log_train(
                "PSO: no hyperparameters checked for search; nothing to do"
            )
            return

        train_pairs = self._build_text_label_pairs()
        if sum(len(labs) for _, labs in train_pairs) < 2:
            self._log_train("PSO: need at least 2 labeled (text, label) pairs")
            return

        eval_pairs = self._build_eval_text_label_pairs()
        metric = str(self.pso_metric_input.value)
        if metric != "val_loss" and (
            self._eval_truth is None or not self._eval_texts
        ):
            self._log_train(f"PSO: metric={metric} requires eval_df; aborting")
            return

        n_particles = int(self.pso_particles_input.value)
        n_iter = int(self.pso_iter_input.value)
        tune_threshold = bool(self.pso_tune_thr_input.value)
        bounds = self._specs_to_pso_bounds(specs)
        d = len(bounds)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        thr = float(self.threshold_input.value)

        # Disable buttons during the search to avoid concurrent retrains.
        prev_disabled = (
            self.train_btn.disabled, self.tune_btn.disabled, self.eval_btn.disabled,
        )
        self.train_btn.disabled = True
        self.tune_btn.disabled = True
        self.eval_btn.disabled = True

        self._log_lines = []
        tune_thr_str = (
            "+threshold" if tune_threshold and metric in ("macro_f1", "micro_f1")
            else "(fixed thr)"
        )
        active_names = [s["name"] for s in specs]
        self._log_train(
            f"PSO start  particles={n_particles}  iterations={n_iter}  "
            f"metric={metric}{tune_thr_str}  dims={d}  "
            f"total_evals={n_particles * (n_iter + 1)}"
        )
        self._log_train(f"  searching: {active_names}")
        if fixed_hp:
            self._log_train(
                f"  fixed:     [{self._format_hp_for_log(fixed_hp)}]"
            )

        initial_state = None
        if hasattr(self.model, "snapshot_state"):
            try:
                initial_state = self.model.snapshot_state()
                self._log_train("PSO: snapshotted model state for restoration between evals")
            except Exception as e:
                self._log_train(f"PSO: snapshot_state failed: {e} (evals will share state)")

        try:
            rng = np.random.default_rng(self._random_state)
            positions = lo + (hi - lo) * rng.random((n_particles, d))
            velocities = (hi - lo) * (rng.random((n_particles, d)) - 0.5) * 0.1
            # Per-particle best threshold (populated by the eval loop).
            pbest_thr = np.full(n_particles, thr, dtype=float)
            gbest_thr = thr

            def eval_swarm(
                positions: np.ndarray, label: str,
            ) -> tuple[np.ndarray, np.ndarray]:
                fitness = np.empty(len(positions))
                thrs = np.empty(len(positions))
                for i, pos in enumerate(positions):
                    score, hp_decoded, info = self._pso_evaluate_particle(
                        pos, specs, train_pairs, eval_pairs,
                        initial_state, metric, thr, tune_threshold,
                        fixed_hp=fixed_hp,
                    )
                    fitness[i] = score
                    thrs[i] = float(info.get("best_threshold", thr))
                    if "error" in info:
                        self._log_train(
                            f"  {label} p{i + 1}/{len(positions)}  FAILED: {info['error']}"
                        )
                    else:
                        if metric == "val_loss":
                            metric_str = f"val_loss={score:.4f}"
                        else:
                            metric_str = (
                                f"{metric}={-score:.4f}  thr={thrs[i]:.2f}"
                            )
                        self._log_train(
                            f"  {label} p{i + 1}/{len(positions)}  {metric_str}  "
                            f"[{self._format_hp_for_log(hp_decoded)}]"
                        )
                return fitness, thrs

            fitness, thrs = eval_swarm(positions, "init")
            pbest_pos = positions.copy()
            pbest_fit = fitness.copy()
            pbest_thr = thrs.copy()
            gbest_idx = int(np.argmin(pbest_fit))
            gbest_pos = pbest_pos[gbest_idx].copy()
            gbest_fit = float(pbest_fit[gbest_idx])
            gbest_thr = float(pbest_thr[gbest_idx])

            thr_suffix = (
                f"  thr={gbest_thr:.2f}"
                if metric != "val_loss" else ""
            )
            self._log_train(
                f"init done  best {metric}="
                f"{(-gbest_fit if metric != 'val_loss' else gbest_fit):.4f}"
                f"{thr_suffix}"
            )

            inertia, cognitive, social = 0.5, 1.5, 1.5
            for it in range(n_iter):
                r1 = rng.random((n_particles, d))
                r2 = rng.random((n_particles, d))
                velocities = (
                    inertia * velocities
                    + cognitive * r1 * (pbest_pos - positions)
                    + social * r2 * (gbest_pos[None, :] - positions)
                )
                positions = np.clip(positions + velocities, lo, hi)
                fitness, thrs = eval_swarm(positions, f"iter{it + 1}")
                improved = fitness < pbest_fit
                pbest_pos[improved] = positions[improved]
                pbest_fit[improved] = fitness[improved]
                pbest_thr[improved] = thrs[improved]
                cur_best = int(np.argmin(pbest_fit))
                if pbest_fit[cur_best] < gbest_fit:
                    gbest_pos = pbest_pos[cur_best].copy()
                    gbest_fit = float(pbest_fit[cur_best])
                    gbest_thr = float(pbest_thr[cur_best])
                thr_suffix = (
                    f"  thr={gbest_thr:.2f}"
                    if metric != "val_loss" else ""
                )
                self._log_train(
                    f"iter {it + 1}/{n_iter} done  best {metric}="
                    f"{(-gbest_fit if metric != 'val_loss' else gbest_fit):.4f}"
                    f"{thr_suffix}"
                )

            best_hp = self._decode_pso_position(specs, gbest_pos)
            thr_str = (
                f"  thr={gbest_thr:.2f}" if metric != "val_loss" else ""
            )
            score_str = (
                f"val_loss={gbest_fit:.4f}"
                if metric == "val_loss"
                else f"{metric}={-gbest_fit:.4f}{thr_str}"
            )
            self._log_train(
                f"PSO done  best {score_str}  hp=[{self._format_hp_for_log(best_hp)}]"
            )

            # Push the best values back into the HP widgets so a subsequent
            # Train-now click uses them.
            for name, val in best_hp.items():
                w = self._model_hp_widgets.get(name)
                if w is None:
                    continue
                try:
                    w.value = val
                except Exception:
                    try:
                        w.value = max(w.min, min(w.max, val))  # type: ignore[attr-defined]
                    except Exception:
                        pass

            # Push the best threshold to the slider so eval reflects it.
            if metric != "val_loss" and tune_threshold:
                try:
                    self.threshold_input.value = float(gbest_thr)
                except Exception:
                    pass
                self.model.threshold = float(gbest_thr)

            # Restore the original model state so the user can decide whether
            # to re-train with the new HPs.
            if initial_state is not None and hasattr(self.model, "restore_state"):
                self.model.restore_state(initial_state)
                self._log_train("PSO: restored model to pre-tune state")

            self._update_train_status(
                f"<span style='color:#2a8a2a'>✓ PSO done · best {score_str}</span>"
            )
        finally:
            self.train_btn.disabled, self.tune_btn.disabled, self.eval_btn.disabled = prev_disabled

    def _get_prediction_threshold(self) -> float | None:
        if not hasattr(self, "threshold_input"):
            return None
        return float(self.threshold_input.value)

    def _on_threshold_change(self, change) -> None:
        self._render()
        if getattr(self, "_initial_snapshot", None):
            self._render_eval(
                self.initial_plot_out,
                self.initial_metrics_html,
                self._initial_snapshot,
                source_label="Zero-shot",
            )

    @staticmethod
    def _render_terminal(lines: list[str]) -> str:
        visible = lines[-15:] if lines else []
        body = "<br>".join(html.escape(line) for line in visible) or "&nbsp;"
        return (
            "<div style='font-family:Consolas,Menlo,monospace; "
            "background:#1e1e1e; color:#a0e0a0; padding:8px 10px; "
            "border-radius:4px; height:200px; overflow-y:auto; "
            "font-size:11px; line-height:1.5; white-space:pre; "
            "border:1px solid #444; margin:6px 0'>"
            + body
            + "</div>"
        )

    def _log_train(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log_lines.append(f"[{ts}] {line}")
        self.train_log_html.value = self._render_terminal(self._log_lines)

    def _train_progress_callback(self, event: str, **kwargs) -> None:
        if event == "start":
            self._log_lines = []
            self._log_train(
                f"start  train_pairs={kwargs['n_pairs']}  "
                f"eval_pairs={kwargs.get('n_eval_pairs', 0)}  "
                f"epochs={kwargs['epochs']}  batch={kwargs['batch_size']}  "
                f"trainable={kwargs['trainable_params']:,}/"
                f"{kwargs['total_params']:,}  lr={kwargs['lr']:.2e}"
            )
        elif event == "epoch_start":
            self._log_train(f"epoch {kwargs['epoch']}/{kwargs['n_epochs']}")
        elif event == "batch":
            self._log_train(
                f"  batch {kwargs['batch']:3d}/{kwargs['n_batches']:<3d}  "
                f"train_loss={kwargs['loss']:.4f}"
            )
        elif event == "epoch_end":
            val = kwargs.get("mean_val_loss")
            val_str = (
                f"  val_loss={val:.4f}"
                if val is not None and not np.isnan(val)
                else "  val_loss=n/a"
            )
            self._log_train(
                f"epoch {kwargs['epoch']}/{kwargs['n_epochs']} done  "
                f"train_loss={kwargs['mean_loss']:.4f}{val_str}"
            )
        elif event == "end":
            val = kwargs.get("mean_val_loss")
            val_str = (
                f"  final_val_loss={val:.4f}"
                if val is not None and not np.isnan(val)
                else ""
            )
            self._log_train(
                f"finished  total_batches={kwargs['n_batches']}  "
                f"train_loss={kwargs['mean_loss']:.4f}{val_str}"
            )

    def _compute_eval_snapshot(self, thr: float) -> dict:
        """Encode + score every eligible row at the current model state."""
        snapshot = {"train": None, "test": None, "threshold": thr}

        col = self.df.columns.get_loc(self.labels_column)
        text_col = self.df.columns.get_loc(self.text_column)
        train_idxs = [i for i in range(len(self.df)) if self.df.iat[i, col]]
        all_keys = list(self.model.label_keys)

        if train_idxs:
            texts = [str(self.df.iat[i, text_col]) for i in train_idxs]
            truth = np.zeros((len(train_idxs), len(all_keys)), dtype=bool)
            for r, i in enumerate(train_idxs):
                lbls = set(self.df.iat[i, col])
                for c, key in enumerate(all_keys):
                    truth[r, c] = key in lbls
            embs = np.asarray(
                self.embed_model.encode(texts, normalize_embeddings=True)
            )
            sims = embs @ self.label_embeddings.T
            snapshot["train"] = {
                "sims": sims,
                "truth": truth,
                "n_eval": len(texts),
            }

        if self._eval_df is not None and len(self._eval_texts) > 0:
            embs = np.asarray(
                self.embed_model.encode(self._eval_texts, normalize_embeddings=True)
            )
            sims = embs @ self.label_embeddings.T
            snapshot["test"] = {
                "sims": sims,
                "truth": self._eval_truth,
                "n_eval": len(self._eval_texts),
            }

        return snapshot

    def _compute_row_scores(
        self,
        preds: np.ndarray,
        truth: np.ndarray,
    ) -> dict:
        """Per-row category correctness."""
        cats = list(self.cat_to_flat_idx.keys())
        n_rows = preds.shape[0]
        n_cats = len(cats)
        cat_correct = np.zeros((n_rows, n_cats), dtype=bool)
        for ci, cat in enumerate(cats):
            idxs = np.array(self.cat_to_flat_idx[cat], dtype=int)
            cat_correct[:, ci] = (preds[:, idxs] == truth[:, idxs]).all(axis=1)
        per_row_score = cat_correct.sum(axis=1).astype(int)
        fully_correct = cat_correct.all(axis=1)
        return {
            "categories": cats,
            "cat_correct": cat_correct,
            "per_row_score": per_row_score,
            "fully_correct": fully_correct,
            "fully_correct_count": int(fully_correct.sum()),
            "n_rows": int(n_rows),
            "n_categories": int(n_cats),
            "mean_cat_correct": float(per_row_score.mean()) if n_rows else 0.0,
        }

    @staticmethod
    def _metrics_from_sims(sims: np.ndarray, truth: np.ndarray, thr: float) -> dict:
        preds = sims >= thr
        tp = (preds & truth).sum(axis=0).astype(float)
        fp = (preds & ~truth).sum(axis=0).astype(float)
        fn = (~preds & truth).sum(axis=0).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1 = np.where(
                precision + recall > 0,
                2 * precision * recall / (precision + recall),
                0.0,
            )
        macro_f1 = float(f1.mean())
        tp_s, fp_s, fn_s = float(tp.sum()), float(fp.sum()), float(fn.sum())
        micro_p = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
        micro_r = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        micro_f1 = (
            2 * micro_p * micro_r / (micro_p + micro_r)
            if (micro_p + micro_r) > 0
            else 0.0
        )
        return {
            "preds": preds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "micro_p": micro_p,
            "micro_r": micro_r,
        }

    def _update_train_status(self, extra: str = "") -> None:
        n = self._count_labels()
        every = int(self.retrain_input.value) if hasattr(self, "retrain_input") else self._retrain_every
        since = n - self._labels_at_last_train
        if every > 0:
            remaining = max(0, every - since)
            base = (
                f"{n} labeled · {since} since last train · "
                f"{remaining} until auto-retrain (every {every})"
            )
        else:
            base = f"{n} labeled · auto-retrain disabled"
        msg = f"<span style='color:#555'>{base}</span>"
        if extra:
            msg += f" &nbsp; {extra}"
        self.train_status.value = msg

    def _on_toggle(self, change) -> None:
        super()._on_toggle(change)
        every = int(self.retrain_input.value)
        if every > 0 and (self._count_labels() - self._labels_at_last_train) >= every:
            self._train()
        else:
            self._update_train_status()

    def _jump_next_unlabeled(self) -> None:
        col = self.df.columns.get_loc(self.labels_column)
        while self._queue:
            nxt = self._queue.pop(0)
            if 0 <= nxt < len(self.df) and not self.df.iat[nxt, col]:
                self._jump(nxt)
                self._update_train_status(
                    f"<span style='color:#666'>· {len(self._queue)} left in uncertainty queue</span>"
                )
                return
        super()._jump_next_unlabeled()

    def _refresh_queue(self, verbose: bool = False) -> None:
        col = self.df.columns.get_loc(self.labels_column)
        text_col = self.df.columns.get_loc(self.text_column)
        unlabeled = [i for i in range(len(self.df)) if not self.df.iat[i, col]]
        if not unlabeled:
            self._queue = []
            if verbose:
                self._update_train_status("<span style='color:#888'>no unlabeled rows</span>")
            return

        texts = [str(self.df.iat[i, text_col]) for i in unlabeled]
        embs = np.asarray(self.embed_model.encode(texts, normalize_embeddings=True))
        for idx, emb in zip(unlabeled, embs):
            self._text_emb_cache[idx] = emb

        sims = embs @ self.label_embeddings.T
        thr = float(self.threshold_input.value)
        strategy = self.strategy_input.value

        if strategy == "margin":
            score = -np.min(np.abs(sims - thr), axis=1)
        elif strategy == "least_confidence":
            score = -np.max(sims, axis=1)
        elif strategy == "random":
            rng = np.random.default_rng(self._random_state)
            score = rng.random(len(unlabeled))
        else:
            score = np.zeros(len(unlabeled))

        order = np.argsort(-score)
        self._queue = [unlabeled[i] for i in order]

        if verbose:
            self._update_train_status(
                f"<span style='color:#666'>queue refreshed ({strategy}): "
                f"{len(self._queue)} rows</span>"
            )

    def _build_text_label_pairs(self) -> list[tuple[str, list[str]]]:
        col = self.df.columns.get_loc(self.labels_column)
        text_col = self.df.columns.get_loc(self.text_column)
        pairs = []
        for i in range(len(self.df)):
            labs = list(self.df.iat[i, col])
            if labs:
                pairs.append((str(self.df.iat[i, text_col]), labs))
        return pairs

    def _build_eval_text_label_pairs(self) -> list[tuple[str, list[str]]]:
        """Build (text, [label_key]) tuples for the held-out eval_df."""
        if self._eval_df is None or self._eval_truth is None:
            return []
        keys = list(self.model.label_keys)
        pairs = []
        for r, text in enumerate(self._eval_texts):
            labs = [keys[c] for c in range(len(keys)) if self._eval_truth[r, c]]
            if labs:
                pairs.append((str(text), labs))
        return pairs

    def _mlflow_run_ctx(self):
        if self._mlflow_experiment is None:
            return None
        try:
            import mlflow

            return mlflow.start_run(
                run_name=f"al_round_{len(self._train_history) + 1}"
            )
        except Exception as e:
            print(f"[ActiveLearner] mlflow.start_run failed: {e}")
            return None

    def _train(self) -> None:
        pairs = self._build_text_label_pairs()
        n_pairs = sum(len(labs) for _, labs in pairs)
        if n_pairs < 2:
            self.train_status.value = (
                "<span style='color:#c00'>need at least 2 labeled (text, label) pairs to train</span>"
            )
            return

        hp_values = self._collect_model_hp_values()
        epochs = int(hp_values.get("epochs", self._epochs))
        thr = float(self.threshold_input.value)
        # Batch size is owned by the model — passing None falls through to
        # the model's own default (typically "use the full training set").
        batch_size: int | None = None
        if "batch_size" in hp_values:
            batch_size = int(hp_values["batch_size"])

        self.train_status.value = (
            f"<span style='color:#666'>training: {n_pairs} pairs, "
            f"{epochs} epoch(s)...</span>"
        )

        run_ctx = self._mlflow_run_ctx()
        active_run = None
        if run_ctx is not None:
            try:
                active_run = run_ctx.__enter__()
                import mlflow

                mlflow.log_params({
                    "round": len(self._train_history) + 1,
                    "retrain_every": int(self.retrain_input.value),
                    "threshold": thr,
                    "query_strategy": self.strategy_input.value,
                    "n_labeled_rows": self._count_labels(),
                    "n_train_pairs": n_pairs,
                    "model_class": type(self.model).__name__,
                    **{f"hp/{k}": v for k, v in hp_values.items()},
                })
            except Exception as e:
                print(f"[ActiveLearner] mlflow.log_params failed: {e}")

        eval_pairs = self._build_eval_text_label_pairs()

        # ``epochs`` and ``batch_size`` come from hp_values; pass the rest of
        # the model's declared hyperparameters as kwargs untouched.
        extra_kwargs = {
            k: v for k, v in hp_values.items()
            if k not in {"epochs", "batch_size"}
        }

        try:
            metrics = self.model.train(
                pairs,
                epochs=epochs,
                batch_size=batch_size,
                progress_callback=self._train_progress_callback,
                eval_text_label_pairs=eval_pairs,
                **extra_kwargs,
            )
        except Exception as e:
            self.train_status.value = f"<span style='color:#c00'>training failed: {e}</span>"
            if run_ctx is not None:
                try:
                    run_ctx.__exit__(None, None, None)
                except Exception:
                    pass
            return

        mean_loss = float(metrics.get("mean_loss", float("nan")))
        trainable_pct = float(metrics.get("trainable_pct", float("nan")))
        trainable_params = int(metrics.get("trainable_params", 0))
        print(
            f"[ActiveLearner] trained {trainable_params:,} params "
            f"({trainable_pct:.1f}% of total), mean_loss={mean_loss:.4f}"
        )

        self._text_emb_cache.clear()
        descriptions = [f"{lab}. {desc}" for _, lab, desc in self.flat_labels]
        self.label_embeddings = np.asarray(
            self.embed_model.encode(descriptions, normalize_embeddings=True)
        )

        self._labels_at_last_train = self._count_labels()
        self._train_history.append(
            {
                "round": len(self._train_history) + 1,
                "n_pairs": n_pairs,
                "epochs": epochs,
                "labeled_total": self._labels_at_last_train,
                "mean_loss": mean_loss,
                "macro_f1": None,
                "micro_f1": None,
                "mlflow_run_id": getattr(getattr(active_run, "info", None), "run_id", None),
            }
        )

        self._refresh_queue()
        self._render()
        try:
            self._evaluate_and_plot()
        finally:
            if run_ctx is not None:
                try:
                    run_ctx.__exit__(None, None, None)
                except Exception:
                    pass

    def _gather_train_eval(self, thr: float) -> dict | None:
        col = self.df.columns.get_loc(self.labels_column)
        text_col = self.df.columns.get_loc(self.text_column)
        labeled = [i for i in range(len(self.df)) if self.df.iat[i, col]]
        if not labeled:
            return None

        all_keys = list(self.model.label_keys)
        texts = [str(self.df.iat[i, text_col]) for i in labeled]
        truth = np.zeros((len(labeled), len(all_keys)), dtype=bool)
        for r, i in enumerate(labeled):
            true_set = set(self.df.iat[i, col])
            for c, key in enumerate(all_keys):
                truth[r, c] = key in true_set
        return self.model.evaluate(texts, truth, threshold=thr, log_to_mlflow=False)

    def _gather_test_eval(self, thr: float) -> dict | None:
        if self._eval_df is None or len(self._eval_texts) == 0:
            return None
        return self.model.evaluate(
            self._eval_texts,
            self._eval_truth,
            threshold=thr,
            log_to_mlflow=True,
        )

    @staticmethod
    def _confusion_counts(preds: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, ...]:
        tp = (preds & truth).sum(axis=0).astype(int)
        fp = (preds & ~truth).sum(axis=0).astype(int)
        tn = (~preds & ~truth).sum(axis=0).astype(int)
        fn = (~preds & truth).sum(axis=0).astype(int)
        return tp, fp, tn, fn

    @staticmethod
    def _binary_confusion_matrix(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> np.ndarray:
        """Return a 2x2 binary confusion matrix.

        Layout:

            [[TN, FP],
             [FN, TP]]

        Rows are actual values. Columns are predicted values.
        Each returned matrix sums to the number of evaluated rows.
        """
        actual = np.asarray(actual, dtype=bool)
        predicted = np.asarray(predicted, dtype=bool)

        tn = int((~actual & ~predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        tp = int((actual & predicted).sum())

        return np.array([[tn, fp], [fn, tp]], dtype=int)

    def _category_binary_cms(
        self,
        preds: np.ndarray,
        truth: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build one binary confusion matrix per category.

        A category is true for a row if any label in that category is true.
        A category is predicted for a row if any label in that category is predicted.

        Each category matrix sums to n_eval.
        """
        out: dict[str, np.ndarray] = {}

        for cat, idxs in self.cat_to_flat_idx.items():
            idxs = np.asarray(idxs, dtype=int)
            actual_cat = truth[:, idxs].any(axis=1)
            predicted_cat = preds[:, idxs].any(axis=1)
            out[cat] = self._binary_confusion_matrix(
                actual=actual_cat,
                predicted=predicted_cat,
            )

        return out

    def _label_binary_cms(
        self,
        preds: np.ndarray,
        truth: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build one binary confusion matrix per label.

        Each label matrix sums to n_eval.
        """
        out: dict[str, np.ndarray] = {}

        for i, key in enumerate(self.model.label_keys):
            out[key] = self._binary_confusion_matrix(
                actual=truth[:, i],
                predicted=preds[:, i],
            )

        return out

    @staticmethod
    def _empty_panel(ax, msg: str) -> None:
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#888",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_binary_cm_group(
        self,
        fig,
        spec,
        cms: dict[str, np.ndarray],
        title: str,
        max_cols: int = 4,
    ) -> None:
        """Plot a group of 2x2 binary confusion matrices."""
        import math

        names = list(cms.keys())
        n = len(names)

        if n == 0:
            ax = fig.add_subplot(spec)
            self._empty_panel(ax, "no data")
            ax.set_title(title, fontsize=11, fontweight="bold")
            return

        cols = min(max_cols, max(1, n))
        rows = max(1, math.ceil(n / cols))

        sub = spec.subgridspec(
            rows,
            cols,
            wspace=0.45,
            hspace=0.85,
        )

        axes = [
            fig.add_subplot(sub[r, c])
            for r in range(rows)
            for c in range(cols)
        ]

        vmax = max(int(cm.max()) for cm in cms.values()) or 1

        for ax, name in zip(axes, names):
            cm = cms[name]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)

            ax.set_title(name, fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
            ax.set_yticklabels(["Actual 0", "Actual 1"], fontsize=7)
            ax.set_xlabel(f"sum={int(cm.sum())}", fontsize=7)

            for i in range(2):
                for j in range(2):
                    value = int(cm[i, j])
                    color = "white" if value > vmax * 0.5 else "#222"
                    ax.text(
                        j,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )

        for ax in axes[n:]:
            ax.axis("off")

        axes[0].text(
            0.0,
            1.55,
            title,
            transform=axes[0].transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )

    def _render_eval(
        self,
        plot_out,
        metrics_html,
        snapshot: dict,
        source_label: str = "Current",
    ) -> dict | None:
        """Render evaluation results.

        Shows:
        - one binary confusion matrix per category
        - one binary confusion matrix per label

        Every individual 2x2 matrix sums to n_eval.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            self.train_status.value = (
                f"<span style='color:#c00'>matplotlib required: {e}</span>"
            )
            return None

        import math

        thr = float(self.threshold_input.value)

        train_block = snapshot.get("train")
        test_block = snapshot.get("test")

        if train_block is None and test_block is None:
            with plot_out:
                plot_out.clear_output(wait=True)
                print(f"({source_label}) nothing to evaluate")
            metrics_html.value = ""
            return None

        def _build(block):
            if block is None:
                return None

            m = self._metrics_from_sims(
                block["sims"],
                block["truth"],
                thr,
            )
            rs = self._compute_row_scores(
                m["preds"],
                block["truth"],
            )

            return {
                **m,
                "truth": block["truth"],
                "n_eval": block["n_eval"],
                "row_scores": rs,
            }

        train_result = _build(train_block)
        test_result = _build(test_block)

        n_cats = len(self.cat_to_flat_idx)
        n_labs = len(self.model.label_keys)

        cat_cols = min(4, max(1, n_cats))
        lab_cols = min(4, max(1, n_labs))

        cat_rows = max(1, math.ceil(max(1, n_cats) / cat_cols))
        lab_rows = max(1, math.ceil(max(1, n_labs) / lab_cols))

        section_heights = [
            max(1, cat_rows),
            max(1, lab_rows),
            max(1, cat_rows),
            max(1, lab_rows),
        ]

        fig_height = 3.0 + 2.15 * sum(section_heights)

        plot_out.clear_output(wait=True)

        with plot_out:
            fig = plt.figure(figsize=(16, fig_height))

            outer = fig.add_gridspec(
                4,
                1,
                height_ratios=section_heights,
                hspace=0.9,
            )

            def render_section(spec, result, source: str, level: str) -> None:
                if result is None:
                    ax = fig.add_subplot(spec)
                    self._empty_panel(ax, f"no {source} data")
                    ax.set_title(
                        f"{source_label} · {source} · {level}",
                        fontsize=11,
                        fontweight="bold",
                    )
                    return

                if level == "categories":
                    cms = self._category_binary_cms(
                        preds=result["preds"],
                        truth=result["truth"],
                    )
                    title = (
                        f"{source_label} · {source} · per-category binary confusion matrices "
                        f"(thr={thr:.2f}, n={result['n_eval']})"
                    )
                elif level == "labels":
                    cms = self._label_binary_cms(
                        preds=result["preds"],
                        truth=result["truth"],
                    )
                    title = (
                        f"{source_label} · {source} · per-label binary confusion matrices "
                        f"(thr={thr:.2f}, n={result['n_eval']})"
                    )
                else:
                    raise ValueError(f"unknown level: {level}")

                self._plot_binary_cm_group(
                    fig=fig,
                    spec=spec,
                    cms=cms,
                    title=title,
                    max_cols=4,
                )

            render_section(outer[0], train_result, "Train", "categories")
            render_section(outer[1], train_result, "Train", "labels")
            render_section(outer[2], test_result, "Test", "categories")
            render_section(outer[3], test_result, "Test", "labels")

            fig.tight_layout()
            plt.show()
            plt.close(fig)

        metrics_html.value = self._render_metrics_html(
            train_result,
            test_result,
            thr,
        )

        return test_result or train_result

    def _evaluate_and_plot(self) -> None:
        thr = float(self.threshold_input.value)
        snapshot = self._compute_eval_snapshot(thr)
        primary = self._render_eval(
            self.current_plot_out,
            self.current_metrics_html,
            snapshot,
            source_label="Current",
        )
        if primary is None:
            self.train_status.value = (
                "<span style='color:#c00'>"
                "nothing to evaluate: no labeled rows and no eval_df"
                "</span>"
            )
            return

        if self._train_history:
            self._train_history[-1]["macro_f1"] = primary["macro_f1"]
            self._train_history[-1]["micro_f1"] = primary["micro_f1"]

        self._update_train_status(
            f"<span style='color:#2a8a2a'>✓ eval · "
            f"macro-F1={primary['macro_f1']:.3f} · "
            f"micro-F1={primary['micro_f1']:.3f}</span>"
        )

    def _render_metrics_html(
        self,
        train_result: dict | None,
        test_result: dict | None,
        thr: float,
    ) -> str:
        def block(title: str, result: dict | None) -> str:
            if result is None:
                return (
                    f"<div style='margin-right:24px'>"
                    f"<b>{title}</b><br>"
                    f"<span style='color:#888'>not available</span>"
                    f"</div>"
                )
            macro_p = float(np.asarray(result["precision"]).mean())
            macro_r = float(np.asarray(result["recall"]).mean())
            macro_f1 = float(result["macro_f1"])
            micro_p = float(result["micro_p"])
            micro_r = float(result["micro_r"])
            micro_f1 = float(result["micro_f1"])
            rows = "".join(
                f"<tr><td style='padding:1px 10px 1px 0'>{name}</td>"
                f"<td style='text-align:right;padding:1px 8px'>{p:.3f}</td>"
                f"<td style='text-align:right;padding:1px 8px'>{r:.3f}</td>"
                f"<td style='text-align:right;padding:1px 8px'>{f:.3f}</td></tr>"
                for name, p, r, f in [
                    ("macro", macro_p, macro_r, macro_f1),
                    ("micro", micro_p, micro_r, micro_f1),
                ]
            )
            rs = result.get("row_scores") or {}
            n_full = int(rs.get("fully_correct_count", 0))
            n_rows = int(rs.get("n_rows", result["n_eval"]))
            n_cats = int(rs.get("n_categories", 0))
            mean_cat = float(rs.get("mean_cat_correct", 0.0))
            pct = 100.0 * n_full / max(n_rows, 1)
            row_block = (
                f"<div style='margin-top:6px;font-size:12px;color:#444'>"
                f"rows fully correct: <b>{n_full}/{n_rows}</b> "
                f"<span style='color:#888'>({pct:.1f}%)</span><br>"
                f"avg categories correct/row: <b>{mean_cat:.2f}/{n_cats}</b>"
                f"</div>"
            )
            return (
                f"<div style='margin-right:24px'>"
                f"<b>{title}</b> "
                f"<span style='color:#888'>(n={result['n_eval']})</span>"
                f"<table style='border-collapse:collapse;font-size:13px;margin-top:2px'>"
                f"<tr style='color:#888'><td></td>"
                f"<td style='text-align:right;padding:0 8px'>P</td>"
                f"<td style='text-align:right;padding:0 8px'>R</td>"
                f"<td style='text-align:right;padding:0 8px'>F1</td></tr>"
                f"{rows}"
                f"</table>"
                f"{row_block}"
                f"</div>"
            )

        return (
            "<div style='display:flex;align-items:flex-start;margin:6px 0;font-family:sans-serif'>"
            + block("Train", train_result)
            + block("Test", test_result)
            + f"<div style='color:#888;font-size:12px'>threshold={thr:.2f}</div>"
            + "</div>"
        )

    def _save_model(self) -> None:
        if self._model_save_path is None:
            self.train_status.value = (
                "<span style='color:#c00'>no model_save_path was set</span>"
            )
            return
        try:
            self.model.threshold = float(self.threshold_input.value)
            self.model.save_pretrained(self._model_save_path)
            self._update_train_status(
                f"<span style='color:#2a8a2a'>✓ model saved to {self._model_save_path}</span>"
            )
        except Exception as e:
            self.train_status.value = f"<span style='color:#c00'>save failed: {e}</span>"

    def predict(
        self,
        texts: list[str] | pd.Series,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Pass-through to ``self.model.predict``."""
        thr = self._threshold if threshold is None else float(threshold)
        params = {"threshold": thr}
        return self.model.predict(None, texts, params=params)

    def get_row_scores(
        self,
        split: Literal["test", "train"] = "test",
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Per-row correctness scores at the current model state.

        Each row is scored 0..n_categories where the score counts the number
        of categories whose label set was predicted exactly correctly.
        ``fully_correct`` is True iff the score equals ``n_categories``.
        """
        thr = float(self.threshold_input.value) if threshold is None else float(threshold)
        snapshot = self._compute_eval_snapshot(thr)
        block = snapshot.get(split)
        if block is None:
            return pd.DataFrame(
                columns=["text", "score", "fully_correct"]
            )

        m = self._metrics_from_sims(block["sims"], block["truth"], thr)
        rs = self._compute_row_scores(m["preds"], block["truth"])

        if split == "test":
            texts = list(self._eval_texts)
        else:
            col = self.df.columns.get_loc(self.labels_column)
            text_col = self.df.columns.get_loc(self.text_column)
            texts = [
                str(self.df.iat[i, text_col])
                for i in range(len(self.df))
                if self.df.iat[i, col]
            ]

        df = pd.DataFrame({
            "text": texts,
            "score": rs["per_row_score"],
            "fully_correct": rs["fully_correct"],
        })
        for ci, cat in enumerate(rs["categories"]):
            df[f"{cat}_correct"] = rs["cat_correct"][:, ci]
        return df
