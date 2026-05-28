"""XGBoost-based multi-label text matcher on sentence-transformer embeddings.

Architecture:
    text  ─→  frozen sentence-transformer encoder ─→  text_emb
    label ─→  frozen sentence-transformer encoder ─→  label_emb
    (text_emb, label_emb) → optional feature combination → optional PCA → XGBoost
    XGBoost outputs P(text matches label) ∈ [0, 1]; threshold gives the
    boolean prediction.

The class exposes the same surface as :class:`mlflow_model.SentenceTransformerClassifier`
so it can be passed straight into :class:`active_learner.ActiveLearner`.

Install requirements:
    pip install xgboost scikit-learn

Usage:
    from xgboost_model import XGBoostMatcher
    from active_learner import ActiveLearner

    clf = XGBoostMatcher(
        label_dict=labels,
        embed_model_name_or_path="all-MiniLM-L6-v2",
        threshold=0.5,
        use_pca=True,
        pca_components=64,
        feature_mode="concat+diff+prod",
        n_negatives_per_text=3,
    )
    widget = ActiveLearner(model=clf, ...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


LABEL_SEP = "::"

FeatureMode = Literal[
    "concat",
    "diff",
    "prod",
    "concat+diff",
    "concat+prod",
    "diff+prod",
    "concat+diff+prod",
]


class XGBoostMatcher:
    """Binary "does this text match this label?" classifier on top of a frozen
    sentence-transformer encoder.

    Both the text embedding and the label-description embedding are fed
    (optionally after PCA) to an XGBoost classifier whose binary output is
    ``P(match)``. Multi-label inference scores every text against every label.
    """

    def __init__(
        self,
        label_dict: dict[str, dict[str, str]],
        embed_model_name_or_path: str = "all-MiniLM-L6-v2",
        threshold: float = 0.5,
        use_pca: bool = False,
        pca_components: int = 64,
        n_negatives_per_text: int = 3,
        feature_mode: FeatureMode = "concat+diff+prod",
        xgb_params: dict[str, Any] | None = None,
        n_estimators_per_epoch: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        device: str | None = None,
        random_state: int = 42,
        batch_size: int | None = None,
    ):
        self.label_dict = label_dict
        self.embed_model_name_or_path = embed_model_name_or_path
        self.threshold = float(threshold)
        self.learning_rate = float(learning_rate)
        self.use_pca = bool(use_pca)
        self.pca_components = int(pca_components)
        self.n_negatives_per_text = int(n_negatives_per_text)
        self.feature_mode = feature_mode
        self.xgb_params = dict(xgb_params or {})
        self.n_estimators_per_epoch = int(n_estimators_per_epoch)
        self.max_depth = int(max_depth)
        self.device = device
        self.random_state = int(random_state)
        # XGBoost doesn't use mini-batch SGD; this exists for API parity and
        # downstream code that asks how big a single fit's training set is.
        self.batch_size = batch_size
        # Compatibility with the SentenceTransformerClassifier surface; XGBoost
        # has no notion of frozen layers, but ActiveLearner doesn't read this.
        self.n_trainable_transformer_layers = None

        self.flat_labels: list[tuple[str, str, str]] = [
            (cat, lab, desc)
            for cat, labs in label_dict.items()
            for lab, desc in labs.items()
        ]
        self.label_keys: list[str] = [
            f"{cat}{LABEL_SEP}{lab}" for cat, lab, _ in self.flat_labels
        ]
        self._key_to_idx = {k: i for i, k in enumerate(self.label_keys)}

        self._st_model = None
        self._label_embeddings: np.ndarray | None = None
        self._pca = None
        self._xgb = None
        self._n_features = None  # set on first train

    # ── pickle support ────────────────────────────────────────────────────
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_st_model"] = None
        state["_label_embeddings"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ── lazy model loading ────────────────────────────────────────────────
    @property
    def st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._st_model = SentenceTransformer(
                self.embed_model_name_or_path, **kwargs
            )
            self._refresh_label_embeddings()
        return self._st_model

    @property
    def label_embeddings(self) -> np.ndarray:
        if self._label_embeddings is None:
            _ = self.st_model
        return self._label_embeddings

    def _refresh_label_embeddings(self) -> None:
        descs = [f"{lab}. {desc}" for _, lab, desc in self.flat_labels]
        self._label_embeddings = np.asarray(
            self._st_model.encode(descs, normalize_embeddings=True)
        )

    def encode(self, texts, normalize_embeddings: bool = True, **kwargs):
        return self.st_model.encode(
            texts, normalize_embeddings=normalize_embeddings, **kwargs
        )

    def snapshot_state(self) -> dict | None:
        """Save the trained XGBoost + PCA for restoration."""
        return {"xgb": self._xgb, "pca": self._pca}

    def restore_state(self, snapshot) -> None:
        if snapshot is None:
            return
        self._xgb = snapshot.get("xgb")
        self._pca = snapshot.get("pca")

    @classmethod
    def train_hyperparams(cls) -> list[dict[str, Any]]:
        """Declare the train-time hyperparameters this model exposes to a UI.

        Each entry may include ``pso`` (default ``True``) to set whether the
        hyperparameter is included in particle-swarm search by default.
        """
        return [
            {
                "name": "epochs",
                "label": "Epochs",
                "kind": "int",
                "default": 1, "min": 1, "max": 50,
                "description": "Boost rounds = epochs × n_estimators_per_epoch.",
                "pso": False,
            },
            {
                "name": "learning_rate",
                "label": "LR",
                "kind": "log_float",
                "default": 0.1, "min": 1e-3, "max": 1.0,
                "description": "XGBoost learning_rate (shrinkage).",
                "pso": True,
            },
            {
                "name": "max_depth",
                "label": "Depth",
                "kind": "int",
                "default": 6, "min": 1, "max": 12,
                "description": "Maximum tree depth.",
                "pso": True,
            },
            {
                "name": "n_negatives_per_text",
                "label": "Neg/text",
                "kind": "int",
                "default": 3, "min": 0, "max": 20,
                "description": "Negative label-description pairs sampled per text.",
                "pso": False,
            },
            {
                "name": "use_pca",
                "label": "PCA",
                "kind": "bool",
                "default": False,
                "description": "Apply PCA to features before XGBoost.",
                "pso": True,
            },
            {
                "name": "pca_components",
                "label": "PCA k",
                "kind": "int",
                "default": 64, "min": 2, "max": 512,
                "description": "PCA components when PCA is on.",
                "pso": True,
            },
            {
                "name": "feature_mode",
                "label": "Features",
                "kind": "choice",
                "default": "concat+diff+prod",
                "choices": [
                    "concat", "diff", "prod",
                    "concat+diff", "concat+prod", "diff+prod", "concat+diff+prod",
                ],
                "description": "How to combine text and label embeddings.",
                "pso": True,
            },
        ]

    # ── feature engineering ───────────────────────────────────────────────
    def _make_features(
        self, text_emb: np.ndarray, label_emb: np.ndarray
    ) -> np.ndarray:
        """Combine text and label embeddings into one feature vector.

        Supports vectorized broadcasting: ``text_emb`` can be ``(d,)`` or
        ``(n, d)`` and ``label_emb`` ``(d,)`` or ``(n, d)``.
        """
        mode = self.feature_mode
        parts: list[np.ndarray] = []
        if "concat" in mode:
            parts.extend([text_emb, label_emb])
        if "diff" in mode:
            parts.append(text_emb - label_emb)
        if "prod" in mode:
            parts.append(text_emb * label_emb)
        if not parts:
            raise ValueError(f"unknown feature_mode: {mode!r}")
        return np.concatenate(parts, axis=-1)

    def _build_xy(
        self, text_label_pairs: list[tuple[str, list[str]]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode texts and build a flat ``(X, y)`` for XGBoost."""
        import random as _random

        rng = _random.Random(self.random_state)
        texts = [str(t) for t, _ in text_label_pairs]
        if not texts:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        text_embs = np.asarray(
            self.st_model.encode(texts, normalize_embeddings=True)
        )
        label_embs = self.label_embeddings
        all_keys = set(self.label_keys)

        X_rows: list[np.ndarray] = []
        y: list[int] = []
        for ti, (_text, labels) in enumerate(text_label_pairs):
            pos_keys = [k for k in labels if k in all_keys]
            if not pos_keys:
                continue
            t_emb = text_embs[ti]
            for key in pos_keys:
                X_rows.append(self._make_features(t_emb, label_embs[self._key_to_idx[key]]))
                y.append(1)
            negatives = [k for k in self.label_keys if k not in pos_keys]
            rng.shuffle(negatives)
            for key in negatives[: self.n_negatives_per_text]:
                X_rows.append(self._make_features(t_emb, label_embs[self._key_to_idx[key]]))
                y.append(0)

        X = np.asarray(X_rows, dtype=np.float32)
        return X, np.asarray(y, dtype=np.int32)

    # ── training ──────────────────────────────────────────────────────────
    def train(
        self,
        text_label_pairs: list[tuple[str, list[str]]],
        epochs: int = 1,
        batch_size: int | None = None,  # accepted for API parity, mapped below
        learning_rate: float | None = None,
        log_to_mlflow: bool = True,
        progress_callback=None,
        eval_text_label_pairs: list[tuple[str, list[str]]] | None = None,
        n_negatives_per_text: int | None = None,
        random_state: int | None = None,
        max_depth: int | None = None,
        use_pca: bool | None = None,
        pca_components: int | None = None,
        feature_mode: str | None = None,
        **_ignored,
    ) -> dict[str, float]:
        """Fit XGBoost on ``(text, [label_key])`` tuples.

        ``epochs`` is multiplied by ``n_estimators_per_epoch`` to compute the
        total number of boosting rounds, so the widget's epoch knob still
        produces meaningful "more training = more capacity."
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise RuntimeError(
                "xgboost is required for XGBoostMatcher; "
                "install it with `pip install xgboost`"
            ) from e

        try:
            from sklearn.decomposition import PCA
            from sklearn.metrics import log_loss
        except ImportError as e:
            raise RuntimeError(
                "scikit-learn is required for XGBoostMatcher; "
                "install it with `pip install scikit-learn`"
            ) from e

        if n_negatives_per_text is not None:
            self.n_negatives_per_text = int(n_negatives_per_text)
        if random_state is not None:
            self.random_state = int(random_state)
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        if max_depth is not None:
            self.max_depth = int(max_depth)
        if use_pca is not None:
            self.use_pca = bool(use_pca)
        if pca_components is not None:
            self.pca_components = int(pca_components)
        if feature_mode is not None:
            self.feature_mode = feature_mode

        X_train, y_train = self._build_xy(text_label_pairs)
        if eval_text_label_pairs:
            X_eval, y_eval = self._build_xy(eval_text_label_pairs)
        else:
            X_eval, y_eval = None, None

        if len(X_train) < 2 or len(np.unique(y_train)) < 2:
            return {
                "mean_loss": float("nan"),
                "mean_val_loss": float("nan"),
                "n_pairs": float(len(X_train)),
                "n_eval_pairs": float(0 if X_eval is None else len(X_eval)),
                "epochs": 0.0,
                "batch_size": 0.0,
                "lr": float(self.learning_rate),
                "trainable_params": 0.0,
                "total_params": 0.0,
                "trainable_pct": 0.0,
            }

        # Optional PCA on the combined feature vector.
        if self.use_pca:
            n_comp = max(2, min(self.pca_components, X_train.shape[1], X_train.shape[0]))
            self._pca = PCA(n_components=n_comp, random_state=self.random_state)
            X_train_p = self._pca.fit_transform(X_train).astype(np.float32)
            X_eval_p = (
                self._pca.transform(X_eval).astype(np.float32)
                if X_eval is not None
                else None
            )
        else:
            self._pca = None
            X_train_p = X_train
            X_eval_p = X_eval

        self._n_features = int(X_train_p.shape[1])
        batches_per_epoch = self.n_estimators_per_epoch
        n_estimators = max(int(epochs), 1) * batches_per_epoch
        lr = self.learning_rate

        params: dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": self.random_state,
            "max_depth": self.max_depth,
            "learning_rate": lr,
            "n_estimators": n_estimators,
            **self.xgb_params,
        }

        def _emit(event: str, **kwargs):
            if progress_callback is None:
                return
            try:
                progress_callback(event, **kwargs)
            except Exception as e:
                print(f"[xgb-train] callback raised: {e}")

        # Rough "param count" — leaves-per-tree × n_estimators.
        approx_total = int(n_estimators * (2 ** self.max_depth))
        approx_trainable = approx_total  # XGBoost has no notion of freezing.
        _emit(
            "start",
            n_pairs=len(X_train),
            n_eval_pairs=int(0 if X_eval_p is None else len(X_eval_p)),
            epochs=int(epochs),
            batch_size=batches_per_epoch,
            batches_per_epoch=batches_per_epoch,
            trainable_params=approx_trainable,
            total_params=approx_total,
            lr=lr,
        )

        train_loss_curve: list[float] = []
        val_loss_curve: list[float] = []

        class _Progress(xgb.callback.TrainingCallback):
            def __init__(cb_self):
                cb_self._last_epoch = -1

            def after_iteration(cb_self, model, epoch, evals_log):
                cur_epoch = epoch // batches_per_epoch + 1
                cur_batch = epoch % batches_per_epoch + 1
                if cb_self._last_epoch != cur_epoch and cur_batch == 1:
                    _emit("epoch_start", epoch=cur_epoch, n_epochs=int(epochs))
                    cb_self._last_epoch = cur_epoch

                train_curve = (
                    evals_log.get("validation_0", {}).get("logloss")
                    or evals_log.get("train", {}).get("logloss")
                    or []
                )
                eval_curve = (
                    evals_log.get("validation_1", {}).get("logloss")
                    or evals_log.get("eval", {}).get("logloss")
                    or []
                )
                tloss = float(train_curve[-1]) if train_curve else float("nan")
                vloss = float(eval_curve[-1]) if eval_curve else float("nan")
                train_loss_curve.append(tloss)
                val_loss_curve.append(vloss)

                _emit(
                    "batch",
                    epoch=cur_epoch,
                    batch=cur_batch,
                    n_batches=batches_per_epoch,
                    loss=tloss,
                )
                if cur_batch == batches_per_epoch:
                    _emit(
                        "epoch_end",
                        epoch=cur_epoch,
                        n_epochs=int(epochs),
                        mean_loss=tloss,
                        mean_val_loss=vloss,
                    )
                return False

        self._xgb = xgb.XGBClassifier(**params, callbacks=[_Progress()])
        eval_set: list[tuple[np.ndarray, np.ndarray]] = [(X_train_p, y_train)]
        if X_eval_p is not None:
            eval_set.append((X_eval_p, y_eval))

        self._xgb.fit(
            X_train_p,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        train_proba = self._xgb.predict_proba(X_train_p)[:, 1]
        mean_loss = float(log_loss(y_train, train_proba, labels=[0, 1]))
        if X_eval_p is not None:
            val_proba = self._xgb.predict_proba(X_eval_p)[:, 1]
            mean_val_loss = float(log_loss(y_eval, val_proba, labels=[0, 1]))
        else:
            mean_val_loss = float("nan")

        _emit(
            "end",
            n_batches=n_estimators,
            mean_loss=mean_loss,
            mean_val_loss=mean_val_loss,
        )

        metrics = {
            "mean_loss": mean_loss,
            "mean_val_loss": mean_val_loss,
            "n_pairs": float(len(X_train)),
            "n_eval_pairs": float(0 if X_eval_p is None else len(X_eval_p)),
            "epochs": float(epochs),
            "batch_size": float(batches_per_epoch),
            "lr": float(lr),
            "n_estimators": float(n_estimators),
            "use_pca": float(int(self.use_pca)),
            "pca_components": float(
                self._pca.n_components_ if self._pca is not None else 0
            ),
            "trainable_params": float(approx_trainable),
            "total_params": float(approx_total),
            "trainable_pct": 100.0,
        }

        if log_to_mlflow:
            self._safe_log_metrics({
                "train_mean_loss": mean_loss,
                "val_mean_loss": mean_val_loss,
                "n_pairs": float(len(X_train)),
                "n_eval_pairs": metrics["n_eval_pairs"],
                "n_estimators": metrics["n_estimators"],
            })
        return metrics

    # ── inference ─────────────────────────────────────────────────────────
    def predict_scores(self, texts: list[str]) -> np.ndarray:
        """``(n_texts, n_labels)`` matrix of match probabilities.

        Before training, falls back to cosine similarity between text and
        label-description embeddings so the widget renders reasonable
        zero-shot scores.
        """
        text_embs = np.asarray(
            self.st_model.encode(texts, normalize_embeddings=True)
        )
        if self._xgb is None:
            return text_embs @ self.label_embeddings.T

        n_texts = len(texts)
        n_labels = len(self.label_keys)
        d = text_embs.shape[1]
        # Vectorized feature construction: tile and repeat to get all-pairs.
        tiled_text = np.repeat(text_embs, n_labels, axis=0)              # (n_t*n_l, d)
        tiled_label = np.tile(self.label_embeddings, (n_texts, 1))       # (n_t*n_l, d)
        X = self._make_features(tiled_text, tiled_label)
        if self._pca is not None:
            X = self._pca.transform(X).astype(np.float32)
        else:
            X = X.astype(np.float32)
        probs = self._xgb.predict_proba(X)[:, 1]
        return probs.reshape(n_texts, n_labels)

    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        elif isinstance(model_input, pd.Series):
            texts = model_input.astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        thr = self.threshold
        if params and "threshold" in params:
            thr = float(params["threshold"])
        sims = self.predict_scores(texts)
        preds = sims >= thr
        out = pd.DataFrame(sims, columns=[f"sim::{k}" for k in self.label_keys])
        for c, k in enumerate(self.label_keys):
            out[k] = preds[:, c]
        return out

    def evaluate(
        self,
        texts: list[str],
        truth: np.ndarray,
        threshold: float | None = None,
        log_to_mlflow: bool = True,
    ) -> dict[str, Any]:
        thr = self.threshold if threshold is None else float(threshold)
        sims = self.predict_scores(texts)
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

        result = {
            "sims": sims,
            "preds": preds,
            "truth": truth,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "micro_p": float(micro_p),
            "micro_r": float(micro_r),
            "threshold": thr,
            "n_eval": len(texts),
            "label_keys": list(self.label_keys),
        }
        if log_to_mlflow:
            self._safe_log_metrics({
                "eval_macro_f1": macro_f1,
                "eval_micro_f1": micro_f1,
                "eval_micro_p": result["micro_p"],
                "eval_micro_r": result["micro_r"],
            })
        return result

    # ── persistence ───────────────────────────────────────────────────────
    def save_pretrained(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        st_path = path / "st_model"
        self.st_model.save(str(st_path))
        if self._xgb is not None:
            self._xgb.save_model(str(path / "xgb.json"))
        if self._pca is not None:
            import joblib
            joblib.dump(self._pca, str(path / "pca.joblib"))
        config = {
            "label_dict": self.label_dict,
            "threshold": self.threshold,
            "learning_rate": self.learning_rate,
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
            "n_negatives_per_text": self.n_negatives_per_text,
            "feature_mode": self.feature_mode,
            "xgb_params": self.xgb_params,
            "n_estimators_per_epoch": self.n_estimators_per_epoch,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "device": self.device,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))
        return path

    @classmethod
    def load_pretrained(cls, path: str | Path) -> "XGBoostMatcher":
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        clf = cls(
            label_dict=config["label_dict"],
            embed_model_name_or_path=str(path / "st_model"),
            threshold=config["threshold"],
            learning_rate=config["learning_rate"],
            use_pca=config["use_pca"],
            pca_components=config["pca_components"],
            n_negatives_per_text=config["n_negatives_per_text"],
            feature_mode=config["feature_mode"],
            xgb_params=config.get("xgb_params") or {},
            n_estimators_per_epoch=config["n_estimators_per_epoch"],
            max_depth=config["max_depth"],
            random_state=config["random_state"],
            device=config.get("device"),
        )
        _ = clf.st_model
        xgb_path = path / "xgb.json"
        if xgb_path.exists():
            import xgboost as xgb
            booster = xgb.XGBClassifier()
            booster.load_model(str(xgb_path))
            clf._xgb = booster
        pca_path = path / "pca.joblib"
        if pca_path.exists():
            import joblib
            clf._pca = joblib.load(str(pca_path))
        return clf

    @staticmethod
    def _safe_log_metrics(metrics: dict[str, float]) -> None:
        try:
            import mlflow
            if mlflow.active_run() is not None:
                mlflow.log_metrics(metrics)
        except Exception:
            pass
