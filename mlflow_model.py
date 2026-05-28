
import json
from pathlib import Path 
from typing import Any

import numpy as np
import pandas as pd

LABEL_SEP = "::"

try:
    import mlflow.pyfunc as _mlflow_pyfunc
    _PyfuncBase = _mlflow_pyfunc.PythonModel
except Exception:  # mlflow not installed → fall back to plain object
    _PyfuncBase = object

class SentenceTransformerClassifier(_PyfuncBase):
    """Multi-label classifier built on a sentence-transformer encoder.

    Label predictions are cosine similarities between the text embedding and
    each ``"<label_name>. <description>"`` embedding; a similarity at or above
    ``threshold`` counts as positive for that label.
    """

    def __init__(
        self,
        label_dict: dict[str, dict[str, str]],
        model_name_or_path: str = "all-MiniLM-L6-v2",
        threshold: float = 0.5,
        learning_rate: float = 2e-5,
        device: str | None = None,
        n_trainable_transformer_layers: int | None = 1,
        batch_size: int | None = None,
    ):
        """Multi-label classifier wrapping a sentence-transformer encoder.

        Args:
            n_trainable_transformer_layers: how many encoder layers (counting
                from the top of the transformer stack) remain trainable
                during ``train()``. ``None`` = train every parameter
                (full fine-tune). ``N > 0`` = train the last N transformer
                layers plus all head modules. ``0`` = freeze the transformer
                entirely and only train head modules — only useful when the
                head contains trainable params (e.g. a Dense layer); most
                stock encoders like ``all-MiniLM-L6-v2`` only have a
                parameterless pooling head, so ``0`` would leave nothing
                to optimize and ``train()`` will raise.
        """
        self.label_dict = label_dict
        self.model_name_or_path = model_name_or_path
        self.threshold = float(threshold)
        self.learning_rate = float(learning_rate)
        self.device = device
        self.n_trainable_transformer_layers = n_trainable_transformer_layers
        self.batch_size = batch_size  # None ⇒ use one max-size batch

        self.flat_labels: list[tuple[str, str, str]] = [
            (cat, lab, desc)
            for cat, labs in label_dict.items()
            for lab, desc in labs.items()
        ]
        self.label_keys: list[str] = [
            f"{cat}{LABEL_SEP}{lab}" for cat, lab, _ in self.flat_labels
        ]

        self._st_model = None
        self._label_embeddings: np.ndarray | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_st_model"] = None
        state["_label_embeddings"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    @property
    def st_model(self):
        if self._st_model is None:
            self._load_st(self.model_name_or_path)
        return self._st_model

    @property
    def label_embeddings(self) -> np.ndarray:
        if self._label_embeddings is None:
            _ = self.st_model
        return self._label_embeddings

    def _load_st(self, path_or_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        self._st_model = SentenceTransformer(path_or_name, **kwargs)
        self._refresh_label_embeddings()

    def _refresh_label_embeddings(self) -> None:
        descs = [f"{lab}. {desc}" for _, lab, desc in self.flat_labels]
        self._label_embeddings = np.asarray(
            self._st_model.encode(descs, normalize_embeddings=True)
        )

    @staticmethod
    def _get_last_transformer_layers(hf_model) -> list:
        """Return transformer block layers for common Hugging Face layouts.

        Supports common encoder-only models like MiniLM/BERT/MPNet and
        decoder-style models like Qwen/Llama/Mistral.

        Returns an empty list when no known layer container is found.
        """
        if hf_model is None:
            return []

        # BERT / MiniLM / MPNet style:
        # AutoModel(...).encoder.layer
        encoder = getattr(hf_model, "encoder", None)
        layers = getattr(encoder, "layer", None) if encoder is not None else None
        if layers is not None:
            return list(layers)

        # Some encoder models expose encoder.layers.
        layers = getattr(encoder, "layers", None) if encoder is not None else None
        if layers is not None:
            return list(layers)

        # Qwen / Llama / Mistral style:
        # AutoModel(...).model.layers
        inner_model = getattr(hf_model, "model", None)
        layers = getattr(inner_model, "layers", None) if inner_model is not None else None
        if layers is not None:
            return list(layers)

        # Some architectures expose layers directly on the model.
        layers = getattr(hf_model, "layers", None)
        if layers is not None:
            return list(layers)

        # T5-style models.
        encoder = getattr(hf_model, "encoder", None)
        block = getattr(encoder, "block", None) if encoder is not None else None
        if block is not None:
            return list(block)

        return []

    def _apply_layer_freeze(self) -> tuple[int, int]:
        """Apply the freeze policy from ``n_trainable_transformer_layers``.

        Returns ``(trainable_params, total_params)``.

        ``None`` trains every parameter. For integer values, this method freezes
        everything first, then re-enables all non-transformer
        SentenceTransformer modules plus the last N Hugging Face transformer
        layers when they can be located.

        This version supports both encoder-style layer paths
        (``hf_model.encoder.layer``) and Qwen/Llama-style layer paths
        (``hf_model.model.layers``). If N > 0 and no layer container is found,
        it falls back to full fine-tuning rather than leaving zero trainable
        parameters.
        """
        model = self.st_model
        n = self.n_trainable_transformer_layers

        total = sum(p.numel() for p in model.parameters())

        # Full fine-tune.
        if n is None:
            for p in model.parameters():
                p.requires_grad = True

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"[SentenceTransformerClassifier] full fine-tune: "
                f"{trainable:,}/{total:,} trainable params"
            )

            if trainable == 0:
                raise RuntimeError(
                    "model has zero trainable parameters even with "
                    "n_trainable_transformer_layers=None"
                )

            return trainable, total

        # Freeze everything first.
        for p in model.parameters():
            p.requires_grad = False

        # Keep non-transformer SentenceTransformer modules trainable when they
        # have parameters, for example Dense/Normalize heads. Pooling modules
        # usually have no parameters.
        for module_idx in range(1, len(model)):
            for p in model[module_idx].parameters():
                p.requires_grad = True

        # Unfreeze last N transformer layers.
        if int(n) > 0:
            transformer = model[0]
            hf_model = getattr(transformer, "auto_model", None)
            candidate_layers = self._get_last_transformer_layers(hf_model)

            if candidate_layers:
                for layer in candidate_layers[-int(n):]:
                    for p in layer.parameters():
                        p.requires_grad = True
            else:
                print(
                    "[SentenceTransformerClassifier] warning: could not locate "
                    "transformer layers for this architecture; falling back to "
                    "full fine-tune. You can also pass "
                    "n_trainable_transformer_layers=None explicitly."
                )
                for p in model.parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(
            f"[SentenceTransformerClassifier] freeze policy n={n}: "
            f"{trainable:,}/{total:,} trainable params"
        )

        if trainable == 0:
            raise RuntimeError(
                "no trainable parameters after applying freeze policy. "
                "Use n_trainable_transformer_layers=None, restart the kernel, "
                "and recreate SentenceTransformerClassifier."
            )

        return trainable, total

    def encode(self, texts, normalize_embeddings: bool = True, **kwargs):
        return self.st_model.encode(
            texts, normalize_embeddings=normalize_embeddings, **kwargs
        )

    def snapshot_state(self) -> dict | None:
        """Save trainable state for later restoration (used by HP tuning).

        Returns a CPU-tensor copy of the model's ``state_dict`` keyed by
        parameter name. ``restore_state`` reverses it.
        """
        if self._st_model is None:
            return None
        return {
            k: v.detach().cpu().clone()
            for k, v in self._st_model.state_dict().items()
        }

    def restore_state(self, snapshot) -> None:
        if snapshot is None or self._st_model is None:
            return
        try:
            device = next(self._st_model.parameters()).device
        except StopIteration:
            device = None
        state = {
            k: (v.to(device) if device is not None else v)
            for k, v in snapshot.items()
        }
        self._st_model.load_state_dict(state)
        self._refresh_label_embeddings()

    @classmethod
    def train_hyperparams(cls) -> list[dict[str, Any]]:
        """Declare the hyperparameters this model exposes to a UI.

        Each entry is a dict with ``name`` (must be a kwarg of ``train()``),
        ``label`` (display text), ``kind`` (``int|float|log_float|bool|choice``),
        ``default``, plus optional ``min``, ``max``, ``step``, ``choices``,
        ``description``, and ``pso`` (whether the HP is included in
        particle-swarm search by default).
        """
        return [
            {
                "name": "epochs",
                "label": "Epochs",
                "kind": "int",
                "default": 1, "min": 1, "max": 100,
                "description": "Passes over the (text, label) pair set.",
                "pso": False,
            },
            {
                "name": "learning_rate",
                "label": "LR",
                "kind": "log_float",
                "default": 2e-5, "min": 1e-7, "max": 1e-2,
                "description": "AdamW learning rate.",
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
        ]

    def train(
        self,
        text_label_pairs: list[tuple[str, list[str]]],
        epochs: int = 1,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        log_to_mlflow: bool = True,
        progress_callback=None,
        eval_text_label_pairs: list[tuple[str, list[str]]] | None = None,
        n_negatives_per_text: int = 3,
        random_state: int = 42,
    ) -> dict[str, float]:
        """Fine-tune on ``(text, [label_key])`` tuples.

        Uses :class:`losses.CosineSimilarityLoss`: each ``(text, "label. desc")``
        pair gets target ``1.0`` (positive) or ``0.0`` (sampled negative). This
        avoids the in-batch conflicts that ``MultipleNegativesRankingLoss``
        hits on multi-label data with shared label descriptions.

        If ``eval_text_label_pairs`` is given, the same loss is computed on
        that set after every epoch (no gradients) and reported as
        ``mean_val_loss`` in the ``epoch_end`` callback.

        Returns a metrics dict with ``mean_loss``, ``mean_val_loss`` (or
        ``NaN``), ``n_pairs``, ``n_eval_pairs``, ``epochs``, ``batch_size``,
        ``lr``, plus the trainable-param info.
        """
        import random as _random
        import torch
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader

        lr = self.learning_rate if learning_rate is None else float(learning_rate)
        key_to_desc = {
            f"{cat}{LABEL_SEP}{lab}": f"{lab}. {desc}"
            for cat, lab, desc in self.flat_labels
        }
        all_keys = set(key_to_desc.keys())
        rng = _random.Random(random_state)

        def _build_examples(
            pairs: list[tuple[str, list[str]]],
        ) -> list[InputExample]:
            built: list[InputExample] = []
            for text, labels in pairs:
                pos_keys = [k for k in labels if k in key_to_desc]
                if not pos_keys:
                    continue
                text = str(text)
                for key in pos_keys:
                    built.append(InputExample(
                        texts=[text, key_to_desc[key]], label=1.0,
                    ))
                negatives = [k for k in all_keys if k not in pos_keys]
                rng.shuffle(negatives)
                for key in negatives[:int(n_negatives_per_text)]:
                    built.append(InputExample(
                        texts=[text, key_to_desc[key]], label=0.0,
                    ))
            return built

        examples = _build_examples(text_label_pairs)
        eval_examples = (
            _build_examples(eval_text_label_pairs)
            if eval_text_label_pairs else []
        )

        if len(examples) < 2:
            return {
                "mean_loss": float("nan"),
                "mean_val_loss": float("nan"),
                "n_pairs": float(len(examples)),
                "n_eval_pairs": float(len(eval_examples)),
                "epochs": 0.0,
                "batch_size": 0.0,
                "lr": lr,
            }

        # Batch size: kwarg overrides instance attribute; falling all the way
        # through ``None`` means "use the entire training set as one batch".
        bs_raw = batch_size
        if bs_raw is None:
            bs_raw = self.batch_size
        if bs_raw is None:
            bs_raw = len(examples)
        bs = max(2, min(int(bs_raw), len(examples)))
        model = self.st_model
        trainable_params, total_params = self._apply_layer_freeze()
        loader = DataLoader(
            examples,
            shuffle=True,
            batch_size=bs,
            collate_fn=model.smart_batching_collate,
        )
        eval_loader = (
            DataLoader(
                eval_examples,
                shuffle=False,
                batch_size=bs,
                collate_fn=model.smart_batching_collate,
            )
            if eval_examples else None
        )
        loss_fn = losses.CosineSimilarityLoss(model)
        device = getattr(model, "device", None) or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        loss_fn.to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "no trainable parameters; check n_trainable_transformer_layers"
            )
        optimizer = torch.optim.AdamW(trainable, lr=lr)

        def _to_device(features_batch):
            """Move features to ``device``, skipping non-tensor metadata.

            Newer sentence-transformers versions and some architectures (Qwen3,
            Llama-based, etc.) include string metadata alongside tensors. We
            only ``.to(device)`` actual tensors and pass everything else
            through untouched.
            """
            moved = []
            for feat in features_batch:
                d = {}
                for k, v in feat.items():
                    if isinstance(v, torch.Tensor):
                        d[k] = v.to(device)
                    else:
                        d[k] = v
                moved.append(d)
            return moved

        def _compute_eval_loss() -> float:
            if eval_loader is None:
                return float("nan")
            model.eval()
            total = 0.0
            n_b = 0
            with torch.no_grad():
                for feats, lbls in eval_loader:
                    feats = _to_device(feats)
                    lbls = lbls.to(device)
                    loss_v = loss_fn(feats, lbls)
                    total += float(loss_v.detach().cpu())
                    n_b += 1
            model.train()
            return total / max(n_b, 1)

        total_loss = 0.0
        n_batches = 0
        batches_per_epoch = len(loader)
        last_val_loss = float("nan")

        def _emit(event, **kwargs):
            if progress_callback is None:
                return
            try:
                progress_callback(event, **kwargs)
            except Exception as cb_err:
                print(f"[train] progress_callback raised: {cb_err}")

        _emit(
            "start",
            n_pairs=len(examples),
            n_eval_pairs=len(eval_examples),
            epochs=int(epochs),
            batch_size=bs,
            batches_per_epoch=batches_per_epoch,
            trainable_params=trainable_params,
            total_params=total_params,
            lr=lr,
        )

        try:
            model.train()
            for epoch in range(int(epochs)):
                _emit("epoch_start", epoch=epoch + 1, n_epochs=int(epochs))
                ep_loss = 0.0
                ep_batches = 0
                for batch_i, (features, labels_tensor) in enumerate(loader):
                    features = _to_device(features)
                    labels_tensor = labels_tensor.to(device)
                    optimizer.zero_grad()
                    loss_val = loss_fn(features, labels_tensor)
                    loss_val.backward()
                    optimizer.step()
                    batch_loss = float(loss_val.detach().cpu())
                    total_loss += batch_loss
                    ep_loss += batch_loss
                    n_batches += 1
                    ep_batches += 1
                    _emit(
                        "batch",
                        epoch=epoch + 1,
                        batch=batch_i + 1,
                        n_batches=batches_per_epoch,
                        loss=batch_loss,
                    )
                last_val_loss = _compute_eval_loss()
                _emit(
                    "epoch_end",
                    epoch=epoch + 1,
                    n_epochs=int(epochs),
                    mean_loss=ep_loss / max(ep_batches, 1),
                    mean_val_loss=last_val_loss,
                )
        finally:
            model.eval()
            _emit(
                "end",
                n_batches=n_batches,
                mean_loss=total_loss / max(n_batches, 1),
                mean_val_loss=last_val_loss,
            )

        mean_loss = total_loss / max(n_batches, 1)
        self._refresh_label_embeddings()

        metrics = {
            "mean_loss": float(mean_loss),
            "mean_val_loss": float(last_val_loss),
            "n_pairs": float(len(examples)),
            "n_eval_pairs": float(len(eval_examples)),
            "epochs": float(epochs),
            "batch_size": float(bs),
            "lr": float(lr),
            "trainable_params": float(trainable_params),
            "total_params": float(total_params),
            "trainable_pct": (
                100.0 * trainable_params / max(total_params, 1)
            ),
        }
        if log_to_mlflow:
            log_metrics = {
                "train_mean_loss": mean_loss,
                "train_n_pairs": float(len(examples)),
                "trainable_params": float(trainable_params),
            }
            if not np.isnan(last_val_loss):
                log_metrics["val_mean_loss"] = float(last_val_loss)
            self._safe_log_metrics(log_metrics)
        return metrics

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        embs = np.asarray(self.st_model.encode(texts, normalize_embeddings=True))
        return embs @ self.label_embeddings.T

    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        """MLflow pyfunc predict.

        ``model_input`` may be a DataFrame with a ``text`` column, a Series,
        or an iterable of strings. Returns a DataFrame with one
        ``sim::<key>`` float column and one ``<key>`` boolean column per
        label.
        """
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
        """Compute per-label and aggregate P/R/F1 on a ground-truth matrix.

        ``truth`` must be a bool array of shape ``(len(texts), len(label_keys))``,
        with columns in the order returned by ``self.label_keys``.
        """
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

    def load_context(self, context) -> None:
        """MLflow pyfunc load hook."""
        st_path = context.artifacts.get("st_model") if context else None
        self._load_st(st_path or self.model_name_or_path)

    def save_pretrained(self, path: str | Path) -> Path:
        """Save the underlying sentence-transformer plus this wrapper's config.

        Layout:
            <path>/st_model/...   underlying sentence-transformer files
            <path>/config.json    label_dict, threshold, learning_rate
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        st_path = path / "st_model"
        self.st_model.save(str(st_path))
        config = {
            "label_dict": self.label_dict,
            "threshold": self.threshold,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "n_trainable_transformer_layers": self.n_trainable_transformer_layers,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))
        return path

    @classmethod
    def load_pretrained(cls, path: str | Path) -> "SentenceTransformerClassifier":
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        clf = cls(
            label_dict=config["label_dict"],
            model_name_or_path=str(path / "st_model"),
            threshold=config.get("threshold", 0.5),
            learning_rate=config.get("learning_rate", 2e-5),
            device=config.get("device"),
            n_trainable_transformer_layers=config.get(
                "n_trainable_transformer_layers",
                1,
            ),
        )
        _ = clf.st_model
        return clf

    def log_to_mlflow(
        self,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
        extra_pip_requirements: list[str] | None = None,
    ) -> str:
        """Log this classifier as an mlflow.pyfunc model to the active run.

        Returns the model URI. Requires an active ``mlflow.start_run`` context.
        """
        import tempfile
        import mlflow

        with tempfile.TemporaryDirectory() as tmp:
            st_temp = Path(tmp) / "st_model"
            self.st_model.save(str(st_temp))

            info = mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=self,
                artifacts={"st_model": str(st_temp)},
                code_paths=[__file__],
                extra_pip_requirements=extra_pip_requirements
                or ["sentence-transformers"],
                registered_model_name=registered_model_name,
            )
        return getattr(info, "model_uri", f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}")

    @staticmethod
    def _safe_log_metrics(metrics: dict[str, float]) -> None:
        try:
            import mlflow

            if mlflow.active_run() is not None:
                mlflow.log_metrics(metrics)
        except Exception:
            pass