"""
Interactive multi-label text classification widget for Jupyter notebooks.

Usage:
    from IPython.display import display
    from sentence_transformers import SentenceTransformer
    from labeler import LabelingWidget

    model = SentenceTransformer("all-MiniLM-L6-v2")

    widget = LabelingWidget(
        embed_model=model,
        label_dict=labels,
        df=df,
        save_path="labeled.parquet",
        text_column="text",
    )

    display(widget)
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import numpy as np
import pandas as pd


LABEL_SEP = "::"


class LabelingWidget(widgets.VBox):
    def __init__(
        self,
        embed_model,
        label_dict: dict[str, dict[str, str]],
        df: pd.DataFrame,
        save_path: str | Path,
        text_column: str = "text",
        labels_column: str = "labels",
        top_k_highlight: int = 3,
    ):
        self.embed_model = embed_model
        self.label_dict = label_dict
        self.text_column = text_column
        self.labels_column = labels_column
        self.save_path = Path(save_path)
        self.top_k_highlight = top_k_highlight

        self.df = self._load_or_init(df)

        self.flat_labels: list[tuple[str, str, str]] = [
            (cat, lab, desc)
            for cat, labs in label_dict.items()
            for lab, desc in labs.items()
        ]

        self.cat_to_flat_idx: dict[str, list[int]] = {}
        for i, (cat, _, _) in enumerate(self.flat_labels):
            self.cat_to_flat_idx.setdefault(cat, []).append(i)

        descriptions = [f"{lab}. {desc}" for _, lab, desc in self.flat_labels]
        self.label_embeddings = np.asarray(
            embed_model.encode(descriptions, normalize_embeddings=True)
        )

        self._text_emb_cache: dict[int, np.ndarray] = {}
        self.current_idx = self._first_unlabeled()

        self._build_ui()
        self._render()

    def _load_or_init(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.save_path.exists():
            try:
                saved = self._read_df(self.save_path)

                if len(saved) == len(df) and self.text_column in saved.columns:
                    out = saved.reset_index(drop=True)

                    if self.labels_column not in out.columns:
                        out[self.labels_column] = [[] for _ in range(len(out))]
                    else:
                        out[self.labels_column] = out[self.labels_column].apply(
                            self._coerce_label_list
                        )

                    return out

            except Exception as e:
                print(f"[labeler] could not resume from {self.save_path}: {e}")

        out = df.copy().reset_index(drop=True)

        if self.labels_column not in out.columns:
            out[self.labels_column] = [[] for _ in range(len(out))]
        else:
            out[self.labels_column] = out[self.labels_column].apply(
                self._coerce_label_list
            )

        return out

    @staticmethod
    def _coerce_label_list(x: Any) -> list[str]:
        if isinstance(x, list):
            return [str(v) for v in x]

        if isinstance(x, str):
            s = x.strip()

            if not s:
                return []

            if s.startswith("["):
                try:
                    return [str(v) for v in json.loads(s)]
                except Exception:
                    pass

            return [s]

        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []

        return []

    def _read_df(self, path: Path) -> pd.DataFrame:
        ext = path.suffix.lower()

        if ext == ".parquet":
            return pd.read_parquet(path)

        if ext == ".csv":
            return pd.read_csv(path)

        if ext == ".json":
            return pd.read_json(path, orient="records")

        if ext in (".pkl", ".pickle"):
            return pd.read_pickle(path)

        raise ValueError(f"unsupported save_path extension: {ext}")

    def _save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        ext = self.save_path.suffix.lower()

        if ext == ".parquet":
            self.df.to_parquet(self.save_path, index=False)

        elif ext == ".csv":
            tmp = self.df.copy()
            tmp[self.labels_column] = tmp[self.labels_column].apply(json.dumps)
            tmp.to_csv(self.save_path, index=False)

        elif ext == ".json":
            self.df.to_json(self.save_path, orient="records", indent=2)

        elif ext in (".pkl", ".pickle"):
            self.df.to_pickle(self.save_path)

        else:
            raise ValueError(f"unsupported save_path extension: {ext}")

    def _first_unlabeled(self) -> int:
        for i, labs in enumerate(self.df[self.labels_column]):
            if not labs:
                return i

        return 0

    def _text_embedding(self, idx: int) -> np.ndarray:
        if idx not in self._text_emb_cache:
            text = str(self.df.iloc[idx][self.text_column])
            emb = self.embed_model.encode([text], normalize_embeddings=True)
            self._text_emb_cache[idx] = np.asarray(emb)[0]

        return self._text_emb_cache[idx]

    def _build_ui(self) -> None:
        self.progress_label = widgets.HTML()
        self.text_html = widgets.HTML()
        self.categories_box = widgets.VBox()
        self.status = widgets.HTML()

        self.filter_input = widgets.Text(
            value="",
            placeholder="Filter labels by category, name, or description...",
            description="Filter:",
            layout=widgets.Layout(width="100%"),
        )

        self.filter_input.observe(self._on_filter_change, names="value")

        btn_layout = widgets.Layout(width="90px")

        self.prev_btn = widgets.Button(
            description="◀ Prev",
            layout=btn_layout,
        )

        self.next_btn = widgets.Button(
            description="Next ▶",
            button_style="primary",
            layout=btn_layout,
        )

        self.skip_btn = widgets.Button(
            description="Skip",
            layout=btn_layout,
        )

        self.clear_btn = widgets.Button(
            description="Clear",
            button_style="warning",
            layout=btn_layout,
        )

        self.jump_input = widgets.BoundedIntText(
            value=1,
            min=1,
            max=max(len(self.df), 1),
            layout=widgets.Layout(width="80px"),
        )

        self.jump_btn = widgets.Button(
            description="Go",
            layout=widgets.Layout(width="50px"),
        )

        self.next_unlabeled_btn = widgets.Button(
            description="Next unlabeled ▶▶",
            layout=widgets.Layout(width="140px"),
        )

        self.prev_btn.on_click(lambda b: self._navigate(-1))
        self.next_btn.on_click(lambda b: self._navigate(+1))
        self.skip_btn.on_click(lambda b: self._navigate(+1))
        self.clear_btn.on_click(lambda b: self._clear_current())
        self.jump_btn.on_click(lambda b: self._jump(self.jump_input.value - 1))
        self.next_unlabeled_btn.on_click(lambda b: self._jump_next_unlabeled())

        nav = widgets.HBox(
            [
                self.prev_btn,
                self.next_btn,
                self.skip_btn,
                self.clear_btn,
                self.next_unlabeled_btn,
                widgets.Label("Jump to #"),
                self.jump_input,
                self.jump_btn,
            ]
        )

        super().__init__(
            children=[
                self.progress_label,
                self.text_html,
                self.filter_input,
                self.categories_box,
                nav,
                self.status,
            ]
        )

    def _on_filter_change(self, change) -> None:
        self._render()

    def _label_matches_filter(self, cat: str, lab: str, desc: str, query: str) -> bool:
        if not query:
            return True

        haystack = f"{cat} {lab} {desc}".lower()
        terms = query.lower().split()

        return all(term in haystack for term in terms)

    def _render(self) -> None:
        n = len(self.df)
        i = self.current_idx
        row = self.df.iloc[i]

        self.jump_input.value = i + 1

        text = "" if pd.isna(row[self.text_column]) else str(row[self.text_column])

        n_labeled = sum(1 for x in self.df[self.labels_column] if x)

        self.progress_label.value = (
            f"<h3 style='margin:4px 0'>{i + 1} / {n} "
            f"<span style='color:#888;font-weight:normal'>"
            f"({n_labeled} labeled, {n - n_labeled} remaining)</span></h3>"
        )

        self.text_html.value = (
            "<div style='padding:14px; border:1px solid #ddd; border-radius:6px; "
            "background:#fafafa; font-size:15px; line-height:1.55; "
            "white-space:pre-wrap; max-height:260px; overflow:auto;'>"
            f"{html.escape(text)}</div>"
        )

        text_emb = self._text_embedding(i)
        sims = self.label_embeddings @ text_emb

        current_labels = set(
            self.df.iat[i, self.df.columns.get_loc(self.labels_column)]
        )

        query = self.filter_input.value.strip()

        category_widgets = []
        total_visible = 0

        for cat, idxs in self.cat_to_flat_idx.items():
            sorted_idxs = sorted(idxs, key=lambda k: -float(sims[k]))
            buttons = []

            for rank, k in enumerate(sorted_idxs):
                _, lab, desc = self.flat_labels[k]

                if not self._label_matches_filter(cat, lab, desc, query):
                    continue

                key = f"{cat}{LABEL_SEP}{lab}"
                selected = key in current_labels
                is_top = rank < self.top_k_highlight

                btn = widgets.ToggleButton(
                    value=selected,
                    description=f"{lab}  ({sims[k]:.2f})",
                    tooltip=desc,
                    button_style=self._style_for(selected, is_top),
                    layout=widgets.Layout(width="auto", margin="2px"),
                )

                btn._label_key = key
                btn._is_top = is_top
                btn.observe(self._on_toggle, names="value")

                buttons.append(btn)

            if buttons:
                total_visible += len(buttons)

                category_widgets.append(
                    widgets.VBox(
                        [
                            widgets.HTML(
                                f"<div style='margin-top:8px;font-weight:600;color:#333'>"
                                f"{html.escape(cat)}</div>"
                            ),
                            widgets.Box(
                                buttons,
                                layout=widgets.Layout(
                                    display="flex",
                                    flex_flow="row wrap",
                                ),
                            ),
                        ]
                    )
                )

        if not category_widgets:
            category_widgets.append(
                widgets.HTML(
                    "<div style='margin:8px 0; color:#888;'>"
                    "No labels match the current filter."
                    "</div>"
                )
            )

        self.categories_box.children = category_widgets

        if query:
            self.status.value = (
                f"<span style='color:#666'>"
                f"showing {total_visible} label option(s) matching "
                f"<code>{html.escape(query)}</code>"
                f"</span>"
            )
        else:
            self.status.value = ""

    @staticmethod
    def _style_for(selected: bool, is_top: bool) -> str:
        if selected:
            return "success"

        if is_top:
            return "info"

        return ""

    def _on_toggle(self, change) -> None:
        btn = change["owner"]
        key: str = btn._label_key

        col = self.df.columns.get_loc(self.labels_column)
        labels = list(self.df.iat[self.current_idx, col])

        if change["new"]:
            if key not in labels:
                labels.append(key)
        else:
            if key in labels:
                labels.remove(key)

        self.df.iat[self.current_idx, col] = labels
        btn.button_style = self._style_for(change["new"], btn._is_top)

        try:
            self._save()
            self.status.value = (
                f"<span style='color:#2a8a2a'>✓ saved · "
                f"{len(labels)} label(s) on this row</span>"
            )
        except Exception as e:
            self.status.value = f"<span style='color:#c00'>save failed: {e}</span>"

    def _navigate(self, delta: int) -> None:
        new_idx = self.current_idx + delta

        if 0 <= new_idx < len(self.df):
            self.current_idx = new_idx
            self._render()

    def _jump(self, idx: int) -> None:
        if 0 <= idx < len(self.df):
            self.current_idx = idx
            self._render()

    def _jump_next_unlabeled(self) -> None:
        col = self.df.columns.get_loc(self.labels_column)

        for i in range(self.current_idx + 1, len(self.df)):
            if not self.df.iat[i, col]:
                self._jump(i)
                return

        for i in range(0, self.current_idx):
            if not self.df.iat[i, col]:
                self._jump(i)
                return

        self.status.value = "<span style='color:#888'>no unlabeled rows left</span>"

    def _clear_current(self) -> None:
        col = self.df.columns.get_loc(self.labels_column)
        self.df.iat[self.current_idx, col] = []

        self._save()
        self._render()

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    def as_onehot(self) -> pd.DataFrame:
        """Return df with one boolean column per category::label."""
        out = self.df.copy()

        keys = [f"{cat}{LABEL_SEP}{lab}" for cat, lab, _ in self.flat_labels]

        for key in keys:
            out[key] = out[self.labels_column].apply(lambda labs: key in labs)

        return out
