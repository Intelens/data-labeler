import functools
import os

import duckdb
import numpy as np
import pandas as pd
import plotly.io as pio
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.environ.get("COMMENTS_CSV", os.path.join(HERE, "comments.csv"))
CORRECTIONS = os.path.splitext(CSV)[0] + ".corrections.csv"  # append-only label edits, last one per id wins
TYPES = ["Interested", "Not interested", "Neutral", "Already customer"]
PAGE_SIZE = 25
DROPDOWNS = {"campaign": "Campaigns", "smo": "SMO's", "channel": "Channel", "delivery_code": "Delivery codes"}
KEYS = list(DROPDOWNS)
TAGS = {"campaign": "Campaign", "smo": "SMO", "pf": "Product", "channel": "Channel", "delivery_code": "Delivery code"}  # chip order
FILTER_INPUTS = [Input(k, "value") for k in KEYS] + [Input("pf", "data")]
NF = len(FILTER_INPUTS)


def selection(args):
    """Callback args -> {dropdown key: values, "pf": ["Product=Type", ...]}"""
    return dict(zip(KEYS + ["pf"], args[:NF]))


def split(s):
    return [x for x in (s or "").split("|") if x]


def pairs(s):
    return [tuple(x.split("=")) for x in split(s)]


# ---------- data ----------

DIMS = ["campaign", "smo", "channel", "delivery_code", "day", "fb_key"]  # filter columns, present on both tables


def build_bridge(cur, where="true", params=()):
    """comment_feedback: one row per (comment, product, feedback type) pair, carrying the comment's filter
    columns so chart/count queries never join back to the wide comments table (the join alone cost ~270 ms
    at 2M rows). Feedback strings repeat heavily, so each distinct string is split once and joined back
    (row-by-row unnest + split_part took 95 s on DuckDB 0.9; this takes <1 s)."""
    params = list(params)
    cur.execute(f"DELETE FROM comment_feedback WHERE {where}", params)
    cur.execute(f"""INSERT INTO comment_feedback
                    WITH d AS (SELECT DISTINCT feedback AS s FROM comments WHERE feedback <> '' AND {where}),
                         x AS (SELECT s, split_part(item, '=', 1) AS product, split_part(item, '=', 2) AS type
                               FROM (SELECT s, unnest(string_split(s, '|')) AS item FROM d))
                    SELECT c.id, x.product, x.type, {', '.join('c.' + d for d in DIMS)}
                    FROM comments c JOIN x ON c.feedback = x.s WHERE {where}""", params + params)


def load():
    """In-memory DuckDB rebuilt from the comments CSV + its corrections file on every start."""
    con = duckdb.connect()
    path = CSV.replace("\\", "/").replace("'", "''")
    # stored in (ts, id) order: a page of comments is then a narrow ts range, which DuckDB's zone maps
    # turn into a few row groups instead of a scan of the whole (wide) table
    # fb_key = '|A=T|B=T|': a product/feedback filter is then one contains() per pair (~60 ms at 2M rows)
    # instead of a semi-join through the bridge table (~550 ms)
    con.execute(f"""CREATE TABLE comments AS
                    SELECT *, date_trunc('day', ts) AS day, '|' || feedback || '|' AS fb_key
                    FROM read_csv_auto('{path}', header=true, timestampformat='%Y-%m-%d %H:%M',
                                       types={{'id': 'INTEGER', 'ts': 'TIMESTAMP'}})
                    ORDER BY ts, id""")
    if os.path.exists(CORRECTIONS):
        corr = pd.read_csv(CORRECTIONS, keep_default_na=False).drop_duplicates("id", keep="last")
        con.register("corr", corr)
        con.execute("""UPDATE comments SET products = corr.products, intents = corr.intents, feedback = corr.feedback,
                                           fb_key = '|' || corr.feedback || '|'
                       FROM corr WHERE comments.id = corr.id""")
    con.execute("CREATE INDEX comments_id ON comments(id)")  # label save goes by id (an IN list doesn't use it)
    con.execute("""CREATE TABLE comment_feedback (id INTEGER, product VARCHAR, type VARCHAR, campaign VARCHAR, smo VARCHAR,
                                                  channel VARCHAR, delivery_code VARCHAR, day TIMESTAMP, fb_key VARCHAR)""")
    build_bridge(con)
    return con


# ponytail: in-memory DB in one process; corrections from other processes aren't seen until restart
CON = load()


@functools.lru_cache(maxsize=512)
def _sql(query, params):
    return CON.cursor().execute(query, list(params)).df()  # cursor per call: Dash callbacks run on threads


def sql(query, params=()):
    """Results are cached per (query, params): repeat filter states and the unfiltered start page are free.
    Callers only read the returned frame. Cleared on save_labels."""
    return _sql(query, tuple(params))


PRODUCTS = sql("""SELECT DISTINCT product AS v FROM comment_feedback
                  UNION SELECT unnest(string_split(s, '|')) FROM (SELECT DISTINCT products AS s FROM comments WHERE products <> '')
                  ORDER BY 1""").v.tolist()
INTENTS = sql("SELECT DISTINCT v FROM (SELECT unnest(string_split(intents, '|')) AS v FROM comments) WHERE v <> '' ORDER BY 1").v.tolist()
ALL_VALUES = {k: sql(f"SELECT DISTINCT {k} AS v FROM comments ORDER BY 1").v.tolist() for k in KEYS}
first, last = sql("SELECT min(day) AS a, max(ts) AS b FROM comments").iloc[0]
DAYS = pd.date_range(first, last.normalize())
DATA_END = last


def where(sel, skip=None):
    """OR within a filter, AND across filters. `skip` ignores one filter (for its own option counts).
    Valid against both `comments` and `comment_feedback` (same filter columns).
    Keys are fixed filter names; only the chosen values are user input, and those are bound as parameters."""
    clauses, params = ["true"], []
    for key, chosen in sel.items():
        if not chosen or key == skip:
            continue
        if key == "pf":  # "Product=Type" pairs, matched exactly inside the '|'-delimited key
            clauses.append("(" + " OR ".join(["contains(fb_key, ?)"] * len(chosen)) + ")")
            params += [f"|{v}|" for v in chosen]
        else:
            clauses.append(f"{key} IN ({', '.join('?' * len(chosen))})")
            params += chosen
    return " AND ".join(clauses), params


# ---------- cube: every count, dropdown option and chart comes from one cached group-by ----------
# Comments repeat their filter values heavily, so grouping 2M rows by all filter columns leaves ~50k rows.
# Filtering that in pandas takes single-digit ms; the same queries against DuckDB cost 60-500 ms each
# (and a filter change fires ~10 of them). Rebuilt lazily after a label save (the sql() cache is cleared).

CUBE_DIMS = ["campaign", "smo", "channel", "delivery_code", "day", "fb_key"]


def cube(table):
    """comments: rows (dims, n comments). comment_feedback: rows (product, type, dims, n pairs)."""
    extra = "product, type, " if table == "comment_feedback" else ""
    return sql(f"SELECT {extra}{', '.join(CUBE_DIMS)}, count(*) AS n FROM {table} GROUP BY ALL")


cube("comments"), cube("comment_feedback")  # ~1.3 s at 2M rows: pay it at startup, not on the first click


def cube_mask(df, sel, skip=None):
    """Same semantics as where(): OR within a filter, AND across, `skip` ignores one filter."""
    m = np.ones(len(df), dtype=bool)
    for key, chosen in sel.items():
        if not chosen or key == skip:
            continue
        if key == "pf":  # match pairs on the ~140 distinct keys, then map back
            hits = {k for k in df["fb_key"].unique() if any(f"|{v}|" in k for v in chosen)}
            m &= df["fb_key"].isin(hits).to_numpy()
        else:
            m &= df[key].isin(chosen).to_numpy()
    return m


def pair_counts(sel, skip=None):
    """(product, type, n) for the comments matching sel."""
    f = cube("comment_feedback")
    return f[cube_mask(f, sel, skip)].groupby(["product", "type"])["n"].sum().reset_index()


def match_count(sel):
    c = cube("comments")
    return int(c["n"][cube_mask(c, sel)].sum())


def count_label(n):
    return html.Span(f"{n:,}" if n else "no matches", className="count")


def options(sel, key):
    c = cube("comments")
    counts = c[cube_mask(c, sel, skip=key)].groupby(key)["n"].sum()
    opts = []
    for v in ALL_VALUES[key]:
        n = int(counts.get(v, 0))
        # count is a separate span so CSS can hide it inside the selected-value chip
        opts.append({"label": html.Span([v, html.Span(f"{n:,}" if n else "no matches", className="opt-count")]),
                     "value": v, "search": v, "disabled": n == 0 and v not in (sel[key] or [])})
    return opts


# ---------- product / feedback tree ----------

def pf_counts(sel):
    return {(r[0], r[1]): int(r[2]) for r in pair_counts(sel, skip="pf").itertuples(index=False)}  # a product appears once per comment


def pf_labels(chosen):
    """[(chip value, label)]: "Product/*" when every feedback type of a product is selected, else one per pair."""
    out = []
    for p in PRODUCTS:
        types = [t for t in TYPES if f"{p}={t}" in chosen]
        if len(types) == len(TYPES):
            out.append((f"{p}=*", f"{p}/*"))
        else:
            out += [(f"{p}={t}", f"{p}/{t}") for t in types]
    return out


ARIA = {"on": "true", "half": "mixed", "off": "false"}


def pf_tree(chosen, counts):
    """Server renders counts + committed state; ticking happens client-side (assets/tree_dropdown.js)
    and the selection is applied to the "pf" store when the dropdown closes."""
    groups = []
    for p in PRODUCTS:
        picked = [t for t in TYPES if f"{p}={t}" in chosen]
        state = "on" if len(picked) == len(TYPES) else "half" if picked else "off"
        total = sum(counts.get((p, t), 0) for t in TYPES)
        children = []
        for t in TYPES:
            n, on = counts.get((p, t), 0), t in picked
            children.append(html.Button(
                [html.Span(className=f"check {'on' if on else 'off'}"), html.Span(t), count_label(n)],
                className="tree-row child", disabled=not n and not on, role="checkbox",
                **{"aria-checked": ARIA["on" if on else "off"], "data-pair": f"{p}={t}"}))
        groups.append(html.Details([
            html.Summary([
                html.Button(className=f"check {state}", disabled=not total and not picked, role="checkbox",
                            **{"aria-checked": ARIA[state], "aria-label": f"All feedback for {p}"}),
                html.Span(p), count_label(total),
            ], className="tree-row" if total or picked else "tree-row disabled"),
            *children,
        ], className="tree-group"))
    return groups


def save_labels(comment_id, products, intents, feedback):
    row = {"products": "|".join(products), "intents": "|".join(intents), "feedback": "|".join(feedback)}
    cur = CON.cursor()
    cur.execute("UPDATE comments SET products = ?, intents = ?, feedback = ?, fb_key = ? WHERE id = ?",
                [*row.values(), f"|{row['feedback']}|", comment_id])
    build_bridge(cur, "id = ?", [comment_id])
    _sql.cache_clear()
    pd.DataFrame([{"id": comment_id, **row}]).to_csv(CORRECTIONS, mode="a", header=not os.path.exists(CORRECTIONS), index=False)


def keyword_search(q):
    """SQL clause + params. ponytail: full scan of text, add DuckDB's fts extension if that gets slow."""
    return "(contains(lower(text), lower(?)) OR contains(lower(author), lower(?)))", [q, q]


def vector_search(q):
    # ponytail: placeholder for semantic search — embed q, query a vector index, restrict/rank comments by similarity.
    # Falls back to keyword matching until that exists.
    return keyword_search(q)


# ---------- figures ----------
# Built as plain dicts: plotly.express cost ~100 ms per figure just to construct the object (measured on 2M rows,
# where the SQL behind it takes 20-120 ms). The "plotly" template is embedded once so the look is unchanged.

TEMPLATE = pio.templates["plotly"].to_plotly_json()
LEGEND = {"title": {"text": "Feedback type"}}


def figure(data, height, title=None, **layout):
    base = {"template": TEMPLATE, "height": height, "margin": {"l": 50, "r": 20, "t": 50 if title else 20, "b": 40},
            "font": {"family": "Open Sans, verdana, arial, sans-serif"}, "legend": LEGEND}
    if title:
        base["title"] = {"text": title}
    return {"data": data, "layout": base | layout}


def by_type(counts, index):
    """counts: rows (v, type, n) -> {type: [n per v in index order]}, zero-filled so every figure lists all TYPES in order."""
    table = counts.pivot_table(index="v", columns="type", values="n", fill_value=0, aggfunc="sum").reindex(index=index, columns=TYPES, fill_value=0)
    return {t: table[t].tolist() for t in TYPES}


PER = {"day": "Day", "campaign": "Campaign", "smo": "SMO", "product": "Product", "channel": "Channel", "delivery_code": "Delivery code"}


def trend_fig(per, sel):
    """Feedbacks per <per>: lines over time for "day", sorted stacked bars for any other dimension."""
    f = cube("comment_feedback")
    counts = f[cube_mask(f, sel)].groupby([per, "type"])["n"].sum().reset_index().rename(columns={per: "v"})
    if per == "day":
        x = [d.strftime("%Y-%m-%d") for d in DAYS]
        series = by_type(counts, DAYS)
        data = [{"type": "scatter", "mode": "lines+markers", "name": t, "x": x, "y": series[t]} for t in TYPES]
        return figure(data, 340, xaxis={"title": {"text": "Day"}}, yaxis={"title": {"text": "Feedbacks"}})
    order = counts.groupby("v")["n"].sum().sort_values(ascending=False).index.tolist()
    series = by_type(counts, order)
    data = [{"type": "bar", "name": t, "x": order, "y": series[t], "hovertemplate": "%{x} · " + t + ": %{y:,}<extra></extra>"} for t in TYPES]
    return figure(data, 340, barmode="stack", xaxis={"title": {"text": PER[per]}, "automargin": True, "categoryorder": "array", "categoryarray": order},
                  yaxis={"title": {"text": "Feedbacks"}})


def products_fig(pf):
    """One horizontal bar per product, stacked by feedback type, biggest product on top."""
    order = pf.groupby("product")["n"].sum().sort_values().index.tolist()  # y categories draw bottom-up
    series = by_type(pf.rename(columns={"product": "v"}), order)
    data = [{"type": "bar", "orientation": "h", "name": t, "y": order, "x": series[t],
             "hovertemplate": "%{y} · " + t + ": %{x:,}<extra></extra>"} for t in TYPES]
    return figure(data, 320, "Products mentioned", barmode="stack", xaxis={"title": {"text": "Feedbacks"}},
                  yaxis={"automargin": True, "categoryorder": "array", "categoryarray": order})


def pie_fig(counts, title):
    data = [{"type": "pie", "labels": list(counts.index), "values": [int(v) for v in counts.values], "sort": False,
             "textinfo": "percent", "hovertemplate": "%{label}: %{value:,}<extra></extra>"}]
    return figure(data, 320, title)


# ---------- comment rows ----------

def label_row(name, children):
    return html.Div([html.Span(name, className="label-name"), html.Div(children, className="label-values")], className="label-row")


def view_labels(r):
    return [
        label_row("Product", [html.Span(p, className="tag prod") for p in split(r.products)]),
        label_row("Intent", [html.Span(i, className="tag intent") for i in split(r.intents)]),
        label_row("Feedback", [html.Span([html.Span(p, className="pair-p"), html.Span(t, className="pair-t")], className="pair")
                               for p, t in pairs(r.feedback)]),
    ]


def edit_labels(r):
    i = r.id
    pair_opts = [{"label": f"{p}: {t}", "value": f"{p}={t}"} for p in PRODUCTS for t in TYPES]
    return [
        label_row("Product", dcc.Dropdown(PRODUCTS, split(r.products), multi=True, id={"type": "ed-prod", "id": i})),
        label_row("Intent", dcc.Dropdown(INTENTS, split(r.intents), multi=True, id={"type": "ed-int", "id": i})),
        label_row("Feedback", dcc.Dropdown(pair_opts, split(r.feedback), multi=True, id={"type": "ed-fb", "id": i})),
        html.Div([
            html.Button("Save", id={"type": "save", "id": i}, className="btn primary"),
            html.Button("Cancel", id={"type": "cancel", "id": i}, className="btn"),
            html.Span("Corrections retrain the model", className="muted small push"),
        ], className="actions"),
    ]


LONG_TEXT = 280  # ponytail: chars, roughly 3 lines at this column width; CSS clamps to 3 lines regardless


def comment_text(r):
    """Long comments are clamped to 3 lines inside a native <details>; clicking the text or the label expands it (CSS only)."""
    if len(r.text) <= LONG_TEXT:
        return html.Div(r.text, className="text")
    return html.Details(html.Summary([html.Div(r.text, className="text clamp"), html.Span(className="more-label")]),
                        className="text-cell more")


def comment_row(r, editing):
    return html.Div([
        html.Div([html.B(r.author), html.Span(r.channel), html.Span(f"{r.ts:%d %b · %H:%M}"),
                  html.Span(f"{r.campaign} · {r.smo}", className="muted"), html.Span(r.delivery_code, className="muted")],
                 className="src"),
        comment_text(r),
        html.Div(edit_labels(r) if editing else view_labels(r), className="labels"),
        html.Div(html.Button("✎", id={"type": "edit", "id": r.id}, title="Correct labels",
                             className="pencil on" if editing else "pencil")),
    ], className="row editing" if editing else "row")


# ---------- layout ----------

def field(label, control):
    return html.Div([html.Label(label), control], className="field")


def dropdown(key):
    return field(DROPDOWNS[key], dcc.Dropdown(id=key, multi=True, placeholder="Select…"))


tree_dropdown = field("Products / Feedback", html.Details([
    html.Summary([html.Span(id="pf-summary", className="tree-value"), html.Span(className="sep"), html.Span(className="caret")],
                 className="tree-control"),
    html.Div(id="pf-tree", className="tree-menu"),
], className="tree-dd"))

app = Dash(__name__, title="Social Feedback Monitor")
app.layout = html.Div([
    html.Div([html.H1("Social Feedback Monitor"),
              html.Span(f"Data through {DATA_END:%d %b %Y}", className="muted")], className="header"),
    html.Div([dropdown("campaign"), dropdown("smo"), tree_dropdown, dropdown("channel"), dropdown("delivery_code")],
             className="filters"),
    html.Div([html.Span("Active filters", className="active-label"), html.Span(id="chips", className="chips"),
              html.Span(id="match-count", className="push"), html.Button("Clear all", id="clear", className="link")],
             className="active"),
    html.Div([
        html.Div([html.Span("Feedbacks per"),
                  dcc.Dropdown([{"label": v, "value": k} for k, v in PER.items()], "day", id="per", clearable=False, className="per")],
                 className="chart-head"),
        dcc.Graph(id="trend", config={"displaylogo": False}),
    ], className="card"),
    html.Div([html.Div(dcc.Graph(id="dist", config={"displaylogo": False}), className="card"),
              html.Div(dcc.Graph(id="prods", config={"displaylogo": False}), className="card")], className="two"),
    html.Div([
        html.Div([html.Span("Comments", className="h2"), html.Span(id="n-comments", className="muted")], className="push"),
        html.Label("Sort by"),
        dcc.Dropdown(["Newest first", "Oldest first"], "Newest first", id="sort", clearable=False, className="sort"),
        html.Div([dcc.Input(id="search", type="search", placeholder="Search comments…", debounce=True, className="search"),
                  dcc.Checklist([{"label": "AI search", "value": "on"}], [], id="ai", className="ai")], className="search-group"),
    ], className="toolbar"),
    html.Div(id="ai-banner"),
    html.Div(id="comments", className="table"),
    html.Div([html.Span(id="showing"),
              html.Div([html.Button("‹", id="prev", className="btn"), html.Span(id="page-label"),
                        html.Button("›", id="next", className="btn")], className="pager")], className="footer"),
    dcc.Store(id="pf", data=[]), dcc.Store(id="page", data=0), dcc.Store(id="editing"), dcc.Store(id="version", data=0),
], className="page")


# ---------- callbacks ----------

@app.callback([Output(k, "value") for k in KEYS], Output("pf", "data"),
              Input({"type": "chip", "key": ALL, "value": ALL}, "n_clicks"), Input("clear", "n_clicks"),
              Input({"type": "pf-chip", "value": ALL}, "n_clicks"),  # × on chips inside the tree dropdown
              [State(k, "value") for k in KEYS], State("pf", "data"), prevent_initial_call=True)
def remove_filter(_, __, ___, *state):
    if not ctx.triggered[0]["value"]:  # chips re-rendered, not clicked
        raise PreventUpdate
    *current, chosen = state
    chosen = chosen or []
    t = ctx.triggered_id
    keep = [no_update] * len(KEYS)
    if t == "clear":
        return [[] for _ in KEYS] + [[]]
    if t.get("key", "pf") == "pf":  # chip: "Product=*" removes the whole product, "Product=Type" one pair
        v = t["value"]
        return keep + [[x for x in chosen if not (x.startswith(v[:-1]) if v.endswith("=*") else x == v)]]
    return [[v for v in (cur or []) if v != t["value"]] if k == t["key"] else no_update
            for k, cur in zip(KEYS, current)] + [no_update]


@app.callback(Output("page", "data"),
              Input("prev", "n_clicks"), Input("next", "n_clicks"), *FILTER_INPUTS,
              Input("search", "value"), Input("sort", "value"), Input("ai", "value"), State("page", "data"),
              prevent_initial_call=True)
def change_page(*args):
    page = args[-1]
    if ctx.triggered_id == "prev":
        return max(page - 1, 0)
    if ctx.triggered_id == "next":
        return page + 1  # upper bound enforced by disabling "next"
    return 0


@app.callback(Output("editing", "data"), Output("version", "data"),
              Input({"type": "edit", "id": ALL}, "n_clicks"), Input({"type": "save", "id": ALL}, "n_clicks"),
              Input({"type": "cancel", "id": ALL}, "n_clicks"),
              State({"type": "ed-prod", "id": ALL}, "value"), State({"type": "ed-int", "id": ALL}, "value"),
              State({"type": "ed-fb", "id": ALL}, "value"), State("editing", "data"), State("version", "data"),
              prevent_initial_call=True)
def edit_comment(_, __, ___, prods, intents, fbs, editing, version):
    if not ctx.triggered[0]["value"]:
        raise PreventUpdate
    t = ctx.triggered_id
    if t["type"] == "edit":
        return (None if editing == t["id"] else t["id"]), no_update
    if t["type"] == "save":
        save_labels(t["id"], prods[0] or [], intents[0] or [], fbs[0] or [])
        return None, version + 1
    return None, no_update


# Filter panel is its own callback so filters update without waiting for charts and comments.
@app.callback(
    [Output(k, "options") for k in KEYS], Output("pf-tree", "children"), Output("pf-summary", "children"),
    Output("chips", "children"), Output("match-count", "children"),
    *FILTER_INPUTS, Input("version", "data"),
)
def filter_panel(*args):
    sel = selection(args)
    labels = pf_labels(sel["pf"] or [])
    chips = [html.Button([html.Span("×", className="x"), html.Span([html.Span(tag, className="chip-tag"), label], className="chip-label")],
                         id={"type": "chip", "key": k, "value": value}, className="chip")
             for k, tag in TAGS.items()
             for value, label in (labels if k == "pf" else [(v, v) for v in sel[k] or []])] or html.Span("none", className="muted")
    summary = [html.Span([html.Span("×", id={"type": "pf-chip", "value": value}, className="value-x", **{"aria-label": f"Remove {label}"}),
                          html.Span(label, className="value-label")], className="value-chip")
               for value, label in labels] or html.Span("Select…", className="placeholder")
    return (*[options(sel, k) for k in KEYS], pf_tree(sel["pf"] or [], pf_counts(sel)), summary,
            chips, [html.B(f"{match_count(sel):,}"), " comments match"])


@app.callback(Output("trend", "figure"), *FILTER_INPUTS, Input("per", "value"), Input("version", "data"))
def trend(*args):
    per = args[NF] if args[NF] in PER else "day"
    return trend_fig(per, selection(args))


@app.callback(
    Output("dist", "figure"), Output("prods", "figure"),
    Output("n-comments", "children"), Output("ai-banner", "children"), Output("comments", "children"),
    Output("showing", "children"), Output("page-label", "children"),
    Output("prev", "disabled"), Output("next", "disabled"),
    *FILTER_INPUTS,
    Input("search", "value"), Input("sort", "value"), Input("ai", "value"),
    Input("page", "data"), Input("editing", "data"), Input("version", "data"),
)
def render(*args):
    sel = selection(args)
    query, sort, ai, page, editing, _ = args[NF:]
    pf = pair_counts(sel)
    dist = pf.groupby("type")["n"].sum().reindex(TYPES, fill_value=0)

    banner = None
    cw, cp = where(sel)
    n = match_count(sel)
    if query:
        sw, sp = (vector_search if ai else keyword_search)(query)
        cw, cp = f"{cw} AND {sw}", cp + sp
        n = int(sql(f"SELECT count(*) AS n FROM comments WHERE {cw}", cp).n[0])
        if ai:
            banner = html.Div([html.B("AI search"), " Vector search isn't connected yet — showing keyword matches for now."],
                              className="banner")
    pages = max((n - 1) // PAGE_SIZE + 1, 1)
    page = min(page, pages - 1)
    order = "DESC" if sort == "Newest first" else "ASC"
    # sort narrow (ts, id) rows to pick the page, then fetch the wide text rows within that page's ts range
    keys = sql(f"SELECT id, ts FROM comments WHERE {cw} ORDER BY ts {order}, id {order} LIMIT {PAGE_SIZE} OFFSET {page * PAGE_SIZE}", cp)
    shown = sql(f"""SELECT * FROM comments WHERE ts BETWEEN ? AND ? AND id IN ({', '.join('?' * len(keys))})
                    ORDER BY ts {order}, id {order}""", [keys.ts.min(), keys.ts.max(), *keys.id.tolist()]) if len(keys) \
        else sql("SELECT * FROM comments WHERE false")
    header = html.Div([html.Div("Source"), html.Div("Comment"), html.Div("Predicted labels"), html.Div()], className="row head")
    table = [header] + [comment_row(r, r.id == editing) for r in shown.itertuples()]
    if not n:
        table.append(html.Div("No comments match.", className="no-results"))

    return (
        pie_fig(dist, "Feedback distribution"),
        products_fig(pf),
        f"{n:,}", banner, table,
        f"Showing {page * PAGE_SIZE + 1 if n else 0}–{page * PAGE_SIZE + len(shown)} of {n:,}",
        f"{page + 1} / {pages}", page == 0, page >= pages - 1,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # reloader would load the dataset twice
