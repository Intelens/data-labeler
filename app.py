"""Dash UI over data.py (kept alongside the Streamlit version, streamlit_app.py).  Run: python app.py"""
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from data import (ALL_VALUES, DATA_END, DROPDOWNS, INTENTS, KEYS, LONG_TEXT, PAGE_SIZE, PER, PRODUCTS, TAGS, TYPES,
                  match_count, option_counts, page, pair_counts, pairs, pf_counts, pf_labels, pie_fig, products_fig,
                  save_labels, split, trend_fig)

FILTER_INPUTS = [Input(k, "value") for k in KEYS] + [Input("pf", "data")]
NF = len(FILTER_INPUTS)


def selection(args):
    """Callback args -> {dropdown key: values, "pf": ["Product=Type", ...]}"""
    return dict(zip(KEYS + ["pf"], args[:NF]))


def count_label(n):
    return html.Span(f"{n:,}" if n else "no matches", className="count")


def options(sel, key):
    opts = []
    for v, n in option_counts(sel, key).items():
        # count is a separate span so CSS can hide it inside the selected-value chip
        opts.append({"label": html.Span([v, html.Span(f"{n:,}" if n else "no matches", className="opt-count")]),
                     "value": v, "search": v, "disabled": n == 0 and v not in (sel[key] or [])})
    return opts


# ---------- product / feedback tree ----------

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
              [State(k, "value") for k in KEYS], State("pf", "data"), prevent_initial_call=True)
def remove_filter(_, __, *state):
    if not ctx.triggered[0]["value"]:  # chips re-rendered, not clicked
        raise PreventUpdate
    *current, chosen = state
    chosen = chosen or []
    t = ctx.triggered_id
    keep = [no_update] * len(KEYS)
    if t == "clear":
        return [[] for _ in KEYS] + [[]]
    if t["key"] == "pf":  # chip: "Product=*" removes the whole product, "Product=Type" one pair
        v = t["value"]
        return keep + [[x for x in chosen if not (x.startswith(v[:-1]) if v.endswith("=*") else x == v)]]
    return [[v for v in (cur or []) if v != t["value"]] if k == t["key"] else no_update
            for k, cur in zip(KEYS, current)] + [no_update]


@app.callback(Output("page", "data"),
              Input("prev", "n_clicks"), Input("next", "n_clicks"), *FILTER_INPUTS,
              Input("search", "value"), Input("sort", "value"), Input("ai", "value"), State("page", "data"),
              prevent_initial_call=True)
def change_page(*args):
    page_no = args[-1]
    if ctx.triggered_id == "prev":
        return max(page_no - 1, 0)
    if ctx.triggered_id == "next":
        return page_no + 1  # upper bound enforced by disabling "next"
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
    summary = [html.Span([html.Span("×", className="value-x", **{"aria-label": f"Remove {label}", "data-value": value}),
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
    query, sort, ai, page_no, editing, _ = args[NF:]
    pf = pair_counts(sel)
    dist = pf.groupby("type")["n"].sum().reindex(TYPES, fill_value=0)

    n, shown = page(sel, query, ai, sort, page_no)
    pages = max((n - 1) // PAGE_SIZE + 1, 1)
    if page_no > pages - 1:
        page_no = pages - 1
        n, shown = page(sel, query, ai, sort, page_no)
    banner = None
    if query and ai:
        banner = html.Div([html.B("AI search"), " Vector search isn't connected yet — showing keyword matches for now."],
                          className="banner")
    header = html.Div([html.Div("Source"), html.Div("Comment"), html.Div("Predicted labels"), html.Div()], className="row head")
    table = [header] + [comment_row(r, r.id == editing) for r in shown.itertuples()]
    if not n:
        table.append(html.Div("No comments match.", className="no-results"))

    return (
        pie_fig(dist, "Feedback distribution"),
        products_fig(pf),
        f"{n:,}", banner, table,
        f"Showing {page_no * PAGE_SIZE + 1 if n else 0}–{page_no * PAGE_SIZE + len(shown)} of {n:,}",
        f"{page_no + 1} / {pages}", page_no == 0, page_no >= pages - 1,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # reloader would load the dataset twice
