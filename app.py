"""Social Feedback Monitor — Streamlit UI over data.py.  Run: streamlit run streamlit_app.py"""
import html
import os

import streamlit as st

import data as d

st.set_page_config(page_title="Social Feedback Monitor", layout="wide")
ss = st.session_state
for k, v in {"pf": [], "page": 0, "editing": None}.items():
    ss.setdefault(k, v)


def sel():
    return {k: ss.get(k) or [] for k in d.KEYS} | {"pf": ss["pf"]}


# ---- callbacks: run before the rerun, so they may change any widget's state ----

def reset_page():
    ss.page = 0


def set_pf(pairs):
    ss.pf = pairs
    reset_page()


def tree_changed():
    """The tree component committed (dropdown closed, or a chip × inside it clicked)."""
    pairs = ss.pf_tree["commit"] or []
    valid = {f"{p}={t}" for p in d.PRODUCTS for t in d.TYPES}  # the value comes from the browser: keep only known pairs
    set_pf([p for p in pairs if p in valid])


def remove_chip(key, value):
    if key == "pf":  # "Product=*" removes the whole product, "Product=Type" one pair
        set_pf([x for x in ss.pf if not (x.startswith(value[:-1]) if value.endswith("=*") else x == value)])
    else:
        ss[key] = [v for v in ss[key] if v != value]
        reset_page()


def clear_all():
    for k in d.KEYS:
        ss[k] = []
    set_pf([])


def save(cid):
    d.save_labels(cid, ss[f"ed_prod_{cid}"], ss[f"ed_int_{cid}"], ss[f"ed_fb_{cid}"])
    ss.editing = None


def set_editing(cid):
    ss.editing = None if ss.editing == cid else cid


# ---- styles for the HTML parts (comment rows, chips) ----

st.html("""<style>
@import url("https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap");
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) { font-family: "Open Sans", verdana, arial, sans-serif; }
.block-container { max-width: 1320px; padding-top: 3rem; padding-bottom: 2rem; }
.stApp { color: #2a3f5f; }
.muted { color: #7b8ba5; }
.header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 6px; }
.header h1 { font-size: 22px; font-weight: 600; color: #2a3f5f; margin: 0; padding: 0; }
.header .muted { font-size: 12px; }
[data-testid="stWidgetLabel"] p, .field-label { font-size: 12px !important; font-weight: 600; color: #506784; }
.field-label { margin: 0 0 -8px; }  /* mimics a widget label above the popover */
.tree-control { min-height: 40px; border-color: #e3e8f0; }  /* match Streamlit's 40px multiselects */
[data-testid="stElementContainer"]:has(.tree-host) { overflow: visible; z-index: 1000; }
.st-key-active { background: #f7f9fc; border: 1px solid #e3e8f0; border-radius: 4px; padding: 6px 12px; }
.st-key-active button { border: 1px solid #c2e0ff; background: #ebf5ff; color: #007eff; border-radius: 2px; font-size: 13px; min-height: 26px; padding: 2px 8px; }
.st-key-active .st-key-clear button { border: none; background: none; color: #007eff; font-size: 12px; }
.st-key-card, .st-key-card1, .st-key-card2 { border: 1px solid #e3e8f0; border-radius: 4px; padding: 10px 12px 0; }
.chart-title { font-size: 17px; color: #2a3f5f; margin: 0; }
.h2 { font-size: 17px; color: #2a3f5f; }
.chips-note { font-size: 12px; color: #506784; }
.table-head { display: grid; grid-template-columns: 150px 1fr 420px; background: #f7f9fc; border: 1px solid #e3e8f0; border-radius: 4px; font-size: 11px; font-weight: 600; color: #506784; text-transform: uppercase; letter-spacing: .4px; padding: 8px 14px; }
.row { display: grid; grid-template-columns: 150px 1fr 420px; gap: 0; font-size: 13px; color: #2a3f5f; line-height: 1.5; }
.src { display: flex; flex-direction: column; gap: 3px; font-size: 12px; color: #506784; }
.src b { color: #2a3f5f; }
.text.clamp { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
details.more summary { list-style: none; cursor: pointer; }
details.more summary::-webkit-details-marker { display: none; }
details.more[open] .text.clamp { display: block; }
.more-label { display: inline-block; margin-top: 4px; font-size: 12px; color: #007eff; }
.more-label::after { content: "Show more"; }
details.more[open] .more-label::after { content: "Show less"; }
.labels { display: flex; flex-direction: column; gap: 7px; font-size: 12px; }
.label-row { display: flex; align-items: flex-start; gap: 6px; }
.label-name { width: 62px; flex: none; color: #7b8ba5; padding-top: 4px; }
.label-values { display: flex; flex-wrap: wrap; gap: 6px; }
.tag { padding: 3px 8px; border-radius: 2px; }
.tag.prod { background: #e8f7f2; border: 1px solid #9fe0cb; color: #0a7a5b; }
.tag.intent { background: #f3ebff; border: 1px solid #d6bdfb; color: #6b2fc2; }
.pair { display: inline-flex; border: 1px solid #c2e0ff; border-radius: 2px; overflow: hidden; }
.pair-p { padding: 3px 8px; background: #ebf5ff; color: #007eff; }
.pair-t { padding: 3px 8px; border-left: 1px solid #c2e0ff; }
</style>""")

e = html.escape

# ---- Products / Feedback tree: same markup, CSS and JS as the Dash UI (assets/tree.css, assets/tree_dropdown.js) ----

ASSETS = os.path.join(d.HERE, "assets")


def tree_component():
    """Custom component mounted straight into the page (no iframe), so the menu can overlay what's below."""
    read = lambda f: open(os.path.join(ASSETS, f), encoding="utf-8").read()
    js = ("const STREAMLIT_TREE = true;\n" + read("tree_dropdown.js") + """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let host = parentElement.querySelector(".tree-host");
  if (!host) {
    host = document.createElement("div");
    host.className = "tree-host";
    parentElement.appendChild(host);
  }
  if (host.dataset.html !== data.html) { // keep DOM (and expanded products) when nothing changed
    const open = [...host.querySelectorAll(".tree-group[open]")].map((g) => g.dataset.product);
    host.innerHTML = data.html;
    host.dataset.html = data.html;
    host.querySelectorAll(".tree-group").forEach((g) => { if (open.includes(g.dataset.product)) g.open = true; });
  }
  // a trigger (not state): it reruns the app and fires on_commit_change every time, even for an identical selection;
  // each render hands over a fresh setter, so listeners always call the latest one
  parentElement.treeCommit = setTriggerValue;
  if (!parentElement.treeCleanup) {
    parentElement.treeCleanup = initTreeDropdown(parentElement, (pairs) => parentElement.treeCommit("commit", pairs));
  }
  return () => { parentElement.treeCleanup?.(); parentElement.treeCleanup = null; };
}""")
    return st.components.v2.component("pf_tree", html="", css=read("tree.css"), js=js, isolate_styles=False)


def tree_html(chosen, counts, labels):
    """Same structure as the Dash tree (app.py pf_tree): tri-state product rows with feedback-type children."""
    def count(n):
        return f"<span class='count'>{n:,}</span>" if n else "<span class='count'>no matches</span>"

    aria = {"on": "true", "half": "mixed", "off": "false"}
    groups = []
    for p in d.PRODUCTS:
        picked = [t for t in d.TYPES if f"{p}={t}" in chosen]
        state = "on" if len(picked) == len(d.TYPES) else "half" if picked else "off"
        total = sum(counts.get((p, t), 0) for t in d.TYPES)
        kids = "".join(
            f"<button class='tree-row child' role='checkbox' aria-checked='{aria['on' if t in picked else 'off']}' data-pair='{e(p)}={e(t)}'"
            f"{' disabled' if not counts.get((p, t)) and t not in picked else ''}>"
            f"<span class='check {'on' if t in picked else 'off'}'></span><span>{e(t)}</span>{count(counts.get((p, t), 0))}</button>"
            for t in d.TYPES)
        disabled = not total and not picked
        groups.append(
            f"<details class='tree-group' data-product='{e(p)}'>"
            f"<summary class='tree-row{' disabled' if disabled else ''}'>"
            f"<button class='check {state}' role='checkbox' aria-checked='{aria[state]}' aria-label='All feedback for {e(p)}'{' disabled' if disabled else ''}></button>"
            f"<span>{e(p)}</span>{count(total)}</summary>{kids}</details>")
    chips = "".join(f"<span class='value-chip'><span class='value-x' data-value='{e(v)}' aria-label='Remove {e(l)}'>×</span>"
                    f"<span class='value-label'>{e(l)}</span></span>" for v, l in labels) or "<span class='placeholder'>Select…</span>"
    return (f"<details class='tree-dd'><summary class='tree-control'><span class='tree-value'>{chips}</span>"
            f"<span class='sep'></span><span class='caret'></span></summary><div class='tree-menu'>{''.join(groups)}</div></details>")


# ---- header + filters ----

st.html(f"<div class='header'><h1>Social Feedback Monitor</h1><span class='muted'>Data through {d.DATA_END:%d %b %Y}</span></div>")

cols = st.columns(5)
current = sel()
for col, key in zip([cols[0], cols[1], cols[3], cols[4]], d.KEYS):
    counts = d.option_counts(current, key)
    # ponytail: options with no matches are hidden rather than greyed out; Streamlit can't disable single options
    opts = [v for v in d.ALL_VALUES[key] if counts[v] or v in current[key]]
    col.multiselect(d.DROPDOWNS[key], opts, key=key, placeholder="Select…", on_change=reset_page)

with cols[2]:
    labels = d.pf_labels(ss.pf)
    st.html("<p class='field-label'>Products / Feedback</p>")
    tree_component()(key="pf_tree", data={"html": tree_html(ss.pf, d.pf_counts(current), labels)}, on_commit_change=tree_changed)

# ---- active filters ----

with st.container(key="active", horizontal=True, vertical_alignment="center"):
    st.markdown("<span class='chips-note'><b>Active filters</b></span>", unsafe_allow_html=True)
    chips = [(k, v, f"{d.TAGS[k]}: {label}") for k, tag in d.TAGS.items()
             for v, label in (labels if k == "pf" else [(x, x) for x in current[k]])]
    for k, v, label in chips:
        st.button(f"× {label}", key=f"chip_{k}_{v}", on_click=remove_chip, args=(k, v))
    if not chips:
        st.markdown("<span class='muted chips-note'>none</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='chips-note'><b>{d.match_count(current):,}</b> comments match</span>", unsafe_allow_html=True)
    st.button("Clear all", key="clear", type="tertiary", on_click=clear_all)

# ---- charts ----

CHART = {"width": "stretch", "theme": None, "config": {"displaylogo": False}}  # theme=None keeps the plotly template look

with st.container(key="card"):
    head, per_col = st.columns([1, 6], vertical_alignment="center")
    head.markdown("<p class='chart-title'>Feedbacks per</p>", unsafe_allow_html=True)
    per = per_col.selectbox("per", list(d.PER), format_func=d.PER.get, label_visibility="collapsed", width=170)
    st.plotly_chart(d.trend_fig(per, current), **CHART)

pf = d.pair_counts(current)
c1, c2 = st.columns(2)
with c1.container(key="card1"):
    st.plotly_chart(d.pie_fig(pf.groupby("type")["n"].sum().reindex(d.TYPES, fill_value=0), "Feedback distribution"), **CHART)
with c2.container(key="card2"):
    st.plotly_chart(d.products_fig(pf), **CHART)

# ---- comments ----

t1, t2, t3, t4 = st.columns([3, 1.4, 3, 1], vertical_alignment="bottom")
sort = t2.selectbox("Sort by", ["Newest first", "Oldest first"], on_change=reset_page)
query = t3.text_input("Search comments", placeholder="Search comments…", on_change=reset_page)
ai = t4.toggle("AI search")
n, shown = d.page(current, query, ai, sort, ss.page)
pages = max((n - 1) // d.PAGE_SIZE + 1, 1)
ss.page = min(ss.page, pages - 1)
t1.markdown(f"<span class='h2'>Comments</span> <span class='muted' style='font-size:13px'>{n:,}</span>", unsafe_allow_html=True)
if query and ai:
    st.info("**AI search** — vector search isn't connected yet; showing keyword matches for now.")
st.html("<div class='table-head'><div>Source</div><div>Comment</div><div>Predicted labels</div></div>")


def text_html(r):
    body = f"<div class='text clamp'>{e(r.text)}</div>"
    if len(r.text) <= d.LONG_TEXT:
        return body
    return f"<details class='more'><summary>{body}<span class='more-label'></span></summary></details>"


def labels_html(r):
    prods = "".join(f"<span class='tag prod'>{e(p)}</span>" for p in d.split(r.products))
    ints = "".join(f"<span class='tag intent'>{e(i)}</span>" for i in d.split(r.intents))
    fb = "".join(f"<span class='pair'><span class='pair-p'>{e(p)}</span><span class='pair-t'>{e(t)}</span></span>" for p, t in d.pairs(r.feedback))
    return (f"<div class='labels'><div class='label-row'><span class='label-name'>Product</span><div class='label-values'>{prods}</div></div>"
            f"<div class='label-row'><span class='label-name'>Intent</span><div class='label-values'>{ints}</div></div>"
            f"<div class='label-row'><span class='label-name'>Feedback</span><div class='label-values'>{fb}</div></div></div>")


pair_opts = [f"{p}={t}" for p in d.PRODUCTS for t in d.TYPES]
if not n:
    st.markdown("<div class='muted' style='text-align:center;padding:24px'>No comments match.</div>", unsafe_allow_html=True)
for r in shown.itertuples():
    with st.container(border=True):
        a, b = st.columns([11, 0.6], vertical_alignment="top")
        src = (f"<div class='src'><b>{e(r.author)}</b><span>{e(r.channel)}</span><span>{r.ts:%d %b · %H:%M}</span>"
               f"<span class='muted'>{e(r.campaign)} · {e(r.smo)}</span><span class='muted'>{e(r.delivery_code)}</span></div>")
        if ss.editing == r.id:
            a.html(f"<div class='row'>{src}{text_html(r)}<div></div></div>")
            with a.form(f"edit_{r.id}", border=False):
                st.multiselect("Product", d.PRODUCTS, d.split(r.products), key=f"ed_prod_{r.id}")
                st.multiselect("Intent", d.INTENTS, d.split(r.intents), key=f"ed_int_{r.id}")
                st.multiselect("Feedback", pair_opts, d.split(r.feedback), key=f"ed_fb_{r.id}", format_func=lambda v: v.replace("=", ": "))
                s1, s2, s3 = st.columns([1, 1, 4])
                s1.form_submit_button("Save", type="primary", on_click=save, args=(r.id,))
                s2.form_submit_button("Cancel", on_click=set_editing, args=(r.id,))
                s3.caption("Corrections retrain the model")
        else:
            a.html(f"<div class='row'>{src}{text_html(r)}{labels_html(r)}</div>")
        b.button("✎", key=f"edit_btn_{r.id}", help="Correct labels", type="primary" if ss.editing == r.id else "secondary",
                 on_click=set_editing, args=(r.id,))

f1, f2 = st.columns([4, 1.6], vertical_alignment="center")
f1.caption(f"Showing {ss.page * d.PAGE_SIZE + 1 if n else 0}–{ss.page * d.PAGE_SIZE + len(shown)} of {n:,}")
with f2.container(horizontal=True, vertical_alignment="center"):
    st.button("‹", key="prev", disabled=ss.page == 0, on_click=lambda: ss.update(page=ss.page - 1))
    st.caption(f"{ss.page + 1} / {pages}")
    st.button("›", key="next", disabled=ss.page >= pages - 1, on_click=lambda: ss.update(page=ss.page + 1))
