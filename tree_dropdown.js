// Products / Feedback tree dropdown (native <details>), Excel-filter style. Shared by both UIs:
// - Dash loads this file from assets/ and commits to the "pf" store (bottom of file)
// - Streamlit prepends it to its component module and commits to component state (streamlit_app.py)
// Ticking is instant and local; onCommit(pairs) runs once when the dropdown closes, or when a chip's × is clicked.
const TREE_ARIA = { on: "true", half: "mixed", off: "false" };

function treeSetCheck(check, state) {
  check.classList.remove("on", "half", "off");
  check.classList.add(state);
  check.closest('[role="checkbox"]').setAttribute("aria-checked", TREE_ARIA[state]);
}

function treeSyncProduct(group) {
  const kids = [...group.querySelectorAll(".tree-row.child .check")];
  const on = kids.filter((c) => c.classList.contains("on")).length;
  treeSetCheck(group.querySelector("summary .check"), on === 0 ? "off" : on === kids.length ? "on" : "half");
}

function treeSelected(dd) {
  return [...dd.querySelectorAll(".tree-row.child")].filter((r) => r.querySelector(".check.on")).map((r) => r.dataset.pair);
}

function initTreeDropdown(root, onCommit) {
  const onClick = (e) => {
    const x = root.contains(e.target) && e.target.closest(".tree-control .value-x");
    if (x) {
      e.preventDefault(); // remove the chip, don't open the dropdown
      const v = x.dataset.value; // "Product=*" removes the whole product, "Product=Type" one pair
      const keep = treeSelected(x.closest("details.tree-dd")).filter((p) => !(v.endsWith("=*") ? p.startsWith(v.slice(0, -1)) : p === v));
      onCommit(keep);
      return;
    }
    const productCheck = root.contains(e.target) && e.target.closest("summary .check");
    const child = root.contains(e.target) && e.target.closest(".tree-row.child");
    if (productCheck) {
      e.preventDefault(); // tick, don't expand/collapse
      if (!productCheck.disabled) {
        const group = productCheck.closest(".tree-group");
        const state = productCheck.classList.contains("on") ? "off" : "on"; // unticked or half -> all feedback types
        group.querySelectorAll(".tree-row.child .check").forEach((c) => treeSetCheck(c, state));
        treeSyncProduct(group);
      }
    } else if (child && !child.disabled) {
      const check = child.querySelector(".check");
      treeSetCheck(check, check.classList.contains("on") ? "off" : "on");
      treeSyncProduct(child.closest(".tree-group"));
    }
    root.querySelectorAll("details.tree-dd[open]").forEach((d) => {
      if (!d.contains(e.target)) d.open = false;
    });
  };
  const onKey = (e) => {
    if (e.key === "Escape") root.querySelectorAll("details.tree-dd[open]").forEach((d) => (d.open = false));
  };
  // "toggle" doesn't bubble, so listen in the capture phase
  const onToggle = (e) => {
    const dd = e.target;
    if (!dd.matches || !dd.matches("details.tree-dd")) return;
    if (dd.open) {
      dd.dataset.before = JSON.stringify(treeSelected(dd));
    } else {
      const now = treeSelected(dd);
      if (JSON.stringify(now) !== dd.dataset.before) onCommit(now);
    }
  };
  // clicks/keys are watched on the whole document so "click outside" closes the dropdown
  document.addEventListener("click", onClick);
  document.addEventListener("keydown", onKey);
  root.addEventListener("toggle", onToggle, true);
  return () => {
    document.removeEventListener("click", onClick);
    document.removeEventListener("keydown", onKey);
    root.removeEventListener("toggle", onToggle, true);
  };
}

if (typeof STREAMLIT_TREE === "undefined") {
  initTreeDropdown(document, (pairs) => window.dash_clientside.set_props("pf", { data: pairs }));
}
