// Products / Feedback tree dropdown (native <details>), Excel-filter style.
// Ticking is instant and local; the selection is sent to the "pf" store once, when the dropdown closes.
const ARIA = { on: "true", half: "mixed", off: "false" };

function setCheck(check, state) {
  check.classList.remove("on", "half", "off");
  check.classList.add(state);
  check.closest('[role="checkbox"]').setAttribute("aria-checked", ARIA[state]);
}

function syncProduct(group) {
  const kids = [...group.querySelectorAll(".tree-row.child .check")];
  const on = kids.filter((c) => c.classList.contains("on")).length;
  setCheck(group.querySelector("summary .check"), on === 0 ? "off" : on === kids.length ? "on" : "half");
}

function selected(dd) {
  return [...dd.querySelectorAll(".tree-row.child")].filter((r) => r.querySelector(".check.on")).map((r) => r.dataset.pair);
}

document.addEventListener("click", (e) => {
  if (e.target.closest(".tree-control .value-x")) {
    e.preventDefault(); // remove the chip (Dash handles it), don't open the dropdown
    return;
  }
  const productCheck = e.target.closest("summary .check");
  const child = e.target.closest(".tree-row.child");
  if (productCheck) {
    e.preventDefault(); // tick, don't expand/collapse
    if (!productCheck.disabled) {
      const group = productCheck.closest(".tree-group");
      const state = productCheck.classList.contains("on") ? "off" : "on"; // unticked or half -> all feedback types
      group.querySelectorAll(".tree-row.child .check").forEach((c) => setCheck(c, state));
      syncProduct(group);
    }
  } else if (child && !child.disabled) {
    const check = child.querySelector(".check");
    setCheck(check, check.classList.contains("on") ? "off" : "on");
    syncProduct(child.closest(".tree-group"));
  }
  document.querySelectorAll("details.tree-dd[open]").forEach((d) => {
    if (!d.contains(e.target)) d.open = false;
  });
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") document.querySelectorAll("details.tree-dd[open]").forEach((d) => (d.open = false));
});

// "toggle" doesn't bubble, so listen in the capture phase
document.addEventListener("toggle", (e) => {
  const dd = e.target;
  if (!dd.matches("details.tree-dd")) return;
  if (dd.open) {
    dd.dataset.before = JSON.stringify(selected(dd));
  } else {
    const now = selected(dd);
    if (JSON.stringify(now) !== dd.dataset.before) window.dash_clientside.set_props("pf", { data: now });
  }
}, true);
