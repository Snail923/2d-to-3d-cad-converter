import { Viewer3D } from "./viewer3d.js";

const SVGNS = "http://www.w3.org/2000/svg";
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const BASE = new URL(".", location.href).href;

// ------------------------------------------------------------------ worker RPC
const _py = new URLSearchParams(location.search).get("pyodide");
let worker = null;
let rpcId = 0;
const pending = new Map();
function startWorker() {
  if (worker) worker.terminate();
  worker = new Worker("worker.js" + (_py ? "?pyodide=" + encodeURIComponent(_py) : ""));
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === "progress") { setLoading(m.msg); return; }
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
  };
}
startWorker();
// A native (GEOS) crash leaves the Python runtime unusable: start a fresh engine.
async function recoverEngine() {
  for (const p of pending.values()) p.reject(new Error("engine restarted"));
  pending.clear();
  startWorker();
  await call("init", { base: BASE });
  setLoading(null);
}
function call(cmd, args = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = ++rpcId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, args }, transfer);
  });
}

// ------------------------------------------------------------------ state
const S = {
  data: null, model: null, view: null, byId: new Map(), selected: null,
  corrections: emptyCorr(), history: [], fileName: null, fileBytes: null, sampleRef: null,
  tool: "select", draft: [], vb: null, samples: [], busy: false,
};
function emptyCorr() { return { version: 1, status_overrides: {}, added_walls: [], room_overrides: [], opening_type_overrides: {} }; }

function setStatus(t) { $("#status").textContent = t; }
function setLoading(msg) { if (msg === null) { $("#loading").classList.add("hidden"); return; } $("#loading").classList.remove("hidden"); $("#loadingMsg").textContent = msg; }
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function fmt(v, d = 2) { return v === null || v === undefined ? "<span class='muted'>unknown</span>" : (typeof v === "number" ? v.toFixed(d) : esc(v)); }
function download(name, data, type) {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ------------------------------------------------------------------ boot
(async function boot() {
  try {
    const info = await call("init", { base: BASE });
    S.samples = info.samples || [];
    const sel = $("#sample");
    for (const s of S.samples) {
      const o = document.createElement("option");
      o.value = s.file; o.textContent = s.title;
      sel.appendChild(o);
    }
    $("#build").textContent = `· engine ${info.version} · built ${info.built}`;
    setLoading(null);
    setStatus("Ready. Open a DXF file or load a sample.");
    renderBenchmark();
    const q = new URLSearchParams(location.search).get("sample");
    if (q) { sel.value = q; loadSample(); }
  } catch (e) {
    setLoading("Failed to start the Python engine: " + e.message + " — reload the page to retry.");
  }
})();

// ------------------------------------------------------------------ loading drawings
$("#file").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  if (/\.dwg$/i.test(f.name)) {
    alert("DWG files must be converted to DXF first (ODA File Converter is a desktop program and cannot run in a browser).\n\n" +
      "Use the command line version of this tool (python -m architectural_converter file.dwg), or convert with ODA File Converter / your CAD program (Save As DXF).");
    return;
  }
  S.sampleRef = null;
  await runFile(f.name, await f.arrayBuffer());
  ev.target.value = "";
});
$("#loadSample").addEventListener("click", loadSample);
async function loadSample() {
  const file = $("#sample").value;
  if (!file) return;
  const s = S.samples.find((x) => x.file === file);
  const buf = await (await fetch(BASE + "samples/" + file)).arrayBuffer();
  S.sampleRef = s && s.reference ? s.reference : null;
  await runFile(file, buf);
}

async function runFile(name, buf) {
  if (S.busy) return;
  S.busy = true;
  S.fileName = name; S.fileBytes = buf.slice(0);
  S.corrections = emptyCorr(); S.history = [];
  setLoading(`Analysing ${name}… (ingestion → detection → reconstruction → topology → validation → 3D)`);
  const t0 = performance.now();
  try {
    const res = await call("process", { name, bytes: buf }, [buf]);
    load(JSON.parse(res));
    setStatus(`${name}: processed in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  } catch (e) {
    setStatus("Error: " + e.message + " — restarting the engine…");
    alert("Processing failed:\n" + e.message + "\n\nThe engine will be restarted. Please report the drawing if this repeats.");
    try { await recoverEngine(); setStatus("Engine restarted after an error. You can load another drawing."); } catch (e2) { setStatus("Engine restart failed: " + e2.message); }
  } finally {
    setLoading(null);
    S.busy = false;
  }
}

async function rerun() {
  if (!S.model || S.busy) return;
  S.busy = true;
  setLoading("Re-running resolution, reconstruction, topology and validation with your corrections…");
  try {
    const t0 = performance.now();
    load(JSON.parse(await call("rerun", { corrections: S.corrections })), true);
    setStatus(`Re-run with ${countCorr()} correction(s) in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  } catch (e) { alert("Re-run failed: " + e.message); }
  finally { setLoading(null); S.busy = false; }
}

function load(data, keepView = false) {
  S.data = data; S.model = data.model; S.view = data.view;
  S.byId = new Map();
  const m = S.model;
  for (const w of m.walls) S.byId.set(w.id, { kind: "wall", obj: w });
  for (const o of m.openings) S.byId.set(o.id, { kind: "opening", obj: o });
  for (const r of m.rooms) S.byId.set(r.id, { kind: "room", obj: r });
  for (const c of m.columns) S.byId.set(c.id, { kind: "column", obj: c });
  for (const h of m.hypotheses.walls) S.byId.set(h.id, { kind: "hypothesis", obj: h });
  for (const r of m.hypotheses.rooms) S.byId.set(r.id, { kind: "room_hypothesis", obj: r });
  $("#empty2d").style.display = "none";
  draw2d(keepView);
  renderLayers();
  renderIssues();
  renderInfo();
  renderCorr();
  viewer3d.setData(data.mesh, S.model);
  const d = data.mesh.metadata.defaults_used || {};
  $("#defaults3d").textContent = Object.keys(d).length ? "Defaults applied (not in drawing): " +
    Object.entries(d).map(([k, v]) => `${k}=${v.value}`).join(", ") : "";
  if (S.selected && S.byId.has(S.selected)) select(S.selected); else { S.selected = null; $("#details").innerHTML = "<span class='muted'>Click a wall, opening, room, column or hypothesis.</span>"; }
}

// ------------------------------------------------------------------ 2D drawing
const svg = $("#svg");
function el(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
const P = (p) => `${p[0]},${-p[1]}`;
function pathD(pts, close = false) { return pts.length ? "M" + pts.map(P).join("L") + (close ? "Z" : "") : ""; }
function polyD(poly) {
  if (!poly) return "";
  let d = pathD(poly.exterior, true);
  for (const h of poly.holes || []) d += pathD(h, true);
  return d;
}

function draw2d(keepView) {
  svg.innerHTML = "";
  const world = el("g", { id: "world" }, svg);
  const v = S.view, m = S.model;
  const gH = el("g", { class: "hatch" }, world);
  for (const h of v.hatches) for (const lp of h.loops) el("path", { d: pathD(lp, true), "data-r": h.r }, gH);
  const gRooms = el("g", { class: "rooms" }, world);
  for (const r of m.hypotheses.rooms) if (r.polygon) el("path", { class: "hyproom", d: polyD(r.polygon) }, gRooms);
  for (const r of m.rooms) {
    if (r.status === "REJECTED" || !r.polygon) continue;
    el("path", { class: `el room st-${r.status}`, d: polyD(r.polygon), "data-id": r.id, "fill-rule": "evenodd" }, gRooms);
  }
  const gD = el("g", { class: "dxf" }, world);
  for (const l of v.lines) el("path", { d: pathD(l.p), "data-r": l.r, class: "role-" + l.k }, gD);
  const gT = el("g", { class: "dxftexts" }, world);
  for (const t of v.texts) {
    const tx = el("text", { class: "dxftext", x: t.p[0], y: -t.p[1], "font-size": Math.max(t.h, 0.12), "text-anchor": "middle" }, gT);
    tx.textContent = t.t.split("\n")[0];
  }
  const gW = el("g", { class: "walls" }, world);
  const gAx = el("g", { class: "axes" }, world);
  const order = { REJECTED: 0, NEEDS_REVIEW: 1, ACCEPTED: 2 };
  for (const w of [...m.walls].sort((a, b) => order[a.status] - order[b.status])) {
    const g = w.geometry;
    const cls = `el wall cls-${w.classification || "none"} st-${w.status}`;
    if (g.polygon) el("path", { class: cls, d: polyD(g.polygon), "data-id": w.id }, gW);
    else el("path", { class: cls, d: pathD([g.start, g.end]), "data-id": w.id, "stroke-width": 4, fill: "none" }, gW);
    const ax = g.start ? [g.start, g.end] : g.axis;
    if (ax) el("path", { class: `axis st-${w.status} cls-${w.classification || "none"}`, d: pathD(ax) }, gAx);
  }
  const gC = el("g", { class: "columns" }, world);
  for (const c of m.columns) if (c.polygon) el("path", { class: `el column st-${c.status}`, d: polyD(c.polygon), "data-id": c.id }, gC);
  const gO = el("g", { class: "openings" }, world);
  for (const o of m.openings) {
    const p = o.position;
    el("path", { class: `el opening t-${o.type} st-${o.status}`, d: pathD([p.start, p.end]), "data-id": o.id }, gO);
  }
  const gHy = el("g", { class: "hyps" }, world);
  for (const h of m.hypotheses.walls) {
    const g = h.geometry;
    el("path", { class: "el hyp", d: pathD(g.start ? [g.start, g.end] : g.axis), "data-id": h.id }, gHy);
  }
  const gL = el("g", { class: "roomlabels" }, world);
  for (const r of m.rooms) {
    if (r.status === "REJECTED" || !r.centroid) continue;
    const t = el("text", { class: "roomlabel", x: r.centroid[0], y: -r.centroid[1], "font-size": 0.32 }, gL);
    t.textContent = `${r.name || r.classification || "room"} · ${r.area.toFixed(1)} m²`;
  }
  const gI = el("g", { class: "issues" }, world);
  for (const i of m.validation.issues) {
    if (!i.location || i.severity === "info") continue;
    el("circle", { class: `issue sev-${i.severity}`, cx: i.location[0], cy: -i.location[1], r: 0.12, "data-issue": i.id }, gI);
  }
  el("g", { id: "pending" }, world);
  el("g", { id: "draft" }, world);
  drawPending();
  applyLayerClasses();
  if (!keepView || !S.vb) fit();
  else setVB(S.vb);
}

function fit() {
  const b = S.view.bounds;
  const pad = Math.max(b[2] - b[0], b[3] - b[1]) * 0.05 + 0.5;
  const r = svg.getBoundingClientRect();
  let w = b[2] - b[0] + 2 * pad, h = b[3] - b[1] + 2 * pad;
  const aspect = r.width / Math.max(r.height, 1);
  if (w / h < aspect) w = h * aspect; else h = w / aspect;
  const cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
  setVB([cx - w / 2, -cy - h / 2, w, h]);
}
function setVB(vb) { S.vb = vb; svg.setAttribute("viewBox", vb.join(" ")); }
$("#fit").addEventListener("click", () => S.view && fit());
window.addEventListener("resize", () => S.view && S.vb && fit());

function toWorld(ev) {
  const pt = svg.createSVGPoint();
  pt.x = ev.clientX; pt.y = ev.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  return [p.x, -p.y];
}
function pxToWorld(px) { return S.vb ? px * S.vb[2] / svg.getBoundingClientRect().width : px; }

// pan / zoom
let drag = null;
svg.addEventListener("wheel", (ev) => {
  if (!S.vb) return;
  ev.preventDefault();
  const f = Math.exp(ev.deltaY * 0.0015);
  const [x, y] = toWorld(ev);
  const [vx, vy, vw, vh] = S.vb;
  const nx = x - (x - vx) * f, ny = -y - (-y - vy) * f;
  setVB([nx, ny, vw * f, vh * f]);
}, { passive: false });
svg.addEventListener("mousedown", (ev) => {
  if (ev.button === 1 || ev.button === 2 || (ev.button === 0 && S.tool === "select")) {
    drag = { x: ev.clientX, y: ev.clientY, vb: S.vb.slice(), moved: false };
  }
});
window.addEventListener("mousemove", (ev) => {
  if (S.vb && ev.target.closest && ev.target.closest("#svg")) {
    const [x, y] = toWorld(ev);
    $("#cursor").textContent = `x ${x.toFixed(3)}  y ${y.toFixed(3)} m`;
    if (S.tool !== "select") drawDraft([x, y]);
  }
  if (!drag) return;
  const r = svg.getBoundingClientRect();
  const dx = (ev.clientX - drag.x) * drag.vb[2] / r.width, dy = (ev.clientY - drag.y) * drag.vb[3] / r.height;
  if (Math.abs(ev.clientX - drag.x) + Math.abs(ev.clientY - drag.y) > 3) drag.moved = true;
  setVB([drag.vb[0] - dx, drag.vb[1] - dy, drag.vb[2], drag.vb[3]]);
});
window.addEventListener("mouseup", () => { setTimeout(() => (drag = null), 0); });
svg.addEventListener("contextmenu", (ev) => ev.preventDefault());

svg.addEventListener("click", (ev) => {
  if (drag && drag.moved) return;
  if (!S.model) return;
  if (S.tool === "select") {
    const t = ev.target.closest("[data-id]");
    if (t) select(t.getAttribute("data-id"));
    else {
      const iss = ev.target.closest("[data-issue]");
      if (iss) showIssue(iss.getAttribute("data-issue"));
    }
    return;
  }
  const p = snap(toWorld(ev));
  if (S.tool === "addwall" || S.tool === "connect") {
    S.draft.push(p);
    if (S.draft.length === 2) {
      const [a, b] = S.draft;
      let t = parseFloat($("#thick").value);
      if (S.tool === "connect") {
        const ts = [nearestWall(a), nearestWall(b)].filter((w) => w && w.thickness).map((w) => w.thickness);
        if (ts.length) t = ts.reduce((x, y) => x + y) / ts.length;
      }
      addCorr(() => S.corrections.added_walls.push({ start: a, end: b, thickness: isFinite(t) && t > 0 ? +t.toFixed(4) : null,
        note: S.tool === "connect" ? "manual endpoint connection" : "manual wall" }));
      S.draft = [];
    }
  } else if (S.tool === "room") {
    S.draft.push(p);
  }
  drawDraft(p);
});
svg.addEventListener("dblclick", (ev) => { if (S.tool === "room") finishRoom(); });
window.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") { S.draft = []; drawDraft(null); }
  if (ev.key === "Enter" && S.tool === "room") finishRoom();
});
function finishRoom() {
  const pts = dedupe(S.draft);
  S.draft = [];
  drawDraft(null);
  if (pts.length < 3) return;
  const name = prompt("Room name (optional):", "") || null;
  addCorr(() => S.corrections.room_overrides.push({ polygon: pts, name }));
}
function dedupe(pts) { return pts.filter((p, i) => i === 0 || Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) > 1e-6); }

function snapPoints() {
  const out = [];
  for (const w of S.model.walls) if (w.status !== "REJECTED" && w.geometry.start) out.push(w.geometry.start, w.geometry.end);
  for (const l of S.view.lines) { out.push(l.p[0], l.p[l.p.length - 1]); }
  return out;
}
function snap(p) {
  const tolW = pxToWorld(10);
  let best = null, bd = tolW;
  for (const q of snapPoints()) {
    const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (d < bd) { bd = d; best = q; }
  }
  if (best) return [+best[0].toFixed(4), +best[1].toFixed(4)];
  if (S.draft.length && S.tool !== "room") {  // orthogonal assist
    const a = S.draft[S.draft.length - 1];
    const dx = Math.abs(p[0] - a[0]), dy = Math.abs(p[1] - a[1]);
    if (dy < pxToWorld(8)) return [p[0], a[1]];
    if (dx < pxToWorld(8)) return [a[0], p[1]];
  }
  return [+p[0].toFixed(4), +p[1].toFixed(4)];
}
function nearestWall(p) {
  let best = null, bd = Infinity;
  for (const w of S.model.walls) {
    if (w.status === "REJECTED" || !w.geometry.start) continue;
    for (const q of [w.geometry.start, w.geometry.end]) {
      const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
      if (d < bd) { bd = d; best = w; }
    }
  }
  return best;
}
function drawDraft(cur) {
  const g = $("#draft");
  if (!g) return;
  g.innerHTML = "";
  const pts = S.draft.slice();
  if (cur && S.tool !== "select") {
    const s = snap(cur);
    el("circle", { class: "snap", cx: s[0], cy: -s[1], r: pxToWorld(4) }, g);
    if (pts.length) pts.push(s);
  }
  if (pts.length >= 2) el("path", { class: "rubber", d: pathD(pts, S.tool === "room" && pts.length > 2) }, g);
}
function drawPending() {
  const g = $("#pending");
  if (!g) return;
  g.innerHTML = "";
  const applied = (S.model && S.model.corrections) || emptyCorr();
  const same = JSON.stringify(applied.added_walls) === JSON.stringify(S.corrections.added_walls) &&
    JSON.stringify(applied.room_overrides) === JSON.stringify(S.corrections.room_overrides);
  if (same) return;
  for (const w of S.corrections.added_walls) el("path", { class: "pending", d: pathD([w.start, w.end]) }, g);
  for (const r of S.corrections.room_overrides) el("path", { class: "pending", d: pathD(r.polygon, true) }, g);
}

// tools
for (const b of $$(".tool")) b.addEventListener("click", () => {
  $$(".tool").forEach((x) => x.classList.toggle("active", x === b));
  S.tool = b.dataset.tool; S.draft = []; drawDraft(null);
  svg.className.baseVal = svg.className.baseVal.replace(/tool-\S+/g, "") + " tool-" + S.tool;
  const hints = { select: "", addwall: "Click the start and end point of the missing wall (snaps to end points). Esc cancels.",
    connect: "Click two wall end points to connect them with a wall of their average thickness.",
    room: "Click the room corners, double-click or press Enter to finish. Esc cancels." };
  $("#hint").textContent = hints[S.tool]; $("#hint").style.display = hints[S.tool] ? "block" : "none";
  applyLayerClasses();
});

// ------------------------------------------------------------------ layers
const LAYERS = [
  ["dxf", "Original DXF linework", "#a9b0bd"], ["hatch", "DXF hatches", "#c9b79c"], ["dxftext", "DXF text", "#8b92a1"],
  ["rooms", "Rooms", "#ffe9a8"], ["roomlabels", "Room labels", "#8a6d00"], ["exterior", "Exterior walls", "#1d4f91"],
  ["interior", "Interior walls", "#2f7d32"], ["partition", "Partition walls", "#7cb342"], ["review", "Uncertain walls", "#f2a33a"],
  ["rejected", "Rejected candidates", "#d32f2f"], ["doors", "Doors", "#ef6c00"], ["windows", "Windows", "#0288d1"],
  ["openings", "Unidentified openings", "#8e24aa"], ["columns", "Columns", "#6a1b9a"], ["hyps", "Missing-wall hypotheses", "#ff00aa"],
  ["issues", "Validation markers", "#ff9100"], ["axes", "Wall centre lines", "#555"],
];
const hidden = new Set(["rejected", "dxftext"]);
function renderLayers() {
  const box = $("#layers");
  box.innerHTML = "";
  const m = S.model;
  const cnt = {
    exterior: m.walls.filter((w) => w.classification === "exterior_wall" && w.status !== "REJECTED").length,
    interior: m.walls.filter((w) => w.classification === "interior_wall" && w.status !== "REJECTED").length,
    partition: m.walls.filter((w) => w.classification === "partition_wall" && w.status !== "REJECTED").length,
    review: m.walls.filter((w) => w.status === "NEEDS_REVIEW").length, rejected: m.walls.filter((w) => w.status === "REJECTED").length,
    doors: m.openings.filter((o) => o.type === "door" && o.status !== "REJECTED").length,
    windows: m.openings.filter((o) => o.type === "window" && o.status !== "REJECTED").length,
    openings: m.openings.filter((o) => o.type === "opening" && o.status !== "REJECTED").length,
    rooms: m.rooms.filter((r) => r.status !== "REJECTED").length, columns: m.columns.filter((c) => c.status !== "REJECTED").length,
    hyps: m.hypotheses.walls.length, issues: m.validation.issues.filter((i) => i.severity !== "info").length,
  };
  for (const [key, label, color] of LAYERS) {
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" ${hidden.has(key) ? "" : "checked"}> <span class="sw" style="background:${color}"></span> ${label} ${cnt[key] !== undefined ? `<span class="muted">(${cnt[key]})</span>` : ""}`;
    l.querySelector("input").addEventListener("change", (e) => { e.target.checked ? hidden.delete(key) : hidden.add(key); applyLayerClasses(); });
    box.appendChild(l);
  }
}
function applyLayerClasses() {
  const base = svg.className.baseVal.split(/\s+/).filter((c) => c && !c.startsWith("hide-"));
  for (const k of hidden) base.push("hide-" + k);
  svg.className.baseVal = base.join(" ");
}

// ------------------------------------------------------------------ selection / details
function select(id) {
  S.selected = id;
  $$("#svg .selected").forEach((e) => e.classList.remove("selected"));
  $$("#svg .hl").forEach((e) => e.classList.remove("hl"));
  const e = $(`#svg [data-id="${CSS.escape(id)}"]`);
  if (e) e.classList.add("selected");
  const it = S.byId.get(id);
  if (!it) return;
  const o = it.obj;
  const srcIds = new Set((o.source_entities || []).map((s) => (typeof s === "string" ? s : s.id)));
  for (const p of $$("#svg .dxf path")) if (srcIds.has(p.getAttribute("data-r"))) p.classList.add("hl");
  viewer3d.highlight(id);
  renderDetails(it.kind, o);
}

function evidenceRows(ev) {
  return Object.entries(ev || {}).map(([k, v]) =>
    `<tr><td>${esc(k)}</td><td><div class="bar"><span style="width:${Math.round(v * 100)}%"></span></div><span class="small mono">${v.toFixed(2)}</span></td></tr>`).join("");
}
function sourceRows(src) {
  return (src || []).slice(0, 40).map((s) => typeof s === "string"
    ? `<tr><td colspan=2 class="mono small">${esc(s)}</td></tr>`
    : `<tr><td class="mono small">${esc(s.type)} #${esc(s.handle || "—")}</td><td class="small">layer <b>${esc(s.layer)}</b>${s.block ? ` · block ${esc(s.block)}` : ""}</td></tr>`).join("");
}
function renderDetails(kind, o) {
  const ov = S.corrections.status_overrides[o.id];
  let props = "";
  if (kind === "wall" || kind === "hypothesis") {
    const g = o.geometry;
    props = `<tr><td>kind</td><td>${esc(g.kind)}</td></tr><tr><td>length</td><td>${fmt(g.length)} m</td></tr>
      <tr><td>thickness</td><td>${fmt(o.thickness, 3)} m <span class="muted">(${esc(o.thickness_source)})</span></td></tr>
      <tr><td>height</td><td>${fmt(o.height)}</td></tr>
      <tr><td>openings</td><td>${(o.openings || []).map((x) => `<a href="#" data-goto="${x}">${x}</a>`).join(", ") || "—"}</td></tr>
      <tr><td>connected walls</td><td>${(o.connections || []).length}</td></tr>
      <tr><td>bounds rooms</td><td>${(o.rooms || []).map((x) => `<a href="#" data-goto="${x}">${x}</a>`).join(", ") || "—"}</td></tr>`;
    const rec = o.reconstruction || {};
    if (rec.extensions) props += `<tr><td>reconstruction</td><td class="small">${rec.extensions.map((e) => `extended ${e.end} by ${e.length ?? e.angle_rad} (${e.junction}${e.to_wall ? " → " + e.to_wall : ""})`).join("<br>")}</td></tr>`;
    if (rec.merged) props += `<tr><td>merged</td><td class="small">${rec.merged.map((m) => `${m.wall} (${esc(m.reason)})`).join("<br>")}</td></tr>`;
    if (rec.face_support) props += `<tr><td>face support</td><td class="small">left ${rec.face_support.left} · right ${rec.face_support.right}</td></tr>`;
    if (rec.hypothesis) props += `<tr><td>hypothesis</td><td class="small">${esc(rec.hypothesis.reason)} from ${esc(rec.hypothesis.from_wall)}</td></tr>`;
  } else if (kind === "opening") {
    props = `<tr><td>type</td><td>${esc(o.type)} · ${esc(o.subtype)}</td></tr><tr><td>width</td><td>${fmt(o.width)} m</td></tr>
      <tr><td>height / sill</td><td>${fmt(o.height)} / ${fmt(o.sill_height)}</td></tr>
      <tr><td>host wall</td><td>${o.host_wall ? `<a href="#" data-goto="${o.host_wall}">${o.host_wall}</a>` : "<b>none</b>"}</td></tr>
      <tr><td>rooms</td><td>${(o.rooms || []).join(", ") || "—"}</td></tr>`;
  } else if (kind === "room" || kind === "room_hypothesis") {
    const bs = o.boundary_support || {};
    props = `<tr><td>name</td><td>${esc(o.name || "—")}</td></tr><tr><td>area</td><td>${fmt(o.area)} m²</td></tr>
      <tr><td>perimeter</td><td>${fmt(o.perimeter)} m</td></tr><tr><td>area label</td><td>${fmt(o.area_text)}</td></tr>
      <tr><td>boundary support</td><td class="small">${Object.entries(bs).map(([k, v]) => `${k}: ${(+v).toFixed(2)}`).join("<br>")}</td></tr>
      <tr><td>walls / openings</td><td>${(o.bounding_walls || []).length} / ${(o.openings || []).length}</td></tr>
      <tr><td>adjacent</td><td>${(o.adjacent_rooms || []).map((x) => `<a href="#" data-goto="${x}">${x}</a>`).join(", ") || "—"}</td></tr>`;
  } else if (kind === "column") {
    props = `<tr><td>shape</td><td>${esc(o.shape)}</td></tr><tr><td>size</td><td>${fmt(o.size[0])} × ${fmt(o.size[1])} m</td></tr>`;
  }
  let buttons = "";
  if (kind === "hypothesis") {
    buttons = `<button class="btn primary" id="accHyp">Accept as wall</button>`;
  } else if (kind !== "room_hypothesis") {
    buttons = `<button class="btn" data-set="ACCEPTED">Accept</button> <button class="btn" data-set="REJECTED">Reject</button>
      ${ov ? `<button class="btn" data-set="">Reset (${ov})</button>` : ""}`;
    if (kind === "opening" && o.type === "opening") buttons += `<div class="row"><button class="btn small" data-otype="door">→ door</button><button class="btn small" data-otype="window">→ window</button></div>`;
  }
  $("#details").innerHTML = `
    <div><b>${esc(o.id)}</b> · ${esc(kind)} ${o.classification ? "· " + esc(o.classification) : ""}</div>
    <div style="margin:4px 0"><span class="pill ${o.status}">${o.status}</span> confidence <b>${(o.confidence || 0).toFixed(2)}</b>
      ${o.status_reason ? `<div class="small muted">${esc(o.status_reason)}</div>` : ""}</div>
    <div>${buttons}</div>
    <table>${props}</table>
    <div class="small"><b>Detection methods:</b> ${(o.detection_methods || []).join(", ") || "—"}</div>
    ${o.notes && o.notes.length ? `<div class="small muted">${o.notes.map(esc).join("<br>")}</div>` : ""}
    <h3>Evidence</h3><table>${evidenceRows(o.evidence)}</table>
    <h3>Source DXF entities (${(o.source_entities || []).length})</h3><table>${sourceRows(o.source_entities)}</table>`;
  for (const b of $$("#details [data-set]")) b.addEventListener("click", () => addCorr(() => {
    if (b.dataset.set) S.corrections.status_overrides[o.id] = b.dataset.set; else delete S.corrections.status_overrides[o.id];
  }, true));
  for (const b of $$("#details [data-otype]")) b.addEventListener("click", () => addCorr(() => { S.corrections.opening_type_overrides[o.id] = b.dataset.otype; }, true));
  for (const a of $$("#details [data-goto]")) a.addEventListener("click", (e) => { e.preventDefault(); select(a.dataset.goto); });
  const acc = $("#accHyp");
  if (acc) acc.addEventListener("click", () => addCorr(() => {
    const g = o.geometry;
    S.corrections.added_walls.push({ start: g.start, end: g.end, thickness: o.thickness, note: `accepted hypothesis ${o.id}` });
  }, true));
}

// ------------------------------------------------------------------ corrections
function countCorr() {
  const c = S.corrections;
  return Object.keys(c.status_overrides).length + c.added_walls.length + c.room_overrides.length + Object.keys(c.opening_type_overrides).length;
}
function addCorr(fn, autoRerun = false) {
  S.history.push(JSON.stringify(S.corrections));
  fn();
  renderCorr();
  drawPending();
  if (autoRerun) rerun();
}
function renderCorr() {
  $("#corrCount").textContent = countCorr();
  $("#corrView").textContent = countCorr() ? JSON.stringify(S.corrections, null, 1) : "No corrections yet.";
}
$("#rerun").addEventListener("click", rerun);
$("#undoCorr").addEventListener("click", () => { if (S.history.length) { S.corrections = JSON.parse(S.history.pop()); renderCorr(); drawPending(); } });
$("#clearCorr").addEventListener("click", () => { S.history.push(JSON.stringify(S.corrections)); S.corrections = emptyCorr(); renderCorr(); drawPending(); });
$("#dlCorr").addEventListener("click", () => download((S.fileName || "drawing") + ".corrections.json", JSON.stringify(S.corrections, null, 1), "application/json"));
$("#ulCorr").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  S.history.push(JSON.stringify(S.corrections));
  S.corrections = Object.assign(emptyCorr(), JSON.parse(await f.text()));
  renderCorr(); ev.target.value = ""; rerun();
});

// ------------------------------------------------------------------ issues / info
function renderIssues() {
  const sev = new Set($$(".sevf").filter((c) => c.checked).map((c) => c.value));
  const iss = S.model.validation.issues;
  $("#issueCount").textContent = iss.filter((i) => i.severity !== "info").length;
  const rows = iss.filter((i) => sev.has(i.severity)).map((i) =>
    `<tr class="${i.severity} clickable" data-issue="${i.id}"><td>${i.id}</td><td>${i.severity}</td><td>${esc(i.code)}</td><td>${esc(i.entity_id || "")}</td>
     <td>${esc(i.description)}</td><td>${esc(i.suggestion)}</td><td class="num">${(+i.confidence).toFixed(2)}</td>
     <td class="small">${i.source_entities.slice(0, 3).map((s) => `${esc(s.type)}#${esc(s.handle || "-")}@${esc(s.layer)}`).join("<br>")}</td></tr>`).join("");
  $("#issues").innerHTML = `<tr><th>id</th><th>severity</th><th>code</th><th>element</th><th>problem</th><th>suggested correction</th><th>conf.</th><th>source entities</th></tr>${rows}`;
  for (const tr of $$("#issues tr[data-issue]")) tr.addEventListener("click", () => { showTab("tab2d"); showIssue(tr.dataset.issue); });
}
for (const c of $$(".sevf")) c.addEventListener("change", () => S.model && renderIssues());
function showIssue(id) {
  const i = S.model.validation.issues.find((x) => x.id === id);
  if (!i) return;
  if (i.location) {
    const w = 6, r = svg.getBoundingClientRect(), h = w * r.height / r.width;
    setVB([i.location[0] - w / 2, -i.location[1] - h / 2, w, h]);
  }
  if (i.entity_id && S.byId.has(i.entity_id)) select(i.entity_id);
  setStatus(`${i.code}: ${i.description}`);
}
function renderInfo() {
  const m = S.model, src = m.source, st = m.statistics, ctx = S.data.context;
  const logs = st.logs || {};
  const tol = ctx.tolerances;
  $("#info").innerHTML = `
    <h3>Source</h3><table class="grid">
      <tr><td>file</td><td>${esc(src.file)}</td></tr>
      <tr><td>units</td><td>${esc(src.dxf_units)} (confidence ${src.unit_confidence}, $INSUNITS=${src.insunits_header})</td></tr>
      <tr><td>transform</td><td class="mono small">${esc(src.transform.formula)} · scale ${src.transform.scale_to_m} · offset ${src.transform.offset_drawing_units.join(", ")}</td></tr>
      <tr><td>raw entities</td><td class="mono small">${esc(JSON.stringify(src.summary.raw_entity_counts))}</td></tr>
      <tr><td>layers / blocks / inserts</td><td>${src.summary.layers} / ${src.summary.blocks} / ${src.summary.inserts}</td></tr></table>
    <h3>Result</h3><table class="grid"><tr><td>counts by status</td><td class="mono small">${esc(JSON.stringify(st.counts))}</td></tr>
      <tr><td>timings (s)</td><td class="mono small">${esc(JSON.stringify(st.timings_s))}</td></tr></table>
    <h3>Adaptive tolerances (metres)</h3><table class="grid">
      <tr><td>duplicate</td><td>${tol.duplicate.toExponential(2)}</td></tr><tr><td>endpoint snap</td><td>${tol.snap.toFixed(4)}</td></tr>
      <tr><td>intersection</td><td>${tol.intersection.toFixed(4)}</td></tr><tr><td>parallel angle</td><td>${tol.angle_deg.toFixed(2)}° (max ${tol.angle_max_deg.toFixed(1)}°)</td></tr>
      <tr><td>thickness search</td><td>${tol.thickness_min} – ${tol.thickness_max}</td></tr>
      <tr><td>evidence</td><td class="mono small">${esc(JSON.stringify(tol.evidence))}</td></tr></table>
    <h3>Wall thickness classes found in this drawing</h3><table class="grid"><tr><th>thickness (m)</th><th>share</th><th>pieces</th><th>significant</th></tr>
      ${ctx.thickness_clusters.slice(0, 8).map((c) => `<tr><td class="num">${c.thickness.toFixed(3)}</td><td class="num">${(c.share * 100).toFixed(1)}%</td><td class="num">${c.count}</td><td>${c.significant ? "yes" : "no"}</td></tr>`).join("")}</table>
    <h3>Layer semantics (hints)</h3><table class="grid"><tr><th>layer</th><th>hint</th><th>wall evidence</th><th>entities</th><th>parallel share</th></tr>
      ${Object.values(S.view.layers).filter((l) => Object.keys(l.entity_counts).length).map((l) => `<tr><td>${esc(l.name)}</td><td>${esc(l.primary || "—")}</td><td class="num">${(+l.wall_evidence).toFixed(2)}</td><td class="mono small">${esc(JSON.stringify(l.entity_counts))}</td><td class="num">${(+l.parallel_share).toFixed(2)}</td></tr>`).join("")}</table>
    <h3>Block symbols</h3><div class="mono small">${esc(JSON.stringify(ctx.symbol_roles))}</div>
    <h3>Stage logs</h3><pre class="mono small">${esc(JSON.stringify(logs, null, 1))}</pre>`;
}

// ------------------------------------------------------------------ benchmark
async function renderBenchmark() {
  let html = `<p>The engine is regression-tested against synthetic DXF drawings generated from plan specifications, so the
    reference geometry is exact. Wall precision/recall are length-based (centre lines within max(0.6·t, 8 cm)); rooms match at IoU ≥ 0.5.
    <b>These drawings are synthetic</b> — add your own drawings with reference geometry to measure accuracy on real projects.</p>`;
  try {
    const b = await (await fetch(BASE + "benchmark.json", { cache: "no-cache" })).json();
    const keys = ["wall_precision", "wall_recall", "room_precision", "room_recall", "room_boundary_iou", "door_precision", "door_recall",
      "window_precision", "window_recall", "false_positive_walls", "missed_walls", "manual_corrections_required", "seconds"];
    const cls = (k, v) => typeof v !== "number" || k.includes("walls") || k.includes("corrections") || k === "seconds" ? "" : v >= 0.95 ? "good" : v >= 0.8 ? "mid" : "bad";
    html += `<table class="grid"><tr><th>drawing</th>${keys.map((k) => `<th>${k.replace(/_/g, " ")}</th>`).join("")}</tr>` +
      Object.entries(b).map(([n, m]) => `<tr><td><b>${esc(n)}</b><div class="small muted">${esc(m.description || "")}</div></td>${keys.map((k) =>
        `<td class="num ${cls(k, m[k])}">${typeof m[k] === "number" ? (Number.isInteger(m[k]) ? m[k] : m[k].toFixed(2)) : esc(m[k] ?? "")}</td>`).join("")}</tr>`).join("") + "</table>";
  } catch (e) { html += "<p class='muted'>benchmark.json not available.</p>"; }
  html += `<h3>Evaluate the loaded sample</h3><p class="small muted">Available when a benchmark sample is loaded (uses its reference geometry, including your corrections).</p>
    <button class="btn" id="evalBtn">Evaluate current result</button><div id="evalOut"></div>`;
  $("#bench").innerHTML = html;
  $("#evalBtn").addEventListener("click", async () => {
    if (!S.model || !S.sampleRef) { $("#evalOut").innerHTML = "<p class='muted'>Load a benchmark sample first.</p>"; return; }
    const ref = await (await fetch(BASE + "samples/" + S.sampleRef)).text();
    const m = JSON.parse(await call("evaluate", { reference: ref }));
    $("#evalOut").innerHTML = `<table class="grid">${Object.entries(m).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${typeof v === "number" ? (+v).toFixed(3) : esc(v)}</td></tr>`).join("")}</table>`;
  });
}

// ------------------------------------------------------------------ exports
for (const b of $$("[data-export]")) b.addEventListener("click", async () => {
  if (!S.model) return alert("Process a drawing first.");
  const base = (S.fileName || "drawing").replace(/\.[^.]+$/, "");
  const kind = b.dataset.export;
  try {
    if (kind === "json") download(base + ".model.json", await call("export_json"), "application/json");
    if (kind === "sqlite") download(base + ".model.sqlite", new Blob([await call("export_sqlite")], { type: "application/x-sqlite3" }));
    if (kind === "glb") download(base + ".glb", new Blob([await call("export_glb")], { type: "model/gltf-binary" }));
    if (kind === "report") download(base + ".report.html", await call("export_report"), "text/html");
  } catch (e) { alert("Export failed: " + e.message); }
});

// ------------------------------------------------------------------ tabs + 3D
const viewer3d = new Viewer3D($("#view3d"), (id) => { if (S.byId.has(id)) select(id); });
function showTab(id) {
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === id));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === id));
  if (id === "tab3d") viewer3d.resize();
  if (id === "tab2d" && S.view && !S.vb) fit();
}
for (const b of $$(".tabs button")) b.addEventListener("click", () => showTab(b.dataset.tab));
$("#show3dFloors").addEventListener("change", (e) => viewer3d.setVisible("floor", e.target.checked));
$("#show3dReview").addEventListener("change", (e) => viewer3d.setVisible("review", e.target.checked));
$("#show3dGlass").addEventListener("change", (e) => viewer3d.setVisible("window", e.target.checked));
