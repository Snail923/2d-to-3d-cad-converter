/* Room-by-room validation by a LOCAL vision model (LM Studio, OpenAI-compatible API).

   For every room the model receives the reference drawing around it, the current result drawn over it with a
   short tag on every element, a key (colours + tag -> element), the room's metadata and the guideline
   (web/validation_guideline.md, editable in the panel). It answers with JSON in a fixed schema: a summary and
   a list of edits, each with an action from a closed list, a target tag, a confidence and a reason.
   Edits at or above the auto-apply confidence are applied to the 3D editor at once (each one an undo step
   named "AI: ..."); the others wait in the panel for a person to apply or reject. Nothing leaves the computer. */

export const VDEFAULTS = { endpoint: "http://127.0.0.1:1234", model: "", autoApply: 0.85, timeoutS: 600, imageSize: 896,
  margin: 1.5, temperature: 0, maxTokens: 6000, strict: true };
const STORE = "cad2d3d.validator";
export const ACTIONS = ["delete_wall", "add_door", "add_window", "change_opening", "remove_opening", "set_room_type", "comment"];
export const SCHEMA = {
  type: "object", additionalProperties: false, required: ["room", "analysis", "summary", "edits"],
  properties: {
    room: { type: "string" },
    // written FIRST: what the model sees, rule by rule, before it decides any edit (structured output is
    // generated in this order; without it the model has to start the edit list before looking)
    analysis: { type: "string" },
    summary: { type: "string" },
    edits: { type: "array", items: { type: "object", additionalProperties: false,
      required: ["action", "target", "position", "width", "value", "confidence", "reason"],
      properties: { action: { type: "string", enum: ACTIONS }, target: { type: "string" }, position: { type: "number" },
        width: { type: "number" }, value: { type: "string" }, confidence: { type: "number" }, reason: { type: "string" } } } },
  },
};
const COL = { exterior_wall: "#2f5fa8", interior_wall: "#3f8f3f", partition_wall: "#7cb862", railing: "#7a7f87", wall: "#3f8f3f",
  review: "#f08a24", edited: "#c2185b", door: "#8b4513", window: "#13a3d8", opening: "#8e44ad", room: "#ffe680" };
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const hyp = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);

function inPoly(q, poly) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > q[1]) !== (yj > q[1]) && q[0] < (xj - xi) * (q[1] - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}
function segDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1e-12;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2));
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}
function polyDist(a, b, poly) {                   // shortest distance between segment a-b and a polygon's outline
  let d = Infinity;
  for (let k = 0; k < poly.length; k++) {
    const p = poly[k], q = poly[(k + 1) % poly.length];
    d = Math.min(d, segDist(p, a, b), segDist(q, a, b), segDist(a, p, q), segDist(b, p, q));
  }
  return d;
}

function clipToBox(a, b, bb) {                  // Liang-Barsky: [t0, t1] of segment a-b inside the box, or null
  let t0 = 0, t1 = 1;
  const d = [b[0] - a[0], b[1] - a[1]];
  for (const [pp, q] of [[-d[0], a[0] - bb[0]], [d[0], bb[2] - a[0]], [-d[1], a[1] - bb[1]], [d[1], bb[3] - a[1]]]) {
    if (Math.abs(pp) < 1e-12) { if (q < 0) return null; continue; }
    const r = q / pp;
    if (pp < 0) t0 = Math.max(t0, r); else t1 = Math.min(t1, r);
  }
  return t1 - t0 > 1e-6 ? [t0, t1] : null;
}
function outlineShare(a, b, poly, tol) {        // share of the wall's length that runs along the room's outline
  const n = 9;
  let k = 0;
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) / n, q = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    let d = Infinity;
    for (let j = 0; j < poly.length; j++) d = Math.min(d, segDist(q, poly[j], poly[(j + 1) % poly.length]));
    if (d <= tol) k++;
  }
  return k / n;
}

export class Validator {
  constructor(editor, getData) {
    this.ed = editor;
    this.getData = getData;
    this.s = { ...VDEFAULTS };
    this.guideline = null;
    try { const st = JSON.parse(localStorage.getItem(STORE) || "{}"); Object.assign(this.s, st.settings || {}); this.guideline = st.guideline || null; }
    catch { /* private mode */ }
    // settings saved by the first version: its limits were too small for reasoning models (answers cut off)
    if (!(this.s.maxTokens >= VDEFAULTS.maxTokens)) this.s.maxTokens = VDEFAULTS.maxTokens;
    if (!(this.s.timeoutS >= VDEFAULTS.timeoutS)) this.s.timeoutS = VDEFAULTS.timeoutS;
    if (this.s.strict === undefined) this.s.strict = VDEFAULTS.strict;
    this.results = [];
    this.busy = false; this.stop = false;
    this.defaultGuideline = "";
    fetch(new URL("./validation_guideline.md", import.meta.url)).then((r) => (r.ok ? r.text() : "")).then((t) => { this.defaultGuideline = t; })
      .catch(() => {});
    editor.addAction("validate", "c", "Check rooms with the local model", "Check", () => this.toggle());
    this.panel = document.createElement("div");
    this.panel.className = "ed3d-panel ed3d-validator";
    this.panel.hidden = true;
    editor.container.appendChild(this.panel);
  }
  save() { try { localStorage.setItem(STORE, JSON.stringify({ settings: this.s, guideline: this.guideline })); } catch { /* private mode */ } }
  get guide() { return this.guideline || this.defaultGuideline; }

  // ---------------------------------------------------------------- what the model sees about one room
  rooms() {
    const { model } = this.getData();
    return (model ? model.rooms : []).filter((r) => r.status !== "REJECTED" && !r.is_hypothesis && r.polygon && r.polygon.exterior.length >= 3);
  }
  packet(room) {
    const { model, view } = this.getData(), ed = this.ed;
    const poly = room.polygon.exterior;
    const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]), m = this.s.margin;
    let bb = [Math.min(...xs) - m, Math.min(...ys) - m, Math.max(...xs) + m, Math.max(...ys) + m];
    const w = bb[2] - bb[0], h = bb[3] - bb[1], sz = Math.max(w, h);       // square crop, centred on the room
    bb = [bb[0] - (sz - w) / 2, bb[1] - (sz - h) / 2, bb[0] - (sz - w) / 2 + sz, bb[1] - (sz - h) / 2 + sz];
    const mw = new Map((model.walls || []).map((x) => [x.id, x])), mo = new Map((model.openings || []).map((x) => [x.id, x]));
    const mr = new Map((model.rooms || []).map((x) => [x.id, x]));
    // walls and openings as they are NOW in the editor (hand edits included), in drawing coordinates
    const walls = [], pk_poly = poly;
    for (const el of ed.els.values()) {
      if (el.kind !== "wall" && el.kind !== "railing") continue;
      const [ox, oy] = ed.drawnOffset(el.level);
      const a = [el.a[0] - ox, el.a[1] - oy], b = [el.b[0] - ox, el.b[1] - oy];
      const inCrop = [a, b].some((p) => p[0] >= bb[0] && p[0] <= bb[2] && p[1] >= bb[1] && p[1] <= bb[3]) || segDist([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2], a, b) < sz / 2;
      if (!inCrop) continue;
      const tol = (el.thickness || 0.1) / 2 + 0.15, clip = clipToBox(a, b, bb), L = hyp(a, b);
      // part of the wall inside the image: positions 0..1 are measured along it (its start is drawn)
      const ca = clip ? [a[0] + (b[0] - a[0]) * clip[0], a[1] + (b[1] - a[1]) * clip[0]] : a, cb = clip ? [a[0] + (b[0] - a[0]) * clip[1], a[1] + (b[1] - a[1]) * clip[1]] : b;
      const bounding = hyp(ca, cb) >= 0.4 && polyDist(a, b, poly) <= tol && outlineShare(ca, cb, poly, tol) * hyp(ca, cb) >= Math.min(0.3, 0.5 * hyp(ca, cb));
      walls.push({ el, a, b, ca, cb, sA: clip ? clip[0] * L : 0, sB: clip ? clip[1] * L : L, bounding, meta: mw.get(el.id) });
    }
    const tags = new Map(), back = {};
    const tag = (prefix, id) => { if (!tags.has(id)) { const n = [...tags.values()].filter((t) => t.startsWith(prefix)).length + 1; tags.set(id, prefix + n); back[prefix + n] = id; } return tags.get(id); };
    tag("R", room.id);
    const W = [], O = [];
    for (const x of walls.filter((x) => x.bounding)) {
      const t = tag("W", x.el.id), L = hyp(x.a, x.b), span = Math.max(x.sB - x.sA, 1e-6);
      const { u: wu } = { u: [(x.b[0] - x.a[0]) / L, (x.b[1] - x.a[1]) / L] };
      x.near = (o) => {                           // an opening next to this room (not in another room further along)
        const s = (o.s0 + o.s1) / 2, c = [x.a[0] + wu[0] * s, x.a[1] + wu[1] * s];
        return s >= x.sA && s <= x.sB && polyDist(c, c, pk_poly) <= (x.el.thickness || 0.1) / 2 + 0.4;
      };
      const st = x.el.id.startsWith("edit_") ? "drawn by hand" : (x.meta ? x.meta.status : "ACCEPTED");
      const ops = (x.el.openings || []).filter((o) => x.near(o)).map((o) => {
        const ot = tag(o.type === "door" ? "D" : o.type === "window" ? "N" : "O", o.id), om = mo.get(o.id);
        O.push({ tag: ot, id: o.id, type: o.type, subtype: om ? om.subtype : null, width_m: +(o.s1 - o.s0).toFixed(2), wall: t,
          position: +((((o.s0 + o.s1) / 2) - x.sA) / span).toFixed(2), sill_m: o.type === "window" ? +o.z0.toFixed(2) : null,
          status: om ? om.status : "drawn by hand", rooms: om ? om.rooms : [] });
        return ot;
      });
      W.push({ tag: t, id: x.el.id, kind: x.meta ? x.meta.classification : x.el.cls, status: st, thickness_m: +(x.el.thickness || 0).toFixed(2),
        length_m: +L.toFixed(2), visible_length_m: +span.toFixed(2), height_m: x.el.height ? +x.el.height.toFixed(2) : null, openings: ops,
        other_side: (x.meta && x.meta.rooms ? x.meta.rooms.filter((r) => r !== room.id).map((r) => (mr.get(r) || {}).name || (mr.get(r) || {}).classification || r) : []) });
    }
    const texts = (view.texts || []).filter((t) => t.t && inPoly(t.p, poly)).map((t) => t.t.split("\n")[0]).slice(0, 12);
    const neighbours = (room.connected_rooms || []).map((r) => (mr.get(r) || {}).name || (mr.get(r) || {}).classification || r);
    const ids = new Set([room.id, ...W.map((x) => x.id), ...O.map((x) => x.id)]);
    const issues = ((model.validation || {}).issues || []).filter((i) => ids.has(i.entity_id)).map((i) => `${i.code}: ${i.description}`).slice(0, 10);
    const ed3dRoom = ed.els.get(room.id);
    const meta = {
      room: { tag: "R1", id: room.id, name: room.name, type: (ed3dRoom && ed3dRoom.roomType) || room.classification, area_m2: room.area ? +room.area.toFixed(1) : null,
        texts_inside: texts, reachable_from: neighbours, has_door_or_passage: O.some((o) => o.type !== "window") },
      walls: W, openings: O, validation_issues: issues,
    };
    const span = {};
    for (const x of walls) if (tags.has(x.el.id)) span[x.el.id] = [x.sA, x.sB, hyp(x.a, x.b)];
    return { room, bb, walls, poly, tags, back, meta, view, span };
  }

  // ---------------------------------------------------------------- the two images
  svg(pk, overlay) {
    const [x0, y0, x1, y1] = pk.bb, w = x1 - x0, size = this.s.imageSize, px = w / size;
    const X = (x) => (x - x0).toFixed(3), Y = (y) => (y1 - y).toFixed(3);
    const box = (pts) => { const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]); return Math.max(...xs) >= x0 && Math.min(...xs) <= x1 && Math.max(...ys) >= y0 && Math.min(...ys) <= y1; };
    const lines = pk.view.lines.filter((l) => box(l.p)).map((l) => `<path d="M${l.p.map(([x, y]) => X(x) + " " + Y(y)).join(" L")}"/>`).join("");
    const texts = (pk.view.texts || []).filter((t) => t.t && t.p[0] >= x0 && t.p[0] <= x1 && t.p[1] >= y0 && t.p[1] <= y1)
      .map((t) => `<text x="${X(t.p[0])}" y="${Y(t.p[1])}" font-size="${Math.max(t.h || 0, w / 50).toFixed(3)}">${esc(t.t.split("\n")[0].slice(0, 40))}</text>`).join("");
    const bar = `<path d="M${(w * 0.04).toFixed(3)} ${(w * 0.97).toFixed(3)} h1" stroke="#0055cc" stroke-width="${(3 * px).toFixed(4)}"/>
      <text x="${(w * 0.04).toFixed(3)}" y="${(w * 0.97 - 6 * px).toFixed(3)}" font-size="${(w / 40).toFixed(3)}" fill="#0055cc">1 m</text>`;
    let over = "";
    if (overlay) {
      const P = (pts) => pts.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ");
      over += `<polygon points="${P(pk.poly)}" fill="${COL.room}" fill-opacity="0.55" stroke="#c9a400" stroke-width="${(2 * px).toFixed(4)}"/>`;
      const label = [], placed = [];
      const free = (x, y, fs) => placed.every(([px_, py_]) => Math.hypot(px_ - x, py_ - y) > fs * 1.6);
      const halo = (x, y, t, col, fs) => (placed.push([x, y]), label.push(`<text x="${X(x)}" y="${Y(y)}" font-size="${fs.toFixed(3)}" font-weight="700" text-anchor="middle" dominant-baseline="middle" stroke="#fff" stroke-width="${(fs * 0.25).toFixed(3)}" paint-order="stroke" fill="${col}">${t}</text>`));
      for (const x of pk.walls) {
        const { el, a, b } = x, L = hyp(a, b);
        if (L < 1e-6) continue;
        const u = [(b[0] - a[0]) / L, (b[1] - a[1]) / L], n = [-u[1], u[0]], t = Math.max(el.thickness || 0.1, 2.5 * px) / 2;
        const st = x.meta ? x.meta.status : "", cls = (x.meta && x.meta.classification) || el.cls;
        const col = el.id.startsWith("edit_") ? COL.edited : st === "NEEDS_REVIEW" ? COL.review : COL[cls] || COL.wall;
        const q = (s, side) => [a[0] + u[0] * s + n[0] * side * t, a[1] + u[1] * s + n[1] * side * t];
        over += `<polygon points="${P([q(0, -1), q(L, -1), q(L, 1), q(0, 1)])}" fill="${col}" fill-opacity="${x.bounding ? 0.85 : 0.35}"/>`;
        for (const o of el.openings || []) {
          const oc = COL[o.type] || COL.opening;
          if (o.s1 < x.sA || o.s0 > x.sB) continue;
          over += `<polygon points="${P([q(o.s0, -1.6), q(o.s1, -1.6), q(o.s1, 1.6), q(o.s0, 1.6)])}" fill="${oc}" fill-opacity="0.9"/>`;
          const tg = pk.tags.get(o.id);
          if (tg) { const c = q((o.s0 + o.s1) / 2, 0); halo(c[0] + n[0] * (t + w / 45), c[1] + n[1] * (t + w / 45), tg, oc, w / 42); }
        }
        const tg = pk.tags.get(el.id);
        if (tg) {
          over += `<circle cx="${X(x.ca[0])}" cy="${Y(x.ca[1])}" r="${(5 * px).toFixed(4)}" fill="#000"/>`;     // position 0 of this wall
          const fs = w / 38;                       // label on the wall's outer side, moved along it until it is clear
          let best = null;
          for (const f of [0.3, 0.55, 0.15, 0.8, 0.42, 0.68]) {
            for (const side of [-1, 1]) {
              const c = q(x.sA + (x.sB - x.sA) * f, 0), lx = c[0] + side * n[0] * (t + w / 45), ly = c[1] + side * n[1] * (t + w / 45);
              if (free(lx, ly, fs)) { best = [lx, ly]; break; }
            }
            if (best) break;
          }
          if (!best) { const c = q(x.sA + (x.sB - x.sA) * 0.3, 0); best = [c[0] - n[0] * (t + w / 45), c[1] - n[1] * (t + w / 45)]; }
          halo(best[0], best[1], tg, col, fs);
        }
      }
      const c = pk.poly.reduce((s, p) => [s[0] + p[0] / pk.poly.length, s[1] + p[1] / pk.poly.length], [0, 0]);
      halo(c[0], c[1], "R1", "#7a5c00", w / 30);
      over += label.join("");
    }
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${w.toFixed(3)} ${w.toFixed(3)}">
      <rect width="${w}" height="${w}" fill="#fff"/>
      <g fill="none" stroke="${overlay ? "#b9b9b9" : "#000"}" stroke-width="${(1.6 * px).toFixed(4)}" stroke-linecap="round">${lines}</g>
      <g fill="${overlay ? "#9a9a9a" : "#333"}" font-family="Arial, sans-serif">${texts}</g>${over}${bar}</svg>`;
  }
  raster(svg) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => { const c = document.createElement("canvas"); c.width = img.width; c.height = img.height; c.getContext("2d").drawImage(img, 0, 0); resolve(c.toDataURL("image/png").split(",")[1]); };
      img.onerror = () => reject(new Error("could not render the room image"));
      img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
    });
  }
  key(pk) {
    return [
      "KEY",
      "Reference image: the original drawing (black). Result image: faded drawing (grey) with the result on top:",
      "- yellow fill = the room being checked (R1)",
      "- wall bands: blue = exterior wall, dark green = interior wall, light green = partition, orange = UNCERTAIN wall, grey = railing, magenta = drawn by hand",
      "- marks across walls: brown = door, cyan = window, purple = opening (passage / unidentified)",
      "- a black dot marks where each tagged wall starts in the image (position 0); where it leaves the image or ends is position 1",
      "- blue bar = 1 m",
      "Tags: " + [...pk.tags.entries()].map(([id, t]) => `${t} = ${id}`).join(", "),
    ].join("\n");
  }

  // ---------------------------------------------------------------- LM Studio
  base() { return this.s.endpoint.replace(/\/+$/, "").replace(/\/v1$/, ""); }
  async models() {
    const get = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u} answered ${r.status}`); return r.json(); };
    try {
      const j = await get(this.base() + "/api/v0/models");
      return (j.data || []).filter((m) => m.type !== "embeddings").map((m) => ({ id: m.id, vision: m.type === "vlm", loaded: m.state === "loaded" }));
    } catch { /* older LM Studio or another OpenAI-compatible server */ }
    const j = await get(this.base() + "/v1/models");
    return (j.data || []).map((m) => ({ id: m.id, vision: null, loaded: null }));
  }
  async ask(system, text, images) {
    if (!this.s.model) throw new Error("choose a model first (Find models)");
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), this.s.timeoutS * 1000);
    this.abort = ctl;
    try {
      const content = [{ type: "text", text }, ...images.map((b) => ({ type: "image_url", image_url: { url: "data:image/png;base64," + b } }))];
      const body = { model: this.s.model, temperature: this.s.temperature, max_tokens: this.s.maxTokens,
        messages: [{ role: "system", content: system }, { role: "user", content }] };
      if (this.s.strict) body.response_format = { type: "json_schema", json_schema: { name: "room_review", strict: true, schema: SCHEMA } };
      this.lastRequest = body;
      const r = await fetch(this.base() + "/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctl.signal });
      if (!r.ok) throw new Error(`LM Studio answered ${r.status}: ${(await r.text()).slice(0, 300)}`);
      const j = await r.json();
      const msg = (j.choices && j.choices[0] && j.choices[0].message) || {};
      this.lastReasoning = msg.reasoning_content || msg.reasoning || "";
      if (!msg.content && j.choices && j.choices[0] && j.choices[0].finish_reason === "length")
        throw new Error("the answer was cut off (max tokens): raise the limit or switch strict JSON off");
      return msg.content || "";
    } finally { clearTimeout(timer); }
  }
  prompt(pk) {
    const how = this.s.strict
      ? ["Answer with the JSON schema (room = \"R1\"). Write \"analysis\" FIRST: go through the guideline's rules one by one",
        "for this room (access, uncertain walls, windows, doors, garage, entrance, wall thickness, room type), say what the",
        "reference shows and what the result has. Then list an edit for every difference you found."]
      : ["First write your analysis in plain text: go through the guideline's rules one by one for this room (access,",
        "uncertain walls, windows, doors, garage, entrance, wall thickness, room type), say what the reference shows and",
        "what the result has. Then end with ONE JSON object in a ```json block with the keys room (\"R1\"), analysis (short),",
        "summary and edits, each edit with action, target, position, width, value, confidence and reason."];
    return [this.key(pk), "", "METADATA (the result as data, JSON):", JSON.stringify(pk.meta, null, 1), "",
      "Image 1 is the REFERENCE drawing, image 2 is the RESULT with tags. Check room R1 following the guideline.", ...how].join("\n");
  }

  // ---------------------------------------------------------------- answers -> checked edits
  check(pk, answer) {
    let j;
    const lastJson = (t) => {                      // a ```json block, else the outermost {...}
      const blocks = [...String(t).matchAll(/```(?:json)?\s*([\s\S]*?)```/g)].map((m) => m[1]);
      for (const b of blocks.reverse()) { try { return JSON.parse(b); } catch { /* next */ } }
      return JSON.parse((/\{[\s\S]*\}/.exec(t) || ["{}"])[0]);
    };
    try { j = typeof answer === "string" ? lastJson(answer) : answer; }
    catch (e) { throw new Error("the model did not answer valid JSON: " + String(answer).slice(0, 160)); }
    const kindOf = (tag) => (tag || "").replace(/\d+$/, "");
    const edits = (Array.isArray(j.edits) ? j.edits : []).map((e, k) => {
      const out = { n: k + 1, action: e.action, target: String(e.target || "").trim().toUpperCase(), position: Number(e.position), width: Number(e.width) || 0,
        value: String(e.value || "").trim().toLowerCase(), confidence: Math.max(0, Math.min(1, Number(e.confidence) || 0)), reason: String(e.reason || "").slice(0, 300), status: "pending" };
      out.id = pk.back[out.target] || null;
      const need = { delete_wall: ["W"], add_door: ["W"], add_window: ["W"], change_opening: ["D", "N", "O"], remove_opening: ["D", "N", "O"], set_room_type: ["R"] }[out.action];
      if (!ACTIONS.includes(out.action)) { out.status = "invalid"; out.msg = "unknown action"; }
      else if (need && (!out.id || !need.includes(kindOf(out.target)))) { out.status = "invalid"; out.msg = `target ${out.target || "(none)"} is not a ${need.join("/")} tag of this room`; }
      else if (out.action === "change_opening" && !["door", "window", "opening"].includes(out.value)) { out.status = "invalid"; out.msg = "value must be door, window or opening"; }
      else if (out.action === "set_room_type" && !out.value) { out.status = "invalid"; out.msg = "no room type given"; }
      else if ((out.action === "add_door" || out.action === "add_window") && !(out.position >= 0 && out.position <= 1)) { out.status = "invalid"; out.msg = "position must be between 0 and 1"; }
      else if (out.action === "comment") out.status = "comment";
      return out;
    });
    return { analysis: String(j.analysis || "").slice(0, 4000), summary: String(j.summary || "").slice(0, 600), edits };
  }
  apply(res, e) {
    if (e.status !== "pending" && e.status !== "rejected") return false;
    const sp = (res.span || {})[e.id];
    const pos = sp ? (sp[0] + e.position * (sp[1] - sp[0])) / sp[2] : e.position;
    const r = this.ed.aiEdit({ action: e.action, id: e.id, position: pos, width: e.width, value: e.value,
      label: `AI (${Math.round(e.confidence * 100)}%): ${e.action.replace("_", " ")} ${e.target} in ${res.room}` });
    e.status = r.ok ? "applied" : "failed";
    e.msg = r.msg;
    return r.ok;
  }

  // ---------------------------------------------------------------- running
  async run(rooms) {
    if (this.busy) return;
    this.busy = true; this.stop = false;
    const system = this.guide + "\n\nAnswer ONLY with JSON matching the schema.";
    for (let i = 0; i < rooms.length && !this.stop; i++) {
      const room = rooms[i];
      this.progress = `Checking ${room.id} (${i + 1} / ${rooms.length})…`;
      this.render();
      const rec = { room: room.id, name: room.name || room.classification, model: this.s.model, at: new Date().toISOString() };
      try {
        const pk = this.packet(room);
        const imgs = [await this.raster(this.svg(pk, false)), await this.raster(this.svg(pk, true))];
        const t0 = performance.now();
        const txt = await this.ask(system, this.prompt(pk), imgs);
        rec.seconds = +((performance.now() - t0) / 1000).toFixed(1);
        Object.assign(rec, this.check(pk, txt), { tags: Object.fromEntries(pk.tags), span: pk.span, raw: txt, reasoning: this.lastReasoning });
        rec.request = this.lastRequest;
        for (const e of rec.edits) if (e.status === "pending" && e.confidence >= this.s.autoApply) this.apply(rec, e);
      } catch (e) {
        rec.request = this.lastRequest;
        rec.error = /abort/i.test(e.message) ? "stopped or timed out" : e.message;
        if (/Failed to fetch|NetworkError/i.test(e.message)) {
          rec.error += " — is the LM Studio server running, and is CORS enabled in its server settings?";
          rec.request = this.lastRequest;
          this.results = this.results.filter((x) => x.room !== room.id).concat(rec);
          break;
        }
      }
      this.results = this.results.filter((x) => x.room !== room.id).concat(rec);
      this.render();
    }
    this.busy = false; this.progress = this.stop ? "Stopped" : "Done";
    this.render();
  }
  selectedRooms() {
    const ids = new Set([...this.ed.sel].filter((k) => !k.startsWith("op:")));
    return this.rooms().filter((r) => ids.has(r.id));
  }

  // ---------------------------------------------------------------- panel
  toggle() { this.panel.hidden = !this.panel.hidden; if (!this.panel.hidden) this.render(); }
  render() {
    const p = this.panel;
    if (p.hidden) return;
    const s = this.s, sel = this.selectedRooms(), all = this.rooms();
    const pending = this.results.flatMap((r) => (r.edits || []).filter((e) => e.status === "pending"));
    p.innerHTML = `<div class="ed3d-row"><b>Room check · local model</b><button data-do="close">×</button></div>
      <div class="ed3d-row"><span>server</span><input data-s="endpoint" value="${esc(s.endpoint)}" style="width:170px"></div>
      <div class="ed3d-row"><span>model</span><select data-s="model">${s.model ? `<option>${esc(s.model)}</option>` : "<option value=''>—</option>"}</select><button data-do="models">Find models</button></div>
      <div class="ed3d-row"><span title="On: LM Studio forces the JSON schema (analysis is written first). Off: the model answers freely, like in LM Studio's chat, and ends with a JSON block.">strict JSON answer</span><input data-s="strict" type="checkbox" ${s.strict ? "checked" : ""}></div>
      <div class="ed3d-row"><span>max answer tokens</span><input data-s="maxTokens" type="number" step="500" min="500" value="${s.maxTokens}"></div>
      <div class="ed3d-row"><span>apply automatically from confidence</span><input data-s="autoApply" type="number" step="0.05" min="0" max="1.01" value="${s.autoApply}"></div>
      <details><summary>Guideline${this.guideline ? " (edited)" : ""}</summary><textarea data-g rows="12" style="width:100%;font:11px ui-monospace,monospace">${esc(this.guide)}</textarea>
        <div class="ed3d-row"><button data-do="saveg">Save guideline</button><button data-do="resetg">Default guideline</button></div></details>
      <div class="ed3d-row"><button data-do="runsel" ${sel.length && !this.busy ? "" : "disabled"}>Check selected room${sel.length > 1 ? "s" : ""} (${sel.length})</button>
        <button data-do="runall" ${all.length && !this.busy ? "" : "disabled"}>Check all ${all.length} rooms</button><button data-do="stop" ${this.busy ? "" : "disabled"}>Stop</button></div>
      <div class="muted">${esc(this.progress || "Select a room's floor in the 3D view, or check them all. Edits at or above the confidence are applied at once (Ctrl+Z undoes them); the others wait here.")}</div>
      ${pending.length ? `<div class="ed3d-row"><button data-do="applyall">Apply all ${pending.length} waiting edits</button><button data-do="export">Export results JSON</button></div>` : (this.results.length ? `<div class="ed3d-row"><button data-do="export">Export results JSON</button></div>` : "")}
      <div class="ed3d-vresults">${this.results.slice().reverse().map((r, ri) => this.renderRoom(r, this.results.length - 1 - ri)).join("")}</div>`;
    const bind = (sel_, fn) => p.querySelectorAll(sel_).forEach((b) => b.addEventListener("click", fn));
    bind('[data-do="close"]', () => this.toggle());
    for (const inp of p.querySelectorAll("[data-s]")) inp.addEventListener("change", () => {
      this.s[inp.dataset.s] = inp.type === "checkbox" ? inp.checked : inp.type === "number" ? parseFloat(inp.value) : inp.value; this.save();
    });
    bind('[data-do="models"]', async () => {
      try {
        const ms = await this.models();
        if (!ms.length) throw new Error("no models: load a vision model in LM Studio");
        const best = ms.find((m) => m.vision && m.loaded) || ms.find((m) => m.loaded) || ms.find((m) => m.vision) || ms[0];
        if (!this.s.model) this.s.model = best.id;
        const selEl = p.querySelector('[data-s="model"]');
        selEl.innerHTML = ms.map((m) => `<option value="${esc(m.id)}" ${m.id === this.s.model ? "selected" : ""}>${esc(m.id)}${m.vision ? " · vision" : ""}${m.loaded ? " · loaded" : ""}</option>`).join("");
        this.save();
      } catch (e) { alert(`Could not reach the model server: ${e.message}\nStart LM Studio's server (Developer tab) and enable CORS there.`); }
    });
    bind('[data-do="saveg"]', () => { this.guideline = p.querySelector("[data-g]").value; this.save(); this.render(); });
    bind('[data-do="resetg"]', () => { this.guideline = null; this.save(); this.render(); });
    bind('[data-do="runsel"]', () => this.run(sel));
    bind('[data-do="runall"]', () => this.run(all));
    bind('[data-do="stop"]', () => { this.stop = true; if (this.abort) this.abort.abort(); });
    bind('[data-do="applyall"]', () => { for (const r of this.results) for (const e of r.edits || []) if (e.status === "pending") this.apply(r, e); this.render(); });
    bind('[data-do="export"]', () => this.ed.download("room-check.json", JSON.stringify({ format: "cad2d3d-room-check", version: 1, settings: { model: this.s.model, autoApply: this.s.autoApply },
      guideline: this.guide, results: this.results.map(({ raw, ...r }) => r) }, null, 1), "application/json"));
    bind("[data-act]", (ev) => {
      const b = ev.currentTarget, r = this.results[+b.dataset.r], e = r.edits[+b.dataset.e];
      if (b.dataset.act === "apply") this.apply(r, e);
      else if (b.dataset.act === "reject") e.status = "rejected";
      else if (b.dataset.act === "show") this.ed.showElement(e.id || r.room);
      this.render();
    });
    bind("[data-req]", (ev) => {                  // the exact request (prompt + both images) and answer, to replay or share
      const r = this.results[+ev.currentTarget.dataset.req];
      this.ed.download(`room-check-${r.room}.request.json`, JSON.stringify({ request: r.request, answer: r.raw, reasoning: r.reasoning }, null, 1), "application/json");
    });
  }
  renderRoom(r, ri) {
    const badge = (e) => `<span class="ed3d-badge ed3d-${e.status}">${e.status}</span>`;
    const rows = (r.edits || []).map((e, ei) => `<div class="ed3d-vedit">
        <div class="ed3d-row"><b>${esc(e.action.replace("_", " "))} ${esc(e.target)}${e.value ? ` → ${esc(e.value)}` : ""}${e.action.startsWith("add_") ? ` @ ${(+e.position).toFixed(2)}` : ""}</b>
        <span><span class="ed3d-conf" style="--c:${Math.round(e.confidence * 100)}%"></span>${Math.round(e.confidence * 100)}% ${badge(e)}</span></div>
        <div class="muted">${esc(e.reason)}${e.msg ? ` <i>(${esc(e.msg)})</i>` : ""}</div>
        <div class="ed3d-row">${e.id || r.room ? `<button data-act="show" data-r="${ri}" data-e="${ei}">Show</button>` : ""}
        ${e.status === "pending" || e.status === "rejected" ? `<button data-act="apply" data-r="${ri}" data-e="${ei}">Apply</button>` : ""}
        ${e.status === "pending" ? `<button data-act="reject" data-r="${ri}" data-e="${ei}">Reject</button>` : ""}</div></div>`).join("");
    const more = `<details><summary class="muted">model's analysis and raw answer</summary>
        ${r.reasoning ? `<div class="muted small"><b>reasoning</b><pre class="ed3d-pre">${esc(r.reasoning)}</pre></div>` : ""}
        <div class="small"><b>analysis</b><pre class="ed3d-pre">${esc(r.analysis || "—")}</pre></div>
        <div class="small"><b>raw answer</b><pre class="ed3d-pre">${esc(r.raw || "—")}</pre></div></details>
      ${r.request ? `<button data-req="${ri}">Save request (prompt + images)</button>` : ""}`;
    return `<div class="ed3d-vroom"><div class="ed3d-row"><b>${esc(r.room)}</b><span class="muted">${esc(r.name || "")}${r.seconds ? ` · ${r.seconds} s` : ""}</span></div>
      ${r.error ? `<div class="err">${esc(r.error)}</div>` : `<div>${esc(r.summary || "")}</div>${rows || "<div class='muted'>no edits proposed</div>"}`}${more}</div>`;
  }
}
