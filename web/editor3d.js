import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

/* 3D scene editor (Blender-style): select, move / rotate / scale with axis constraints and typed values,
   draw walls, cut doors and windows, fill gaps, draw floors and roofs, undo / redo.

   Every editable element is kept as the numbers it is built from (Phase 2's mesh "params": wall ends,
   thickness, base, height, openings along the wall; prism rings and heights) and rebuilt here, in the
   browser, when it changes: an edit costs milliseconds and never re-runs the pipeline. Meshes without
   params (stairs, pitched roofs, ground slabs) are fixed shapes that can still be moved, rotated, scaled
   and deleted as a whole. Model coordinates are metres, z up; three.js is y up (the root group turns it).

   Keys, navigation and defaults are settings (stored in the browser, exportable as JSON). */

export const DEFAULT_SETTINGS = {
  wallHeight: 2.8, wallThickness: 0.2, doorWidth: 0.9, doorHeight: 2.1, windowWidth: 1.2, windowSill: 0.9, windowHead: 2.1,
  slabThickness: 0.2, roofStyle: "hip", roofPitch: 30, roofThickness: 0.15, roofOverhang: 0.3,
  grid: 0.05, snap: true, snapPx: 14, angleSnap: 15, nav: "blender", doorLeaves: true, workZ: 0,
};

export const DEFAULT_KEYMAP = {
  grab: "g", rotate: "r", scale: "s", delete: "x", delete_alt: "delete", duplicate: "shift+d",
  select_all: "a", deselect_all: "alt+a", hide: "h", unhide: "alt+h", frame_selected: ".", frame_all: "home",
  undo: "ctrl+z", redo: "ctrl+shift+z", redo_alt: "ctrl+y",
  tool_wall: "w", tool_door: "d", tool_window: "v", fill_gap: "f", tool_floor: "p", tool_roof: "o",
  properties: "n", help: "f1", settings: "ctrl+,",
  axis_x: "x", axis_y: "y", axis_z: "z", cycle_roof: "tab",
};

const ACTION_LABEL = {
  grab: "Move (grab)", rotate: "Rotate about the vertical axis", scale: "Scale (Z: height)", delete: "Delete", delete_alt: "Delete (2nd key)",
  duplicate: "Duplicate and move", select_all: "Select all", deselect_all: "Deselect all", hide: "Hide selected", unhide: "Show hidden",
  frame_selected: "Frame selected", frame_all: "Frame all", undo: "Undo", redo: "Redo", redo_alt: "Redo (2nd key)",
  tool_wall: "Draw walls", tool_door: "Cut a door", tool_window: "Cut a window", fill_gap: "Fill gap / join walls / close opening",
  tool_floor: "Draw a floor slab", tool_roof: "Draw a roof", properties: "Properties panel", help: "Help", settings: "Settings",
  axis_x: "Constrain to X (in a move / scale)", axis_y: "Constrain to Y", axis_z: "Constrain to Z (height)", cycle_roof: "Roof style (while drawing a roof)",
};

const STORE = "cad2d3d.editor3d";
const C = {
  glass: [0.55, 0.75, 0.95, 0.45], door: [0.56, 0.40, 0.26, 1], wall: [0.88, 0.86, 0.82, 1], floor: [0.76, 0.66, 0.52, 1],
  roof: [0.72, 0.36, 0.25, 1], select: 0xff7a1a, ghost: 0x2b8cff, bad: 0xe53935,
};
const ROOF_STYLES = ["flat", "shed", "gable", "hip"];
const clone = (o) => JSON.parse(JSON.stringify(o));
const hyp = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);

function keyOf(ev) {
  const k = (ev.key || "").toLowerCase();
  if (["control", "shift", "alt", "meta"].includes(k)) return null;
  return [(ev.ctrlKey || ev.metaKey) && "ctrl", ev.shiftKey && "shift", ev.altKey && "alt", k === " " ? "space" : k].filter(Boolean).join("+");
}

// ------------------------------------------------------------------ geometry builders (model space, z up)
function wallFrame(a, b) {
  const L = Math.max(hyp(a, b), 1e-9);
  const u = [(b[0] - a[0]) / L, (b[1] - a[1]) / L];
  return { L, u, n: [-u[1], u[0]] };
}

class Geo {
  constructor() { this.p = []; this.i = []; }
  quad(a, b, c, d) { const o = this.p.length / 3; this.p.push(...a, ...b, ...c, ...d); this.i.push(o, o + 1, o + 2, o, o + 2, o + 3); }
  tri(a, b, c) { const o = this.p.length / 3; this.p.push(...a, ...b, ...c); this.i.push(o, o + 1, o + 2); }
  box(a, u, n, s0, s1, t, z0, z1) {           // a box along a wall frame: s along u, ±t/2 along n
    if (s1 - s0 <= 1e-6 || z1 - z0 <= 1e-6) return;
    const P = (s, side, z) => [a[0] + u[0] * s + n[0] * side * t / 2, a[1] + u[1] * s + n[1] * side * t / 2, z];
    const v = [P(s0, -1, z0), P(s1, -1, z0), P(s1, 1, z0), P(s0, 1, z0), P(s0, -1, z1), P(s1, -1, z1), P(s1, 1, z1), P(s0, 1, z1)];
    const f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
    for (const [q, r, s, w] of f) this.quad(v[q], v[r], v[s], v[w]);
  }
  prism(ring, z0, z1, holes = []) {
    if (!ring || ring.length < 3 || z1 - z0 <= 1e-6) return;
    const r = THREE.ShapeUtils.isClockWise(ring.map((p) => new THREE.Vector2(p[0], p[1]))) ? [...ring].reverse() : ring;
    const hs = holes.map((h) => (THREE.ShapeUtils.isClockWise(h.map((p) => new THREE.Vector2(p[0], p[1]))) ? h : [...h].reverse()));
    const all = [r, ...hs].flat();
    const tris = THREE.ShapeUtils.triangulateShape(r.map((p) => new THREE.Vector2(p[0], p[1])), hs.map((h) => h.map((p) => new THREE.Vector2(p[0], p[1]))));
    for (const [a, b, c] of tris) {
      this.tri([...all[a], z1], [...all[b], z1], [...all[c], z1]);
      this.tri([...all[a], z0], [...all[c], z0], [...all[b], z0]);
    }
    for (const loop of [r, ...hs]) {
      for (let k = 0; k < loop.length; k++) {
        const p = loop[k], q = loop[(k + 1) % loop.length];
        this.quad([...p, z0], [...q, z0], [...q, z1], [...p, z1]);
      }
    }
  }
  geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.p, 3));
    g.setIndex(this.i);
    g.computeVertexNormals();
    return g;
  }
  get empty() { return this.i.length === 0; }
}

function minRect(ring) {                        // minimum-area rectangle: {c, u, v, hl, hw} (hl >= hw)
  let best = null;
  for (let k = 0; k < ring.length; k++) {
    const p = ring[k], q = ring[(k + 1) % ring.length];
    const L = hyp(p, q);
    if (L < 1e-9) continue;
    const u = [(q[0] - p[0]) / L, (q[1] - p[1]) / L], v = [-u[1], u[0]];
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (const r of ring) { const a = r[0] * u[0] + r[1] * u[1], b = r[0] * v[0] + r[1] * v[1]; a0 = Math.min(a0, a); a1 = Math.max(a1, a); b0 = Math.min(b0, b); b1 = Math.max(b1, b); }
    const area = (a1 - a0) * (b1 - b0);
    if (!best || area < best.area) best = { area, u, v, a0, a1, b0, b1 };
  }
  if (!best) return null;
  let { u, v, a0, a1, b0, b1 } = best;
  const am = (a0 + a1) / 2, bm = (b0 + b1) / 2;
  const c = [u[0] * am + v[0] * bm, u[1] * am + v[1] * bm];
  let hl = (a1 - a0) / 2, hw = (b1 - b0) / 2;
  if (hw > hl) { [u, v] = [v, [-u[0], -u[1]]]; [hl, hw] = [hw, hl]; }
  return { c, u, v, hl, hw };
}

function offsetRing(ring, d) {                  // grow a polygon by d (simple vertex offset along bisectors)
  if (!d) return ring;
  const cw = THREE.ShapeUtils.isClockWise(ring.map((p) => new THREE.Vector2(p[0], p[1])));
  const s = cw ? -1 : 1, n = ring.length;
  return ring.map((p, k) => {
    const a = ring[(k - 1 + n) % n], b = ring[(k + 1) % n];
    const e1 = wallFrame(a, p), e2 = wallFrame(p, b);
    const n1 = [e1.u[1] * s, -e1.u[0] * s], n2 = [e2.u[1] * s, -e2.u[0] * s];
    const m = [n1[0] + n2[0], n1[1] + n2[1]], ml = Math.hypot(m[0], m[1]) || 1;
    const cos = (m[0] * n1[0] + m[1] * n1[1]) / ml || 1;
    return [p[0] + (m[0] / ml) * d / Math.max(cos, 0.25), p[1] + (m[1] / ml) * d / Math.max(cos, 0.25)];
  });
}

function roofGeo(el) {
  const g = new Geo(), tan = Math.tan((el.pitch || 0) * Math.PI / 180), t = el.thickness || 0.15;
  const ring = offsetRing(el.ring, el.overhang || 0), z = el.z0;
  if (el.style === "flat" || !tan) { g.prism(ring, z, z + t); return g; }
  if (el.style === "shed") {
    const ang = (el.dir || 0) * Math.PI / 180, d = [Math.cos(ang), Math.sin(ang)];
    const lo = Math.min(...ring.map((p) => p[0] * d[0] + p[1] * d[1]));
    const h = (p) => z + tan * (p[0] * d[0] + p[1] * d[1] - lo);
    const r = THREE.ShapeUtils.isClockWise(ring.map((p) => new THREE.Vector2(p[0], p[1]))) ? [...ring].reverse() : ring;
    for (const [a, b, c] of THREE.ShapeUtils.triangulateShape(r.map((p) => new THREE.Vector2(p[0], p[1])), [])) {
      g.tri([...r[a], h(r[a]) + t], [...r[b], h(r[b]) + t], [...r[c], h(r[c]) + t]);
      g.tri([...r[a], h(r[a])], [...r[c], h(r[c])], [...r[b], h(r[b])]);
    }
    for (let k = 0; k < r.length; k++) { const p = r[k], q = r[(k + 1) % r.length]; g.quad([...p, h(p)], [...q, h(q)], [...q, h(q) + t], [...p, h(p) + t]); }
    return g;
  }
  const R = minRect(ring);                      // gable / hip: over the polygon's minimum rectangle
  if (!R) return g;
  const { c, u, v, hl, hw } = R, P = (a, b, zz) => [c[0] + u[0] * a + v[0] * b, c[1] + u[1] * a + v[1] * b, zz];
  const H = z + hw * tan;
  const k0 = P(-hl, -hw, z), k1 = P(hl, -hw, z), k2 = P(hl, hw, z), k3 = P(-hl, hw, z);
  const surf = (lift) => {
    const L = (p) => [p[0], p[1], p[2] + lift];
    if (el.style === "gable") {
      const r0 = P(-hl, 0, H), r1 = P(hl, 0, H);
      g.quad(L(k0), L(k1), L(r1), L(r0)); g.quad(L(k2), L(k3), L(r0), L(r1));
      if (!lift) { g.tri(k3, k0, r0); g.tri(k1, k2, r1); }   // gable ends (walls of the roof)
    } else {
      const e = Math.max(hl - hw, 0), r0 = P(-e, 0, H), r1 = P(e, 0, H);
      g.quad(L(k0), L(k1), L(r1), L(r0)); g.quad(L(k2), L(k3), L(r0), L(r1));
      g.tri(L(k1), L(k2), L(r1)); g.tri(L(k3), L(k0), L(r0));
    }
  };
  surf(0); surf(t);
  return g;
}

function wallGeo(el, leaves) {                  // the same splitting around openings as Phase 2's builder
  const wall = new Geo(), glass = new Geo(), door = new Geo();
  const { L, u, n } = wallFrame(el.a, el.b), t = el.thickness, z0 = el.z0, H = el.height;
  const ops = (el.openings || []).map((o) => ({ ...o, s0: Math.max(0, Math.min(o.s0, o.s1)), s1: Math.min(L, Math.max(o.s0, o.s1)) }))
    .filter((o) => o.s1 - o.s0 > 1e-6).sort((p, q) => p.s0 - q.s0);
  let cur = 0;
  for (const o of ops) {
    const oz0 = Math.max(0, o.z0), oz1 = Math.min(o.z1, H);
    if (o.s0 > cur) wall.box(el.a, u, n, cur, o.s0, t, z0, z0 + H);
    const lo = Math.max(cur, o.s0);
    if (oz0 > 0) wall.box(el.a, u, n, lo, o.s1, t, z0, z0 + oz0);
    if (oz1 < H) wall.box(el.a, u, n, lo, o.s1, t, z0 + oz1, z0 + H);
    if (o.type === "window") glass.box(el.a, u, n, lo, o.s1, Math.max(0.01, t * 0.1), z0 + oz0, z0 + oz1);
    if (o.type === "door" && leaves) door.box(el.a, u, n, lo + 0.01, o.s1 - 0.01, Math.min(0.05, t * 0.4), z0 + oz0, z0 + oz1 - 0.01);
    cur = Math.max(cur, o.s1);
  }
  if (cur < L) wall.box(el.a, u, n, cur, L, t, z0, z0 + H);
  return { wall, glass, door, ops };
}

function railingGeo(el) {
  const wall = new Geo(), glass = new Geo();
  const { L, u, n } = wallFrame(el.a, el.b);
  wall.box(el.a, u, n, 0, L, el.thickness, el.z0, el.z0 + el.parapet);
  if (el.top > el.parapet) glass.box(el.a, u, n, 0, L, Math.max(0.02, el.thickness * 0.25), el.z0 + el.parapet, el.z0 + el.top);
  return { wall, glass };
}

// ------------------------------------------------------------------ element transforms (model space)
function rot(p, c, ang) {
  const s = Math.sin(ang), k = Math.cos(ang), x = p[0] - c[0], y = p[1] - c[1];
  return [c[0] + x * k - y * s, c[1] + x * s + y * k];
}
function pointsOf(el) {
  if (el.kind === "wall" || el.kind === "railing") return [el.a, el.b];
  if (el.kind === "prism" || el.kind === "roof") return el.ring;
  if (el.kind === "mesh") { const b = el.box; return b ? [[b[0], b[1]], [b[3], b[4]]] : []; }
  return [];
}
function mapPoints(el, f) {
  if (el.kind === "wall" || el.kind === "railing") { el.a = f(el.a); el.b = f(el.b); }
  else if (el.kind === "prism" || el.kind === "roof") el.ring = el.ring.map(f);
}

export class Editor3D {
  constructor(container, onPick) {
    this.container = container;
    this.onPick = onPick;
    container.classList.add("ed3d");
    this.settings = { ...DEFAULT_SETTINGS };
    this.keymap = { ...DEFAULT_KEYMAP };
    try {
      const st = JSON.parse(localStorage.getItem(STORE) || "{}");
      Object.assign(this.settings, st.settings || {});
      Object.assign(this.keymap, st.keymap || {});
    } catch { /* private mode */ }
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0xeef1f6);
    container.appendChild(this.renderer.domElement);
    this.canvas = this.renderer.domElement;
    this.canvas.tabIndex = 0;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 2000);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.applyNav();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8c8478, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(30, 60, 20);
    this.scene.add(sun);
    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2;
    this.scene.add(this.root);
    this.overlay = new THREE.Group();          // previews, ghosts, selection frames (model space)
    this.root.add(this.overlay);
    this.grid = null;
    this.els = new Map();                       // id -> element (the document)
    this.objs = new Map();                      // id -> THREE.Group
    this.fixed = new Map();                     // mesh id -> {positions, indices} (never copied into undo)
    this.sel = new Set();                       // element ids, or "op:<wallId>:<openingId>"
    this.hidden = new Set();
    this.visible = { floor: true, roof: true, review: true, window: true, maxLevel: Infinity };
    this.levelRank = new Map();
    this.undoStack = []; this.redoStack = [];
    this.mode = null;                           // null | {type: grab|rotate|scale|wall|door|window|floor|roof, ...}
    this.seq = 0;
    this.dirty = false;
    this.ray = new THREE.Raycaster();
    this.mouse = { x: 0, y: 0, down: null };
    this.buildUi();
    this.bindEvents();
    new ResizeObserver(() => this.resize()).observe(container);
    const loop = () => {
      requestAnimationFrame(loop);
      if (this.container.offsetParent !== null) { this.controls.update(); this.renderer.render(this.scene, this.camera); }
    };
    loop();
  }

  // ---------------------------------------------------------------- settings / navigation
  saveSettings() { try { localStorage.setItem(STORE, JSON.stringify({ settings: this.settings, keymap: this.keymap })); } catch { /* private mode */ } }
  applyNav() {
    const M = THREE.MOUSE;
    // blender: middle orbits, shift+middle pans (OrbitControls pans a rotate-drag with shift), wheel zooms,
    // left selects; alt+left orbits (emulated 3-button mouse). classic: left orbits, right pans.
    this.controls.mouseButtons = this.settings.nav === "classic" ? { LEFT: M.ROTATE, MIDDLE: M.DOLLY, RIGHT: M.PAN }
      : { LEFT: -1, MIDDLE: M.ROTATE, RIGHT: -1 };
  }

  resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ---------------------------------------------------------------- loading
  setData(mesh, model) {
    if (this.dirty && this.els.size && !window.confirm("Replace the 3D scene and lose its edits? (Cancel keeps the edited scene.)")) return;
    for (const id of [...this.objs.keys()]) this.dropObj(id);
    this.els.clear(); this.fixed.clear(); this.sel.clear(); this.hidden.clear();
    this.undoStack = []; this.redoStack = []; this.dirty = false; this.endMode(true);
    const review = new Set((model.walls || []).filter((w) => w.status === "NEEDS_REVIEW").map((w) => w.id));
    const levelOf = new Map();
    for (const k of ["walls", "openings", "rooms", "columns", "stairs"]) for (const e of model[k] || []) levelOf.set(e.id, e.level);
    const lv = (model.building && model.building.stacked) ? model.building.levels.filter((l) => l.kind === "floor_plan") : [];
    this.levelRank = new Map(lv.sort((a, b) => a.z - b.z).map((l, i) => [l.id, i]));
    this.levels = lv.map((l) => ({ id: l.id, name: l.name, z: l.z || 0 }));
    for (const m of mesh.meshes) {
      const p = m.params, base = { id: m.id, type: m.type, cls: m.classification, color: m.color, level: m.level ?? levelOf.get(m.id), review: review.has(m.id) };
      if (p && p.kind === "glass_of") continue;            // rebuilt with its wall
      if (p && p.kind === "wall") {
        this.els.set(m.id, { ...base, kind: "wall", a: p.start, b: p.end, thickness: p.thickness, z0: p.z0, height: p.height,
          openings: (p.openings || []).map((o) => ({ ...o })) });
      } else if (p && p.kind === "railing") {
        this.els.set(m.id, { ...base, kind: "railing", a: p.start, b: p.end, thickness: p.thickness, z0: p.z0, parapet: p.parapet, top: p.top });
      } else if (p && p.kind === "prism") {
        this.els.set(m.id, { ...base, kind: "prism", ring: p.ring, z0: p.z0, z1: p.z1 });
      } else {
        this.fixed.set(m.id, { positions: m.positions, indices: m.indices });
        let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (let k = 0; k < m.positions.length; k += 3) for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], m.positions[k + d]); hi[d] = Math.max(hi[d], m.positions[k + d]); }
        this.els.set(m.id, { ...base, kind: "mesh", matrix: new THREE.Matrix4().toArray(), box: [...lo, ...hi] });
      }
    }
    for (const id of this.els.keys()) this.buildObj(id);
    this.applyVisibility();
    const [lo, hi] = mesh.metadata.bounds || [[0, 0, 0], [10, 10, 3]];
    this.bounds = [lo, hi];
    this.settings.workZ = Math.max(0, lo[2] < 0 ? 0 : lo[2]);
    this.frame(lo, hi);
    this.drawGrid();
    this.resize();
    this.refreshUi();
  }

  frame(lo, hi) {
    const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, cz = (lo[2] + hi[2]) / 2;
    const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 3);
    this.controls.target.set(cx, cz, -cy);
    this.camera.position.set(cx + size * 0.55, cz + size * 0.85, -cy + size * 0.95);
    this.camera.near = size / 500; this.camera.far = size * 20;
    this.camera.updateProjectionMatrix();
  }

  drawGrid() {
    if (this.grid) { this.overlay.remove(this.grid); this.grid.geometry.dispose(); }
    const [lo, hi] = this.bounds || [[0, 0, 0], [20, 20, 3]];
    const size = Math.ceil(Math.max(hi[0] - lo[0], hi[1] - lo[1], 10) * 1.4);
    this.grid = new THREE.GridHelper(size, Math.max(10, Math.round(size)), 0xb0b7c3, 0xd5dae2);
    this.grid.rotation.x = Math.PI / 2;
    this.grid.position.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, this.settings.workZ - 0.002);
    this.grid.visible = !!(this.mode && ["wall", "floor", "roof"].includes(this.mode.type));
    this.overlay.add(this.grid);
  }

  // ---------------------------------------------------------------- building objects
  material(color, extra = {}) {
    const [r, g, b, a] = color;
    return new THREE.MeshStandardMaterial({ color: new THREE.Color(r, g, b), transparent: a < 1, opacity: a, roughness: 0.85,
      metalness: 0, side: THREE.DoubleSide, flatShading: true, depthWrite: a >= 1, ...extra });
  }
  addMesh(group, geo, color, role, extra = {}) {
    if (!geo || geo.empty) return null;
    const m = new THREE.Mesh(geo.geometry(), this.material(color));
    m.userData = { role, ...extra };
    group.add(m);
    return m;
  }
  dropObj(id) {
    const g = this.objs.get(id);
    if (!g) return;
    g.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    this.root.remove(g);
    this.objs.delete(id);
  }
  buildObj(id) {
    this.dropObj(id);
    const el = this.els.get(id);
    if (!el) return;
    const g = new THREE.Group();
    g.userData = { id };
    if (el.kind === "wall") {
      const { wall, glass, door, ops } = wallGeo(el, this.settings.doorLeaves);
      this.addMesh(g, wall, el.color, "body");
      this.addMesh(g, glass, C.glass, "glass");
      this.addMesh(g, door, C.door, "door");
      const { u, n } = wallFrame(el.a, el.b);
      for (const o of ops) {                    // pick proxies: an opening can be clicked in the hole
        const p = new Geo();
        p.box(el.a, u, n, o.s0, o.s1, el.thickness * 1.02, el.z0 + Math.max(0, o.z0), el.z0 + Math.min(o.z1, el.height));
        const sel = this.sel.has(`op:${id}:${o.id}`);
        const mat = new THREE.MeshBasicMaterial({ color: C.select, transparent: true, opacity: sel ? 0.45 : 0, depthWrite: false, colorWrite: sel });
        const pm = new THREE.Mesh(p.geometry(), mat);
        pm.userData = { role: "opening", op: o.id };
        g.add(pm);
      }
    } else if (el.kind === "railing") {
      const { wall, glass } = railingGeo(el);
      this.addMesh(g, wall, el.color, "body");
      this.addMesh(g, glass, C.glass, "glass");
    } else if (el.kind === "prism") {
      const geo = new Geo(); geo.prism(el.ring, el.z0, el.z1);
      this.addMesh(g, geo, el.color, "body");
    } else if (el.kind === "roof") {
      this.addMesh(g, roofGeo(el), el.color, "body");
    } else if (el.kind === "mesh") {
      const f = this.fixed.get(id);
      if (f) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(f.positions, 3));
        geo.setIndex(f.indices);
        geo.computeVertexNormals();
        const m = new THREE.Mesh(geo, this.material(el.color));
        m.userData = { role: el.type === "window" ? "glass" : "body" };
        m.matrixAutoUpdate = false;
        m.matrix.fromArray(el.matrix);
        g.add(m);
      }
    }
    if (this.sel.has(id)) for (const m of g.children) if (m.material && m.material.emissive && m.userData.role !== "glass") m.material.emissive.setHex(0x6b3000);
    this.root.add(g);
    this.objs.set(id, g);
    this.visibilityOf(id);
  }

  // ---------------------------------------------------------------- visibility (the 3D tab's check boxes)
  setVisible(kind, on) { this.visible[kind] = on; this.applyVisibility(); }
  visibilityOf(id) {
    const g = this.objs.get(id), el = this.els.get(id);
    if (!g || !el) return;
    const v = this.visible, rank = this.levelRank.get(el.level) ?? -1;
    g.visible = !this.hidden.has(id) && !(el.type === "floor" && !v.floor) && !(el.type === "roof" && !v.roof) &&
      !(el.review && !v.review) && !(rank > v.maxLevel);
    for (const m of g.children) if (m.userData.role === "glass") m.visible = v.window;
    if (el.kind === "mesh" && el.type === "window") g.visible = g.visible && v.window;
  }
  applyVisibility() { for (const id of this.objs.keys()) this.visibilityOf(id); }

  // ---------------------------------------------------------------- selection
  highlight(id) {                               // selected in the 2D view / details panel
    if (this.mode) return;
    const target = this.els.has(id) ? id : this.opKey(id);
    this.setSelection(target ? [target] : [], false);
  }
  opKey(opId) {
    for (const el of this.els.values()) if (el.kind === "wall" && (el.openings || []).some((o) => o.id === opId)) return `op:${el.id}:${opId}`;
    return null;
  }
  setSelection(ids, notify = true) {
    const before = new Set(this.wallsOf(this.sel));
    this.sel = new Set(ids);
    for (const id of new Set([...before, ...this.wallsOf(this.sel)])) if (this.els.has(id)) this.buildObj(id);
    this.refreshUi();
    if (notify && this.onPick && this.sel.size === 1) {
      const k = [...this.sel][0];
      this.onPick(k.startsWith("op:") ? k.split(":")[2] : k);
    }
  }
  wallsOf(ids) { return [...ids].map((k) => (k.startsWith("op:") ? k.split(":")[1] : k)); }
  opening(key) {
    const [, wid, oid] = key.split(":");
    const w = this.els.get(wid);
    return w ? { wall: w, op: (w.openings || []).find((o) => o.id === oid) } : {};
  }

  // ---------------------------------------------------------------- picking and snapping
  ndc(ev) {
    const r = this.canvas.getBoundingClientRect();
    return new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  }
  hit(ev, filter = null) {
    this.ray.setFromCamera(this.ndc(ev), this.camera);
    const objs = [];
    for (const [id, g] of this.objs) if (g.visible && (!filter || filter(this.els.get(id)))) g.traverse((o) => { if (o.isMesh && o.visible !== false) objs.push(o); });
    const hits = this.ray.intersectObjects(objs, false);
    for (const h of hits) {
      const id = h.object.parent.userData.id, role = h.object.userData.role;
      if (role === "opening" && !filter) return { id, op: h.object.userData.op, point: this.toModel(h.point), face: h.face, object: h.object };
      if (role !== "opening") return { id, point: this.toModel(h.point), face: h.face, object: h.object };
    }
    return null;
  }
  toModel(v) { return this.root.worldToLocal(v.clone()); }
  planePoint(ev, z) {                           // the mouse ray on the horizontal plane at height z (model)
    if (!ev) return null;
    this.ray.setFromCamera(this.ndc(ev), this.camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -z);
    const p = new THREE.Vector3();
    if (!this.ray.ray.intersectPlane(plane, p)) return null;
    return this.toModel(p);
  }
  screenOf(x, y, z) {
    const v = this.root.localToWorld(new THREE.Vector3(x, y, z)).project(this.camera);
    const r = this.canvas.getBoundingClientRect();
    return [(v.x + 1) / 2 * r.width, (1 - v.y) / 2 * r.height];
  }
  snapPoint(ev, z, invert = false) {
    const p = this.planePoint(ev, z);
    if (!p) return null;
    const useSnap = this.settings.snap !== invert;
    if (!useSnap) return { p: [p.x, p.y], snapped: false };
    const r = this.canvas.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = null;
    const tryPt = (q) => {
      const [sx, sy] = this.screenOf(q[0], q[1], z);
      const d = Math.hypot(sx - mx, sy - my);
      if (d < this.settings.snapPx && (!best || d < best.d)) best = { d, q };
    };
    for (const el of this.els.values()) {
      if (this.hidden.has(el.id)) continue;
      if (el.kind === "wall" || el.kind === "railing") { tryPt(el.a); tryPt(el.b); }
      else if ((el.kind === "prism" || el.kind === "roof") && el.ring.length < 64) el.ring.forEach(tryPt);
    }
    if (this.mode && this.mode.pts) this.mode.pts.forEach(tryPt);
    if (best) return { p: [...best.q], snapped: true };
    const gs = this.settings.grid || 0;
    return { p: gs ? [Math.round(p.x / gs) * gs, Math.round(p.y / gs) * gs] : [p.x, p.y], snapped: false };
  }

  // ---------------------------------------------------------------- undo / redo
  snapshot() { return clone([...this.els.values()]); }
  restore(snap) {
    const before = new Map([...this.els].map(([k, v]) => [k, JSON.stringify(v)]));
    this.els = new Map(snap.map((e) => [e.id, e]));
    for (const id of new Set([...before.keys(), ...this.els.keys()])) {
      const now = this.els.get(id);
      if (!now) this.dropObj(id);
      else if (before.get(id) !== JSON.stringify(now)) this.buildObj(id);
    }
    this.sel = new Set([...this.sel].filter((k) => this.els.has(this.wallsOf([k])[0])));
  }
  commit(label) {
    this.undoStack.push({ label, snap: this.pending || this.snapshot() });
    if (this.undoStack.length > 200) this.undoStack.shift();
    this.redoStack = [];
    this.pending = null;
    this.dirty = true;
    this.say(label);
    this.refreshUi();
  }
  begin() { this.pending = this.snapshot(); }   // state before an operation (pushed by commit)
  undo() {
    const u = this.undoStack.pop();
    if (!u) return this.say("Nothing to undo");
    this.redoStack.push({ label: u.label, snap: this.snapshot() });
    this.restore(u.snap);
    this.say(`Undone: ${u.label}`); this.refreshUi();
  }
  redo() {
    const r = this.redoStack.pop();
    if (!r) return this.say("Nothing to redo");
    this.undoStack.push({ label: r.label, snap: this.snapshot() });
    this.restore(r.snap);
    this.say(`Redone: ${r.label}`); this.refreshUi();
  }
  newId(kind) { this.seq += 1; return `edit_${kind}_${String(this.seq).padStart(3, "0")}`; }

  // ---------------------------------------------------------------- modal transforms (G / R / S)
  pivotOf(keys) {
    let xs = [], ys = [], zs = [];
    for (const k of keys) {
      if (k.startsWith("op:")) {
        const { wall, op } = this.opening(k);
        if (!op) continue;
        const { u } = wallFrame(wall.a, wall.b), s = (op.s0 + op.s1) / 2;
        xs.push(wall.a[0] + u[0] * s); ys.push(wall.a[1] + u[1] * s); zs.push(wall.z0 + op.z0);
        continue;
      }
      const el = this.els.get(k);
      if (!el) continue;
      if (el.kind === "mesh") {
        const b = el.box, m = new THREE.Matrix4().fromArray(el.matrix);
        const c = new THREE.Vector3((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, b[2]).applyMatrix4(m);
        xs.push(c.x); ys.push(c.y); zs.push(c.z);
      } else {
        for (const p of pointsOf(el)) { xs.push(p[0]); ys.push(p[1]); }
        zs.push(el.z0 ?? 0);
      }
    }
    if (!xs.length) return null;
    return [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2, Math.min(...zs)];
  }
  startTransform(type, ev) {
    if (!this.sel.size) return this.say("Select something first (click it; Shift+click adds)");
    const pivot = this.pivotOf(this.sel);
    if (!pivot) return;
    const start = this.planePoint(ev || this.lastEv, pivot[2]);
    this.mode = { type, pivot, start: start ? [start.x, start.y] : [pivot[0], pivot[1]], mouse0: [this.mouse.x, this.mouse.y],
      axis: null, num: "", base: this.snapshot() };
    this.pending = this.mode.base;
    this.controls.enabled = false;
    this.refreshUi();
  }
  updateTransform(ev, invertSnap) {
    const m = this.mode;
    this.restore(clone(m.base));
    const typed = m.num !== "" && m.num !== "-" && m.num !== "." ? parseFloat(m.num) : null;
    const grid = this.settings.grid || 0.01;
    if (m.type === "grab") {
      let d = [0, 0, 0];
      const onlyOpenings = [...this.sel].every((k) => k.startsWith("op:"));
      if (typed !== null && onlyOpenings && m.axis !== "z") {          // a door / window: the typed distance is along its wall
        m.value = typed;
        this.applyMove([0, 0, 0], typed);
        this.say(`Slide along the wall ${typed.toFixed(3)} m  [${m.num}]`, true);
        return;
      }
      if (typed !== null) {
        d = m.axis === "y" ? [0, typed, 0] : m.axis === "z" ? [0, 0, typed] : [typed, 0, 0];
      } else if (m.axis === "z") {
        const r = this.canvas.getBoundingClientRect(), dist = this.camera.position.distanceTo(this.controls.target);
        d = [0, 0, (m.mouse0[1] - this.mouse.y) / r.height * dist];
      } else {
        const p = this.planePoint(ev || this.lastEv, m.pivot[2]);
        if (p) d = [p.x - m.start[0], p.y - m.start[1], 0];
        if (m.axis === "x") d[1] = 0;
        if (m.axis === "y") d[0] = 0;
      }
      const snap = this.settings.snap === !!invertSnap ? 0 : grid;   // Ctrl toggles grid snapping
      if (snap && typed === null) d = d.map((x) => Math.round(x / snap) * snap);
      m.value = d;
      this.applyMove(d);
      this.say(`Move ${m.axis ? m.axis.toUpperCase() + " " : ""}Δ ${d.map((x) => x.toFixed(3)).join(", ")} m${m.num ? `  [${m.num}]` : ""}`, true);
    } else if (m.type === "rotate") {
      let ang;
      if (typed !== null) ang = typed * Math.PI / 180;
      else {
        const [px, py] = this.screenOf(m.pivot[0], m.pivot[1], m.pivot[2]);
        const a0 = Math.atan2(m.mouse0[1] - py, m.mouse0[0] - px), a1 = Math.atan2(this.mouse.y - py, this.mouse.x - px);
        ang = -(a1 - a0);
        const step = (this.settings.angleSnap || 15) * Math.PI / 180;
        if (this.settings.snap !== !!invertSnap) ang = Math.round(ang / step) * step;
      }
      m.value = ang;
      this.applyRotate(ang);
      this.say(`Rotate ${(ang * 180 / Math.PI).toFixed(1)}°${m.num ? `  [${m.num}]` : ""}`, true);
    } else if (m.type === "scale") {
      let f;
      if (typed !== null) f = typed;
      else {
        const [px, py] = this.screenOf(m.pivot[0], m.pivot[1], m.pivot[2]);
        const d0 = Math.max(Math.hypot(m.mouse0[0] - px, m.mouse0[1] - py), 1);
        f = Math.hypot(this.mouse.x - px, this.mouse.y - py) / d0;
        if (this.settings.snap !== !!invertSnap) f = Math.round(f * 20) / 20;
      }
      f = Math.max(f, 0.01);
      m.value = f;
      this.applyScale(f, m.axis);
      this.say(`Scale ${m.axis ? m.axis.toUpperCase() + " " : ""}× ${f.toFixed(3)}${m.num ? `  [${m.num}]` : ""}`, true);
    }
  }
  applyMove([dx, dy, dz], along = null) {
    const touched = new Set();
    for (const k of this.sel) {
      if (k.startsWith("op:")) {                  // an opening slides along its wall; Z moves its sill / head
        const { wall, op } = this.opening(k);
        if (!op || this.sel.has(wall.id)) continue;
        const { u, L } = wallFrame(wall.a, wall.b), ds = along !== null ? along : dx * u[0] + dy * u[1], w = op.s1 - op.s0;
        op.s0 = Math.max(0, Math.min(L - w, op.s0 + ds)); op.s1 = op.s0 + w;
        if (dz) { const h = op.z1 - op.z0; op.z0 = Math.max(0, Math.min(wall.height - h, op.z0 + dz)); op.z1 = op.z0 + h; }
        touched.add(wall.id);
        continue;
      }
      const el = this.els.get(k);
      if (!el) continue;
      if (el.kind === "mesh") el.matrix = new THREE.Matrix4().makeTranslation(dx, dy, dz).multiply(new THREE.Matrix4().fromArray(el.matrix)).toArray();
      else { mapPoints(el, (p) => [p[0] + dx, p[1] + dy]); el.z0 += dz; if (el.z1 !== undefined) el.z1 += dz; }
      touched.add(k);
    }
    for (const id of touched) this.buildObj(id);
  }
  applyRotate(ang) {
    const c = this.mode ? this.mode.pivot : this.pivotOf(this.sel);
    for (const k of this.sel) {
      if (k.startsWith("op:")) continue;
      const el = this.els.get(k);
      if (!el) continue;
      if (el.kind === "mesh") {
        const T = new THREE.Matrix4().makeTranslation(c[0], c[1], 0), R = new THREE.Matrix4().makeRotationZ(ang), Ti = new THREE.Matrix4().makeTranslation(-c[0], -c[1], 0);
        el.matrix = T.multiply(R).multiply(Ti).multiply(new THREE.Matrix4().fromArray(el.matrix)).toArray();
      } else {
        mapPoints(el, (p) => rot(p, c, ang));
        if (el.kind === "roof" && el.dir !== undefined) el.dir += ang * 180 / Math.PI;
      }
      this.buildObj(k);
    }
  }
  applyScale(f, axis) {
    const c = this.mode ? this.mode.pivot : this.pivotOf(this.sel), touched = new Set();
    for (const k of this.sel) {
      if (k.startsWith("op:")) {                  // an opening: width (or height with Z) about its centre
        const { wall, op } = this.opening(k);
        if (!op) continue;
        if (axis === "z") { const h = (op.z1 - op.z0) * f; op.z1 = Math.min(wall.height, op.z0 + h); }
        else { const { L } = wallFrame(wall.a, wall.b), m = (op.s0 + op.s1) / 2, w = (op.s1 - op.s0) * f; op.s0 = Math.max(0, m - w / 2); op.s1 = Math.min(L, m + w / 2); }
        touched.add(wall.id);
        continue;
      }
      const el = this.els.get(k);
      if (!el) continue;
      if (axis === "z") {
        if (el.kind === "wall") el.height *= f;
        else if (el.kind === "railing") { el.parapet *= f; el.top *= f; }
        else if (el.kind === "prism") el.z1 = el.z0 + (el.z1 - el.z0) * f;
        else if (el.kind === "roof") el.pitch = Math.min(80, el.pitch * f);
        else if (el.kind === "mesh") {
          const T = new THREE.Matrix4().makeTranslation(0, 0, c[2]), S = new THREE.Matrix4().makeScale(1, 1, f), Ti = new THREE.Matrix4().makeTranslation(0, 0, -c[2]);
          el.matrix = T.multiply(S).multiply(Ti).multiply(new THREE.Matrix4().fromArray(el.matrix)).toArray();
        }
      } else {
        const sx = axis === "y" ? 1 : f, sy = axis === "x" ? 1 : f;
        if (el.kind === "mesh") {
          const T = new THREE.Matrix4().makeTranslation(c[0], c[1], 0), S = new THREE.Matrix4().makeScale(sx, sy, 1), Ti = new THREE.Matrix4().makeTranslation(-c[0], -c[1], 0);
          el.matrix = T.multiply(S).multiply(Ti).multiply(new THREE.Matrix4().fromArray(el.matrix)).toArray();
        } else {
          const L0 = el.kind === "wall" ? hyp(el.a, el.b) : 0;
          mapPoints(el, (p) => [c[0] + (p[0] - c[0]) * sx, c[1] + (p[1] - c[1]) * sy]);
          if (el.kind === "wall" && L0) {           // openings keep their width, their centres follow
            const L1 = hyp(el.a, el.b), r = L1 / L0;
            for (const o of el.openings || []) { const m = (o.s0 + o.s1) / 2 * r, w = o.s1 - o.s0; o.s0 = Math.max(0, m - w / 2); o.s1 = Math.min(L1, m + w / 2); }
          }
        }
      }
      touched.add(k);
    }
    for (const id of touched) this.buildObj(id);
  }
  confirmMode() {
    const m = this.mode;
    if (!m) return;
    if (["grab", "rotate", "scale"].includes(m.type)) {
      const what = { grab: "Moved", rotate: "Rotated", scale: "Scaled" }[m.type];
      this.mode = null; this.controls.enabled = true;
      this.commit(`${what} ${this.sel.size} element(s)`);
      for (const id of this.wallsOf(this.sel)) this.buildObj(id);
    }
  }
  endMode(silent = false) {
    const m = this.mode;
    if (m && m.base && ["grab", "rotate", "scale"].includes(m.type)) {
      this.restore(clone(m.base)); this.pending = null;
      if (m.pre) { this.mode = null; this.pending = m.pre; this.commit("Duplicated (move cancelled)"); }
    }
    this.mode = null;
    this.controls.enabled = true;
    this.clearOverlay();
    if (this.grid) this.grid.visible = false;
    if (!silent) { this.say("Cancelled"); this.refreshUi(); }
  }
  clearOverlay() {
    for (const o of [...this.overlay.children]) {
      if (o === this.grid) continue;
      this.overlay.remove(o);
      o.traverse((x) => { if (x.geometry) x.geometry.dispose(); if (x.material) x.material.dispose(); });
    }
  }

  // ---------------------------------------------------------------- editing commands
  deleteSelection() {
    if (!this.sel.size) return;
    this.begin();
    let n = 0;
    for (const k of this.sel) {
      if (k.startsWith("op:")) {
        const { wall, op } = this.opening(k);
        if (wall && op) { wall.openings = wall.openings.filter((o) => o !== op); this.buildObj(wall.id); n++; }
      } else if (this.els.has(k)) { this.els.delete(k); this.dropObj(k); n++; }
    }
    this.sel.clear();
    this.commit(`Deleted ${n} element(s) (a deleted door / window fills its gap)`);
  }
  duplicate(ev) {
    const ids = [...this.sel].filter((k) => !k.startsWith("op:") && this.els.has(k));
    if (!ids.length) return;
    const pre = this.snapshot(), out = [];
    for (const k of ids) {
      const el = clone(this.els.get(k)), id = this.newId(el.kind);
      el.id = id;
      if (el.kind === "mesh") this.fixed.set(id, this.fixed.get(k));
      if (el.openings) el.openings = el.openings.map((o) => ({ ...o, id: this.newId(o.type) }));
      this.els.set(id, el); this.buildObj(id); out.push(id);
    }
    this.setSelection(out, false);
    this.startTransform("grab", ev);           // one undo step for duplicate + move; cancelling the move keeps the copies
    this.mode.pre = pre;
    this.pending = pre;
  }
  fillGap() {
    const keys = [...this.sel];
    const ops = keys.filter((k) => k.startsWith("op:"));
    if (ops.length) return this.deleteSelection();
    const walls = keys.map((k) => this.els.get(k)).filter((e) => e && e.kind === "wall");
    if (walls.length === 2) {
      const [p, q] = walls, fp = wallFrame(p.a, p.b), fq = wallFrame(q.a, q.b);
      const par = Math.abs(fp.u[0] * fq.u[0] + fp.u[1] * fq.u[1]) > 0.995;
      this.begin();
      if (par) {                                 // collinear pieces: one wall over both, openings kept
        const off = Math.abs((q.a[0] - p.a[0]) * fp.n[0] + (q.a[1] - p.a[1]) * fp.n[1]);
        if (off > Math.max(p.thickness, q.thickness)) { this.pending = null; return this.say("These walls are parallel but not in line: nothing to join"); }
        const s = (r) => (r[0] - p.a[0]) * fp.u[0] + (r[1] - p.a[1]) * fp.u[1];
        const ss = [0, fp.L, s(q.a), s(q.b)], lo = Math.min(...ss), hi = Math.max(...ss);
        const A = [p.a[0] + fp.u[0] * lo, p.a[1] + fp.u[1] * lo], B = [p.a[0] + fp.u[0] * hi, p.a[1] + fp.u[1] * hi];
        const remap = (w) => (w.openings || []).map((o) => {
          const fw = wallFrame(w.a, w.b), m0 = [w.a[0] + fw.u[0] * o.s0, w.a[1] + fw.u[1] * o.s0], m1 = [w.a[0] + fw.u[0] * o.s1, w.a[1] + fw.u[1] * o.s1];
          const a = s(m0) - lo, b = s(m1) - lo;
          return { ...o, s0: Math.min(a, b), s1: Math.max(a, b) };
        });
        p.openings = [...remap(p), ...remap(q)];
        p.a = A; p.b = B; p.height = Math.max(p.height, q.height); p.thickness = Math.max(p.thickness, q.thickness);
        this.els.delete(q.id); this.dropObj(q.id); this.buildObj(p.id);
        this.sel = new Set([p.id]);
        return this.commit("Joined two walls in line (gap filled)");
      }
      const X = this.intersect(p, q);              // a corner: both walls run to where their axes meet
      if (!X) { this.pending = null; return this.say("No corner found between these walls"); }
      for (const w of [p, q]) {
        const da = hyp(w.a, X), db = hyp(w.b, X), f = wallFrame(w.a, w.b);
        if (da < db) {
          const shift = (X[0] - w.a[0]) * f.u[0] + (X[1] - w.a[1]) * f.u[1];
          w.a = X;
          for (const o of w.openings || []) { o.s0 -= shift; o.s1 -= shift; }
        } else w.b = X;
        this.buildObj(w.id);
      }
      return this.commit("Extended two walls to their corner (gap filled)");
    }
    if (walls.length === 1) {                      // one wall: its nearest free end runs to the wall in front of it
      const w = walls[0], f = wallFrame(w.a, w.b);
      let best = null;
      for (const end of ["a", "b"]) {
        const dir = end === "b" ? f.u : [-f.u[0], -f.u[1]], o = w[end];
        for (const q of this.els.values()) {
          if (q === w || q.kind !== "wall") continue;
          const X = this.intersect(w, q);
          if (!X) continue;
          const t = (X[0] - o[0]) * dir[0] + (X[1] - o[1]) * dir[1];
          const fq = wallFrame(q.a, q.b), sq = (X[0] - q.a[0]) * fq.u[0] + (X[1] - q.a[1]) * fq.u[1];
          if (t > 1e-3 && t < 6 && sq > -0.3 && sq < fq.L + 0.3 && (!best || t < best.t)) best = { t, end, X };
        }
      }
      if (!best) return this.say("No wall within 6 m in line with this wall's ends");
      this.begin();
      if (best.end === "a") {
        const shift = -best.t;
        w.a = best.X;
        for (const o of w.openings || []) { o.s0 -= shift; o.s1 -= shift; }
      } else w.b = best.X;
      this.buildObj(w.id);
      return this.commit(`Extended the wall by ${best.t.toFixed(2)} m to the next wall (gap filled)`);
    }
    this.say("Fill gap: select a door / window (closes it), two walls (joins them) or one wall (extends it)");
  }
  intersect(p, q) {
    const fp = wallFrame(p.a, p.b), fq = wallFrame(q.a, q.b);
    const den = fp.u[0] * fq.u[1] - fp.u[1] * fq.u[0];
    if (Math.abs(den) < 1e-6) return null;
    const t = ((q.a[0] - p.a[0]) * fq.u[1] - (q.a[1] - p.a[1]) * fq.u[0]) / den;
    return [p.a[0] + fp.u[0] * t, p.a[1] + fp.u[1] * t];
  }
  hide() { for (const k of this.wallsOf(this.sel)) { this.hidden.add(k); this.visibilityOf(k); } this.setSelection([], false); this.say("Hidden (Alt+H shows them again)"); }
  unhide() { const ids = [...this.hidden]; this.hidden.clear(); for (const k of ids) this.visibilityOf(k); this.say(`${ids.length} element(s) shown again`); }
  frameSelected() {
    const ids = this.sel.size ? this.wallsOf(this.sel) : [...this.objs.keys()];
    const b = new THREE.Box3();
    for (const id of ids) { const g = this.objs.get(id); if (g && g.visible) b.expandByObject(g); }
    if (b.isEmpty()) return;
    const lo = this.toModel(b.min), hi = this.toModel(b.max);
    this.frame([Math.min(lo.x, hi.x), Math.min(lo.y, hi.y), Math.min(lo.z, hi.z)], [Math.max(lo.x, hi.x), Math.max(lo.y, hi.y), Math.max(lo.z, hi.z)]);
  }

  // ---------------------------------------------------------------- drawing tools
  startTool(type) {
    this.endMode(true);
    this.mode = { type, pts: [], roofStyle: this.settings.roofStyle };
    if (this.grid) { this.grid.position.z = this.settings.workZ - 0.002; this.grid.visible = ["wall", "floor", "roof"].includes(type); }
    this.refreshUi();
  }
  toolMove(ev, invertSnap) {
    const m = this.mode;
    this.clearOverlay();
    if (m.type === "door" || m.type === "window") return this.openingGhost(ev, false);
    const sp = this.snapPoint(ev, this.settings.workZ, invertSnap);
    if (!sp) return;
    let p = sp.p;
    if (ev.shiftKey && m.pts.length) p = this.ortho(m.pts[m.pts.length - 1], p);
    m.cursor = p;
    const z = this.settings.workZ + 0.01;
    const pts = [...m.pts, p];
    if (m.type === "wall") {
      for (let k = 1; k < pts.length; k++) this.ghostWall(pts[k - 1], pts[k]);
      if (pts.length > 1) this.say(`Wall ${hyp(pts[pts.length - 2], p).toFixed(2)} m — click the next corner; Enter / right-click ends; Shift: 90° steps; Ctrl: snapping ${this.settings.snap ? "off" : "on"}`, true);
    } else {
      const line = new THREE.BufferGeometry().setFromPoints([...pts, ...(pts.length > 2 ? [pts[0]] : [])].map((q) => new THREE.Vector3(q[0], q[1], z)));
      this.overlay.add(new THREE.Line(line, new THREE.LineBasicMaterial({ color: C.ghost })));
      if (pts.length >= 3) {
        const g = new Geo(); g.prism(pts, this.settings.workZ, this.settings.workZ + 0.02);
        if (!g.empty) this.overlay.add(new THREE.Mesh(g.geometry(), new THREE.MeshBasicMaterial({ color: C.ghost, transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false })));
      }
      if (m.type === "roof") this.say(`Roof (${m.roofStyle}, ${this.settings.roofPitch}°): click the outline corners, Enter finishes; ${this.keymap.cycle_roof.toUpperCase()}: change style`, true);
    }
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 8), new THREE.MeshBasicMaterial({ color: sp.snapped ? C.select : C.ghost }));
    dot.position.set(p[0], p[1], z);
    this.overlay.add(dot);
  }
  ortho(a, p) {
    const dx = p[0] - a[0], dy = p[1] - a[1], ang = Math.atan2(dy, dx), step = Math.PI / 4;
    const q = Math.round(ang / step) * step, L = Math.hypot(dx, dy);
    return [a[0] + Math.cos(q) * L, a[1] + Math.sin(q) * L];
  }
  ghostWall(a, b) {
    if (hyp(a, b) < 1e-3) return;
    const g = new Geo(), f = wallFrame(a, b);
    g.box(a, f.u, f.n, 0, f.L, this.settings.wallThickness, this.settings.workZ, this.settings.workZ + this.settings.wallHeight);
    this.overlay.add(new THREE.Mesh(g.geometry(), new THREE.MeshStandardMaterial({ color: C.ghost, transparent: true, opacity: 0.4, depthWrite: false })));
  }
  toolClick(ev, invertSnap) {
    const m = this.mode;
    if (m.type === "door" || m.type === "window") return this.openingGhost(ev, true);
    if (!m.pts.length) {                         // the first point picks the working height from a surface under it
      const h = this.hit(ev);
      if (h && h.face && h.face.normal && Math.abs(h.object.localToWorld(h.face.normal.clone()).sub(h.object.localToWorld(new THREE.Vector3())).normalize().y) > 0.8) {
        this.settings.workZ = +h.point.z.toFixed(3);
        if (this.grid) this.grid.position.z = this.settings.workZ - 0.002;
      }
    }
    const sp = this.snapPoint(ev, this.settings.workZ, invertSnap);
    if (!sp) return;
    let p = sp.p;
    if (ev.shiftKey && m.pts.length) p = this.ortho(m.pts[m.pts.length - 1], p);
    if (m.type === "wall" && m.pts.length) {
      const a = m.pts[m.pts.length - 1];
      if (hyp(a, p) > 0.05) {
        this.begin();
        const id = this.newId("wall");
        this.els.set(id, { id, kind: "wall", type: "wall", cls: "wall", color: C.wall, level: this.levelAt(this.settings.workZ),
          a, b: p, thickness: this.settings.wallThickness, z0: this.settings.workZ, height: this.settings.wallHeight, openings: [] });
        this.buildObj(id);
        this.commit(`Wall ${hyp(a, p).toFixed(2)} m`);
      }
      if (m.pts.length > 1 && hyp(m.pts[0], p) < 1e-6) return this.finishTool();   // closed the loop
    } else if (m.pts.length >= 3 && hyp(m.pts[0], p) < 1e-6) return this.finishTool();
    m.pts.push(p);
    this.toolMove(ev, invertSnap);
  }
  finishTool() {
    const m = this.mode;
    if (!m) return;
    if ((m.type === "floor" || m.type === "roof") && m.pts.length >= 3) {
      this.begin();
      const id = this.newId(m.type), z = this.settings.workZ;
      if (m.type === "floor") {
        this.els.set(id, { id, kind: "prism", type: "floor", cls: "floor", color: C.floor, level: this.levelAt(z), ring: m.pts, z0: z - this.settings.slabThickness, z1: z });
      } else {
        const inside = (q) => {                    // point in polygon (ray casting)
          let c = false;
          for (let i = 0, j = m.pts.length - 1; i < m.pts.length; j = i++) {
            const [xi, yi] = m.pts[i], [xj, yj] = m.pts[j];
            if ((yi > q[1]) !== (yj > q[1]) && q[0] < (xj - xi) * (q[1] - yi) / (yj - yi) + xi) c = !c;
          }
          return c;
        };
        let top = -Infinity;                       // eaves on the walls under the outline (not on the clicked floor)
        for (const w of this.els.values()) if (w.kind === "wall" && inside([(w.a[0] + w.b[0]) / 2, (w.a[1] + w.b[1]) / 2])) top = Math.max(top, w.z0 + w.height);
        const roofZ = Number.isFinite(top) ? top : z + this.settings.wallHeight;
        const longest = m.pts.reduce((b, p, k) => { const q = m.pts[(k + 1) % m.pts.length], L = hyp(p, q); return L > b.L ? { L, ang: Math.atan2(q[1] - p[1], q[0] - p[0]) } : b; }, { L: 0, ang: 0 });
        this.els.set(id, { id, kind: "roof", type: "roof", cls: "roof", color: C.roof, level: this.levelAt(z), ring: m.pts, z0: roofZ, style: m.roofStyle,
          pitch: this.settings.roofPitch, thickness: this.settings.roofThickness, overhang: this.settings.roofOverhang, dir: longest.ang * 180 / Math.PI + 90 });
      }
      this.buildObj(id);
      this.commit(m.type === "floor" ? "Floor slab added" : `Roof added (${m.roofStyle})`);
      this.endMode(true);
      this.setSelection([id], false);
      return;
    }
    this.endMode(true);
    this.refreshUi();
  }
  levelAt(z) {
    let best = null;
    for (const l of this.levels || []) if (l.z <= z + 0.3 && (!best || l.z > best.z)) best = l;
    return best ? best.id : null;
  }
  openingGhost(ev, place) {
    const m = this.mode, isDoor = m.type === "door";
    this.clearOverlay();
    const h = this.hit(ev, (el) => el && el.kind === "wall");
    if (!h) return this.say(`${isDoor ? "Door" : "Window"}: point at a wall`, true);
    const w = this.els.get(h.id), f = wallFrame(w.a, w.b);
    const width = Math.min(isDoor ? this.settings.doorWidth : this.settings.windowWidth, f.L - 0.1);
    let s = (h.point.x - w.a[0]) * f.u[0] + (h.point.y - w.a[1]) * f.u[1];
    const gs = this.settings.snap !== ev.ctrlKey ? (this.settings.grid || 0) : 0;
    let s0 = Math.max(0.05, Math.min(f.L - 0.05 - width, s - width / 2));
    if (gs) s0 = Math.round(s0 / gs) * gs;
    const s1 = s0 + width;
    const z0 = isDoor ? 0 : this.settings.windowSill, z1 = Math.min(w.height, isDoor ? this.settings.doorHeight : this.settings.windowHead);
    const clash = (w.openings || []).some((o) => Math.min(o.s1, s1) - Math.max(o.s0, s0) > 0.01);
    const g = new Geo(); g.box(w.a, f.u, f.n, s0, s1, w.thickness * 1.06, w.z0 + z0, w.z0 + z1);
    this.overlay.add(new THREE.Mesh(g.geometry(), new THREE.MeshBasicMaterial({ color: clash ? C.bad : C.ghost, transparent: true, opacity: 0.45, depthWrite: false })));
    if (!place) return this.say(`${isDoor ? "Door" : "Window"} ${width.toFixed(2)} m at ${s0.toFixed(2)} m from the wall's end — click to cut${clash ? " (overlaps an opening)" : ""}`, true);
    if (clash) return this.say("That place overlaps an existing opening");
    this.begin();
    const id = this.newId(isDoor ? "door" : "window");
    w.openings = [...(w.openings || []), { id, type: isDoor ? "door" : "window", s0, s1, z0, z1 }];
    this.buildObj(w.id);
    this.commit(`${isDoor ? "Door" : "Window"} cut into ${w.id}`);
  }

  // ---------------------------------------------------------------- events
  bindEvents() {
    const cv = this.canvas;
    cv.addEventListener("contextmenu", (e) => e.preventDefault());
    cv.addEventListener("pointerdown", (ev) => {
      cv.focus();
      this.mouse.down = { x: ev.clientX, y: ev.clientY, b: ev.button, alt: ev.altKey };
      if (this.settings.nav !== "classic" && ev.altKey && ev.button === 0) this.controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    }, true);
    cv.addEventListener("pointermove", (ev) => {
      const r = cv.getBoundingClientRect();
      this.mouse.x = ev.clientX - r.left; this.mouse.y = ev.clientY - r.top; this.lastEv = ev;
      const m = this.mode;
      if (!m) return;
      if (["grab", "rotate", "scale"].includes(m.type)) this.updateTransform(ev, ev.ctrlKey);
      else this.toolMove(ev, ev.ctrlKey);
    });
    cv.addEventListener("pointerup", (ev) => {
      const d = this.mouse.down;
      this.mouse.down = null;
      if (this.settings.nav !== "classic") this.applyNav();
      if (!d || Math.hypot(ev.clientX - d.x, ev.clientY - d.y) > 4 || d.alt) return;   // a drag (orbit / pan), not a click
      const m = this.mode;
      if (ev.button === 2) { if (m && m.pts && m.type !== "door" && m.type !== "window") this.finishTool(); else if (m) this.endMode(); return; }
      if (ev.button !== 0) return;
      if (m && ["grab", "rotate", "scale"].includes(m.type)) return this.confirmMode();
      if (m) return this.toolClick(ev, ev.ctrlKey);
      const h = this.hit(ev);
      const key = h ? (h.op ? `op:${h.id}:${h.op}` : h.id) : null;
      if (ev.shiftKey && key) {
        const s = new Set(this.sel);
        s.has(key) ? s.delete(key) : s.add(key);
        this.setSelection([...s]);
      } else this.setSelection(key ? [key] : []);
    });
    cv.addEventListener("dblclick", () => { if (this.mode && this.mode.pts) this.finishTool(); });
    cv.addEventListener("keydown", (ev) => this.onKey(ev));
  }
  onKey(ev) {
    if (ev.target !== this.canvas) return;
    const k = keyOf(ev);
    if (!k) return;
    const km = this.keymap, m = this.mode, is = (a) => km[a] === k;
    const transforming = m && ["grab", "rotate", "scale"].includes(m.type);
    if (transforming) {
      if (k === "escape") { this.endMode(); }
      else if (k === "enter") { this.confirmMode(); }
      else if (is("axis_x") || is("axis_y") || is("axis_z")) { const a = is("axis_x") ? "x" : is("axis_y") ? "y" : "z"; m.axis = m.axis === a ? null : a; this.updateTransform(this.lastEv); }
      else if (/^[0-9.]$/.test(k) || k === "-" || k === "backspace") {
        m.num = k === "backspace" ? m.num.slice(0, -1) : k === "-" ? (m.num.startsWith("-") ? m.num.slice(1) : "-" + m.num) : m.num + k;
        this.updateTransform(this.lastEv);
      } else return;
      ev.preventDefault();
      return;
    }
    if (m && k === "escape") { this.endMode(); ev.preventDefault(); return; }
    if (m && k === "enter" && m.pts) { this.finishTool(); ev.preventDefault(); return; }
    if (m && m.type === "roof" && is("cycle_roof")) {
      m.roofStyle = ROOF_STYLES[(ROOF_STYLES.indexOf(m.roofStyle) + 1) % ROOF_STYLES.length];
      this.settings.roofStyle = m.roofStyle; this.saveSettings();
      if (this.lastEv) this.toolMove(this.lastEv); ev.preventDefault(); return;
    }
    if (m && m.pts && k === "backspace") { m.pts.pop(); if (this.lastEv) this.toolMove(this.lastEv); ev.preventDefault(); return; }
    const act = Object.keys(km).find((a) => km[a] === k);
    if (!act) return;
    ev.preventDefault();
    this.run(act, ev);
  }
  run(act, ev = null) {
    const t = { tool_wall: "wall", tool_door: "door", tool_window: "window", tool_floor: "floor", tool_roof: "roof" }[act];
    if (t) return this.startTool(t);
    switch (act) {
      case "grab": case "rotate": case "scale": return this.startTransform(act, ev);
      case "delete": case "delete_alt": return this.deleteSelection();
      case "duplicate": return this.duplicate(ev);
      case "select_all": return this.setSelection([...this.objs.keys()].filter((id) => this.objs.get(id).visible));
      case "deselect_all": return this.setSelection([]);
      case "hide": return this.hide();
      case "unhide": return this.unhide();
      case "frame_selected": return this.frameSelected();
      case "frame_all": return this.bounds && this.frame(...this.bounds);
      case "undo": return this.undo();
      case "redo": case "redo_alt": return this.redo();
      case "fill_gap": return this.fillGap();
      case "properties": this.ui.props.hidden = !this.ui.props.hidden; return this.refreshUi();
      case "help": return this.toggleHelp();
      case "settings": return this.toggleSettings();
      default: return null;
    }
  }

  // ---------------------------------------------------------------- overlay UI
  buildUi() {
    const mk = (tag, cls, parent, html = "") => { const e = document.createElement(tag); if (cls) e.className = cls; if (html) e.innerHTML = html; parent.appendChild(e); return e; };
    const ui = this.ui = {};
    ui.shelf = mk("div", "ed3d-shelf", this.container);
    const tools = [["tool_wall", "Wall"], ["tool_door", "Door"], ["tool_window", "Window"], ["fill_gap", "Fill"], ["tool_floor", "Floor"], ["tool_roof", "Roof"],
      [null], ["grab", "Move"], ["rotate", "Rotate"], ["scale", "Scale"], ["delete", "Delete"], [null], ["undo", "Undo"], ["redo", "Redo"], [null],
      ["properties", "Props"], ["settings", "Keys"], ["help", "?"]];
    for (const [act, label] of tools) {
      if (!act) { mk("span", "ed3d-gap", ui.shelf); continue; }
      const b = mk("button", "ed3d-tool", ui.shelf);
      b.dataset.act = act; b.dataset.label = label;
      b.addEventListener("click", (ev) => { ev.preventDefault(); this.canvas.focus(); this.run(act, this.lastEv); });
    }
    ui.status = mk("div", "ed3d-status", this.container);
    ui.props = mk("div", "ed3d-panel ed3d-props", this.container);
    ui.props.hidden = false;
    ui.help = mk("div", "ed3d-panel ed3d-help", this.container); ui.help.hidden = true;
    ui.settings = mk("div", "ed3d-panel ed3d-settings", this.container); ui.settings.hidden = true;
    const bar = mk("div", "ed3d-io", this.container);
    for (const [label, fn] of [["Export GLB", () => this.exportGlb()], ["Save scene", () => this.saveScene()], ["Open scene", () => this.openScene()]]) {
      const b = mk("button", "ed3d-io-btn", bar, label); b.addEventListener("click", fn);
    }
    this.refreshUi();
  }
  say(t, live = false) { if (!this.ui) return; this.ui.status.textContent = t; this.ui.status.dataset.live = live ? "1" : ""; }
  label(act) { return (this.keymap[act] || "").toUpperCase(); }
  refreshUi() {
    if (!this.ui) return;
    for (const b of this.ui.shelf.querySelectorAll("button")) {
      const act = b.dataset.act, key = this.label(act);
      b.innerHTML = `${b.dataset.label}${key ? `<kbd>${key}</kbd>` : ""}`;
      b.title = `${ACTION_LABEL[act] || act}${key ? ` (${key})` : ""}`;
      const t = { tool_wall: "wall", tool_door: "door", tool_window: "window", tool_floor: "floor", tool_roof: "roof", grab: "grab", rotate: "rotate", scale: "scale" }[act];
      b.classList.toggle("on", !!(this.mode && t === this.mode.type));
    }
    const m = this.mode;
    if (!m) this.say(this.sel.size ? `${this.sel.size} selected — ${this.label("grab")} move · ${this.label("rotate")} rotate · ${this.label("scale")} scale · ${this.label("delete")} delete · ${this.label("fill_gap")} fill gap · ${this.label("properties")} properties`
      : `Click to select (Shift adds) · ${this.settings.nav === "classic" ? "left drag orbits, right drag pans" : "middle drag orbits, Shift+middle pans, Alt+left orbits"} · wheel zooms · ${this.label("help")} help`);
    else if (["grab", "rotate", "scale"].includes(m.type)) this.say(`${m.type}: move the mouse · ${this.label("axis_x")}/${this.label("axis_y")}/${this.label("axis_z")} constrain · type a value · click / Enter confirms · right-click / Esc cancels · Ctrl toggles snapping`, true);
    else if (m.type === "wall") this.say("Draw walls: click the corners (snaps to wall ends and the grid); Enter / right-click / double-click ends; Backspace removes the last point", true);
    else if (m.type === "door" || m.type === "window") this.say(`${m.type === "door" ? "Door" : "Window"}: point at a wall and click to cut it (sizes in ${this.label("settings")}); Esc ends`, true);
    else if (m.type === "floor") this.say("Floor: click the outline corners; Enter / double-click finishes", true);
    else if (m.type === "roof") this.say(`Roof (${m.roofStyle}): click the outline corners; Enter finishes; ${this.label("cycle_roof")} changes the style`, true);
    this.renderProps();
  }
  renderProps() {
    const p = this.ui.props;
    if (p.hidden) return;
    const keys = [...this.sel];
    if (keys.length !== 1) { p.innerHTML = `<b>Properties</b><div class="muted">${keys.length ? `${keys.length} elements selected` : "Nothing selected"}</div>`; return; }
    const k = keys[0], fields = [];
    let title = k, target = null;
    const num = (obj, key, label, step = 0.01) => fields.push({ obj, key, label, step });
    if (k.startsWith("op:")) {
      const { wall, op } = this.opening(k);
      if (!op) return;
      title = `${op.type} ${op.id}`; target = wall.id;
      fields.push({ special: "width", obj: op, label: "width (m)", wall });
      num(op, "s0", "from wall start (m)"); num(op, "z0", "sill / bottom (m)"); num(op, "z1", "head / top (m)");
      fields.push({ special: "type", obj: op, label: "type" });
    } else {
      const el = this.els.get(k);
      if (!el) return;
      title = `${el.type} ${el.id}`; target = el.id;
      if (el.kind === "wall") { fields.push({ special: "length", obj: el, label: "length (m)" }); num(el, "thickness", "thickness (m)"); num(el, "z0", "base (m)"); num(el, "height", "height (m)"); }
      else if (el.kind === "railing") { num(el, "thickness", "thickness (m)"); num(el, "z0", "base (m)"); num(el, "parapet", "parapet (m)"); num(el, "top", "rail top (m)"); }
      else if (el.kind === "prism") { num(el, "z0", "bottom (m)"); num(el, "z1", "top (m)"); }
      else if (el.kind === "roof") { fields.push({ special: "style", obj: el, label: "style" }); num(el, "pitch", "pitch (°)", 1); num(el, "z0", "eaves height (m)"); num(el, "thickness", "thickness (m)"); num(el, "overhang", "overhang (m)"); num(el, "dir", "slope direction (°, shed)", 5); }
      else fields.push({ note: "fixed shape: move / rotate / scale / delete it as a whole" });
    }
    p.innerHTML = `<b>${title}</b>`;
    for (const f of fields) {
      const row = document.createElement("label");
      row.className = "ed3d-row";
      if (f.note) { row.textContent = f.note; p.appendChild(row); continue; }
      row.innerHTML = `<span>${f.label}</span>`;
      let inp;
      if (f.special === "type" || f.special === "style") {
        inp = document.createElement("select");
        for (const o of f.special === "type" ? ["door", "window", "opening"] : ROOF_STYLES) inp.add(new Option(o, o));
        inp.value = f.special === "type" ? f.obj.type : f.obj.style;
      } else {
        inp = document.createElement("input");
        inp.type = "number"; inp.step = f.step || 0.01;
        const v = f.special === "length" ? hyp(f.obj.a, f.obj.b) : f.special === "width" ? f.obj.s1 - f.obj.s0 : f.obj[f.key];
        inp.value = (+v || 0).toFixed(f.step >= 1 ? 1 : 3);
      }
      inp.addEventListener("change", () => {
        this.begin();
        const v = parseFloat(inp.value);
        if (f.special === "type") f.obj.type = inp.value;
        else if (f.special === "style") f.obj.style = inp.value;
        else if (f.special === "length" && v > 0.01) { const fr = wallFrame(f.obj.a, f.obj.b); f.obj.b = [f.obj.a[0] + fr.u[0] * v, f.obj.a[1] + fr.u[1] * v]; }
        else if (f.special === "width" && v > 0.01) { const L = hyp(f.wall.a, f.wall.b); f.obj.s1 = Math.min(L, f.obj.s0 + v); }
        else if (f.special === "length" || f.special === "width" || Number.isNaN(v)) { this.pending = null; return; }
        else if (f.key === "s0") { const w = f.obj.s1 - f.obj.s0; f.obj.s0 = v; f.obj.s1 = v + w; }
        else f.obj[f.key] = v;
        this.buildObj(target);
        this.commit(`Changed ${f.label.replace(/ \(.*/, "")}`);
        this.canvas.focus();
      });
      row.appendChild(inp);
      p.appendChild(row);
    }
  }
  toggleHelp() {
    const h = this.ui.help;
    h.hidden = !h.hidden;
    if (h.hidden) return;
    h.innerHTML = `<b>3D editor</b> <span class="muted">(keys can be changed in ${this.label("settings")})</span><table>${Object.keys(this.keymap)
      .map((a) => `<tr><td><kbd>${this.label(a)}</kbd></td><td>${ACTION_LABEL[a] || a}</td></tr>`).join("")}</table>
      <p class="muted">In a move / rotate / scale: type a number for an exact value (metres, degrees, factor), X / Y / Z to constrain,
      Ctrl toggles snapping, click or Enter confirms, right-click or Esc cancels. A selected door or window slides along its wall;
      Z moves its sill. Drawing: the first click on a floor or roof surface sets the working height; Shift keeps 45° steps.</p>`;
  }
  toggleSettings() {
    const s = this.ui.settings;
    s.hidden = !s.hidden;
    if (s.hidden) return this.refreshUi();
    const nums = [["wallHeight", "wall height (m)"], ["wallThickness", "wall thickness (m)"], ["doorWidth", "door width (m)"], ["doorHeight", "door height (m)"],
      ["windowWidth", "window width (m)"], ["windowSill", "window sill (m)"], ["windowHead", "window head (m)"], ["slabThickness", "floor thickness (m)"],
      ["roofPitch", "roof pitch (°)"], ["roofThickness", "roof thickness (m)"], ["roofOverhang", "roof overhang (m)"], ["grid", "grid / snap step (m)"],
      ["angleSnap", "rotation snap (°)"], ["snapPx", "snap distance (px)"], ["workZ", "working height (m)"]];
    s.innerHTML = `<b>Settings</b><div class="ed3d-cols"><div><b class="small">Defaults</b>${nums.map(([k, l]) =>
      `<label class="ed3d-row"><span>${l}</span><input type="number" step="0.01" data-s="${k}" value="${this.settings[k]}"></label>`).join("")}
      <label class="ed3d-row"><span>roof style</span><select data-s="roofStyle">${ROOF_STYLES.map((r) => `<option ${r === this.settings.roofStyle ? "selected" : ""}>${r}</option>`).join("")}</select></label>
      <label class="ed3d-row"><span>navigation</span><select data-s="nav"><option value="blender" ${this.settings.nav === "blender" ? "selected" : ""}>Blender (middle orbits)</option><option value="classic" ${this.settings.nav === "classic" ? "selected" : ""}>Classic (left orbits)</option></select></label>
      <label class="ed3d-row"><span>snapping on</span><input type="checkbox" data-s="snap" ${this.settings.snap ? "checked" : ""}></label>
      <label class="ed3d-row"><span>door leaves</span><input type="checkbox" data-s="doorLeaves" ${this.settings.doorLeaves ? "checked" : ""}></label>
      </div><div><b class="small">Keys</b> <span class="muted small">click a key, then press the new one</span>${Object.keys(this.keymap).map((a) =>
      `<label class="ed3d-row"><span>${ACTION_LABEL[a] || a}</span><button class="ed3d-key" data-k="${a}">${this.label(a) || "—"}</button></label>`).join("")}</div></div>
      <div class="ed3d-row"><button data-do="export">Export settings</button><button data-do="import">Import settings</button><button data-do="reset">Reset all</button><button data-do="close">Close</button></div>`;
    for (const inp of s.querySelectorAll("[data-s]")) inp.addEventListener("change", () => {
      const k = inp.dataset.s;
      this.settings[k] = inp.type === "checkbox" ? inp.checked : inp.tagName === "SELECT" ? inp.value : parseFloat(inp.value);
      this.saveSettings(); this.applyNav();
      if (k === "doorLeaves") for (const [id, el] of this.els) if (el.kind === "wall") this.buildObj(id);
      if (k === "workZ" && this.grid) this.grid.position.z = this.settings.workZ - 0.002;
    });
    for (const b of s.querySelectorAll(".ed3d-key")) b.addEventListener("click", () => {
      b.textContent = "press a key…";
      const h = (ev) => {
        const k = keyOf(ev);
        if (!k) return;
        ev.preventDefault(); ev.stopPropagation();
        window.removeEventListener("keydown", h, true);
        if (k !== "escape") {
          const clash = Object.keys(this.keymap).find((a) => a !== b.dataset.k && this.keymap[a] === k && !a.startsWith("axis_") && !b.dataset.k.startsWith("axis_"));
          if (clash) this.keymap[clash] = "";
          this.keymap[b.dataset.k] = k; this.saveSettings();
        }
        s.hidden = true; this.toggleSettings();
      };
      window.addEventListener("keydown", h, true);
    });
    s.querySelector('[data-do="close"]').addEventListener("click", () => { s.hidden = true; this.canvas.focus(); this.refreshUi(); });
    s.querySelector('[data-do="reset"]').addEventListener("click", () => {
      this.settings = { ...DEFAULT_SETTINGS, workZ: this.settings.workZ }; this.keymap = { ...DEFAULT_KEYMAP }; this.saveSettings(); this.applyNav();
      s.hidden = true; this.toggleSettings();
    });
    s.querySelector('[data-do="export"]').addEventListener("click", () => this.download("editor-settings.json", JSON.stringify({ settings: this.settings, keymap: this.keymap }, null, 1), "application/json"));
    s.querySelector('[data-do="import"]').addEventListener("click", () => this.pickFile(".json", (txt) => {
      const d = JSON.parse(txt);
      Object.assign(this.settings, d.settings || {}); Object.assign(this.keymap, d.keymap || {}); this.saveSettings(); this.applyNav();
      s.hidden = true; this.toggleSettings();
    }));
  }

  // ---------------------------------------------------------------- files
  download(name, data, type) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([data], { type }));
    a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  pickFile(accept, cb) {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = accept;
    inp.addEventListener("change", async () => { if (inp.files[0]) { try { cb(await inp.files[0].text()); } catch (e) { alert("Could not read the file: " + e.message); } } });
    inp.click();
  }
  sceneJson() {
    const fixed = {};
    for (const [id, el] of this.els) if (el.kind === "mesh") fixed[id] = this.fixed.get(id);
    return { format: "cad2d3d-scene", version: 1, elements: [...this.els.values()], fixed, seq: this.seq };
  }
  saveScene() { this.download("scene.edit3d.json", JSON.stringify(this.sceneJson()), "application/json"); this.dirty = false; this.say("Scene saved"); }
  openScene() {
    this.pickFile(".json", (txt) => {
      const d = JSON.parse(txt);
      if (d.format !== "cad2d3d-scene") throw new Error("not a scene saved by this editor");
      this.begin();
      for (const [id, f] of Object.entries(d.fixed || {})) this.fixed.set(id, f);
      this.seq = Math.max(this.seq, d.seq || 0);
      this.restore(d.elements);
      this.commit(`Opened a scene (${d.elements.length} elements)`);
    });
  }
  exportGlb() {                                  // glTF 2.0 binary, y up, one node per element part
    const nodes = [], meshes = [], materials = [], accessors = [], views = [], chunks = [];
    let off = 0;
    const push = (arr) => { const b = new Uint8Array(arr.buffer); chunks.push(b); const pad = (4 - (b.length % 4)) % 4; if (pad) chunks.push(new Uint8Array(pad)); const o = off; off += b.length + pad; return o; };
    this.root.updateMatrixWorld(true);
    for (const [id, g] of this.objs) {
      if (!g.visible) continue;
      g.traverse((o) => {
        if (!o.isMesh || o.userData.role === "opening" || o.visible === false) return;
        const pos = o.geometry.getAttribute("position"), idx = o.geometry.getIndex();
        const v = new THREE.Vector3(), P = new Float32Array(pos.count * 3), lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (let k = 0; k < pos.count; k++) {
          v.fromBufferAttribute(pos, k).applyMatrix4(o.matrixWorld);
          P[k * 3] = v.x; P[k * 3 + 1] = v.y; P[k * 3 + 2] = v.z;
          for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], P[k * 3 + d]); hi[d] = Math.max(hi[d], P[k * 3 + d]); }
        }
        const I = new Uint32Array(idx ? idx.array : [...Array(pos.count).keys()]);
        views.push({ buffer: 0, byteOffset: push(P), byteLength: P.byteLength, target: 34962 });
        accessors.push({ bufferView: views.length - 1, componentType: 5126, count: pos.count, type: "VEC3", min: lo, max: hi });
        views.push({ buffer: 0, byteOffset: push(I), byteLength: I.byteLength, target: 34963 });
        accessors.push({ bufferView: views.length - 1, componentType: 5125, count: I.length, type: "SCALAR" });
        const c = o.material.color, a = o.material.opacity;
        materials.push({ name: `${id}_${o.userData.role}`, doubleSided: true, alphaMode: a < 1 ? "BLEND" : "OPAQUE",
          pbrMetallicRoughness: { baseColorFactor: [c.r, c.g, c.b, a], metallicFactor: 0, roughnessFactor: 0.9 } });
        meshes.push({ name: id, primitives: [{ attributes: { POSITION: accessors.length - 2 }, indices: accessors.length - 1, material: materials.length - 1 }] });
        nodes.push({ name: o.userData.role === "body" ? id : `${id}_${o.userData.role}`, mesh: meshes.length - 1 });
      });
    }
    const json = { asset: { version: "2.0", generator: "cad2d3d editor3d" }, scene: 0, scenes: [{ nodes: nodes.map((_, i) => i) }], nodes, meshes, materials,
      accessors, bufferViews: views, buffers: [{ byteLength: off }] };
    let js = new TextEncoder().encode(JSON.stringify(json));
    const jpad = (4 - (js.length % 4)) % 4;
    if (jpad) { const t = new Uint8Array(js.length + jpad).fill(0x20); t.set(js); js = t; }
    const total = 12 + 8 + js.length + 8 + off, out = new Uint8Array(total), dv = new DataView(out.buffer);
    dv.setUint32(0, 0x46546c67, true); dv.setUint32(4, 2, true); dv.setUint32(8, total, true);
    dv.setUint32(12, js.length, true); dv.setUint32(16, 0x4e4f534a, true); out.set(js, 20);
    let p = 20 + js.length;
    dv.setUint32(p, off, true); dv.setUint32(p + 4, 0x004e4942, true); p += 8;
    for (const c of chunks) { out.set(c, p); p += c.length; }
    this.download("scene.glb", out, "model/gltf-binary");
    this.say(`Exported ${nodes.length} meshes as GLB`);
    return out;
  }
}
