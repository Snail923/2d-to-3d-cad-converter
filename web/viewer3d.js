import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

/* Renders the Phase 2 mesh JSON (model z-up) with three.js (y-up). */
export class Viewer3D {
  constructor(container, onPick) {
    this.container = container;
    this.onPick = onPick;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0xeef1f6);
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 2000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8c8478, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(30, 60, 20);
    this.scene.add(sun);
    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2;  // model z-up -> three y-up
    this.scene.add(this.root);
    this.objects = new Map();
    this.visible = { floor: true, review: true, window: true };
    this.ray = new THREE.Raycaster();
    this.renderer.domElement.addEventListener("click", (ev) => this.pick(ev));
    new ResizeObserver(() => this.resize()).observe(container);
    const loop = () => { requestAnimationFrame(loop); if (this.container.offsetParent !== null) { this.controls.update(); this.renderer.render(this.scene, this.camera); } };
    loop();
  }

  resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setData(mesh, model) {
    for (const o of this.objects.values()) { o.geometry.dispose(); o.material.dispose(); this.root.remove(o); }
    this.objects.clear();
    const statusOf = new Map(model.walls.map((w) => [w.id, w.status]));
    for (const m of mesh.meshes) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(m.positions, 3));
      g.setIndex(m.indices);
      g.computeVertexNormals();
      const [r, gr, b, a] = m.color;
      const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(r, gr, b), transparent: a < 1, opacity: a,
        roughness: 0.85, metalness: 0.0, side: THREE.DoubleSide, flatShading: true, depthWrite: a >= 1 });
      const obj = new THREE.Mesh(g, mat);
      obj.userData = { id: m.id, type: m.type, review: statusOf.get(m.id) === "NEEDS_REVIEW", base: mat.color.clone() };
      this.root.add(obj);
      this.objects.set(m.id + ":" + m.type, obj);
    }
    this.applyVisibility();
    const [lo, hi] = mesh.metadata.bounds || [[0, 0, 0], [10, 10, 3]];
    const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2;
    const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], 3);
    this.controls.target.set(cx, 0, -cy);
    this.camera.position.set(cx + size * 0.55, size * 0.85, -cy + size * 0.95);
    this.camera.near = size / 500; this.camera.far = size * 20;
    this.camera.updateProjectionMatrix();
    this.resize();
  }

  setVisible(kind, on) { this.visible[kind] = on; this.applyVisibility(); }

  applyVisibility() {
    for (const o of this.objects.values()) {
      const u = o.userData;
      o.visible = !(u.type === "floor" && !this.visible.floor) && !(u.review && !this.visible.review) &&
        !(u.type === "window" && !this.visible.window);
    }
  }

  highlight(id) {
    for (const o of this.objects.values()) {
      o.material.color.copy(o.userData.base);
      if (o.userData.id === id) o.material.color.setRGB(1.0, 0.25, 0.45);
    }
  }

  pick(ev) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(v, this.camera);
    const hits = this.ray.intersectObjects([...this.objects.values()].filter((o) => o.visible), false);
    if (hits.length && this.onPick) this.onPick(hits[0].object.userData.id);
  }
}
