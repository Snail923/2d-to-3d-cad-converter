"""The web 3D editor driven like a user (mouse + keys) in headless Chromium: select, move with a typed
value and an axis, undo, cut a door, slide it along its wall, close it, draw a floor and a roof, edit a
number in the properties panel, export GLB. Skipped when node / Playwright / Chromium are missing."""
import functools
import http.server
import json
import os
import shutil
import subprocess
import threading

import pytest

from architectural_converter.benchmark.generate import DEFAULT_DIR
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.reconstruction3d import build_3d

WEB = os.path.join(os.path.dirname(__file__), "..", "..", "web")

PAGE = """<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="style.css">
<style>html,body{margin:0;height:100%} #view3d{position:relative;width:100%;height:100%}</style>
<script type="importmap">{"imports": {"three": "./vendor/three.module.min.js"}}</script></head>
<body><div id="view3d"></div><script type="module">
import { Editor3D } from "./editor3d.js";
const ed = new Editor3D(document.getElementById("view3d"), () => {}); window.ed = ed;
const d = await (await fetch("data.json")).json(); ed.setData(d.mesh, d.model); window.ready = true;
</script></body></html>"""

SCRIPT = r"""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
  const p = await b.newPage({ viewport: { width: 1300, height: 820 }, acceptDownloads: true });
  const errs = []; p.on("pageerror", (e) => errs.push(e.message));
  await p.goto(process.argv[2]); await p.waitForFunction(() => window.ready, null, { timeout: 30000 });
  const R = {};
  const t = await p.evaluate(() => {
    const r = ed.canvas.getBoundingClientRect();
    for (const w of [...ed.els.values()].filter((e) => e.kind === "wall" && !e.openings.length && Math.hypot(e.b[0]-e.a[0], e.b[1]-e.a[1]) > 3)) {
      const [x, y] = ed.screenOf((w.a[0]+w.b[0])/2, (w.a[1]+w.b[1])/2, w.z0 + 1.0);
      const h = ed.hit({ clientX: x + r.left, clientY: y + r.top });
      if (h && h.id === w.id) return { id: w.id, x: x + r.left, y: y + r.top, a: w.a };
    }
  });
  const click = async (x, y) => { await p.mouse.move(x, y); await p.mouse.down(); await p.mouse.up(); };
  const keys = async (...ks) => { for (const k of ks) await p.keyboard.press(k); };
  await click(t.x, t.y); R.sel = await p.evaluate(() => [...ed.sel]);
  await keys("g", "x", "1", "Enter"); R.moved = await p.evaluate((id) => ed.els.get(id).a, t.id);
  await keys("Control+z"); R.undone = await p.evaluate((id) => ed.els.get(id).a, t.id);
  await keys("d"); await p.mouse.move(t.x + 3, t.y); await click(t.x, t.y); await keys("Escape");
  R.door = await p.evaluate((id) => ed.els.get(id).openings.map((o) => [o.type, +(o.s1 - o.s0).toFixed(2), +o.s0.toFixed(2)]), t.id);
  await click(t.x, t.y); R.opSel = await p.evaluate(() => [...ed.sel]);
  await keys("g", "0", ".", "5", "Enter"); R.slid = await p.evaluate((id) => +ed.els.get(id).openings[0].s0.toFixed(2), t.id);
  await keys("f"); R.closed = await p.evaluate((id) => ed.els.get(id).openings.length, t.id);
  const pts = await p.evaluate(() => { const r = ed.canvas.getBoundingClientRect(), [lo] = ed.bounds;
    return [[1, 1], [4, 1], [4, 3.5], [1, 3.5]].map(([x, y]) => { const s = ed.screenOf(lo[0] + x, lo[1] + y, 0); return [s[0] + r.left, s[1] + r.top]; }); });
  await keys("p"); for (const [x, y] of pts) await click(x, y); await keys("Enter");
  await keys("o"); for (const [x, y] of pts) await click(x, y); await keys("Enter");
  R.made = await p.evaluate(() => [...ed.els.values()].filter((e) => e.id.startsWith("edit_")).map((e) => [e.kind, e.kind === "roof" ? e.z0 : e.z1]));
  await click(t.x, t.y - 20);
  await p.evaluate(() => { const i = [...ed.ui.props.querySelectorAll("label")].find((l) => l.textContent.startsWith("height")).querySelector("input"); i.value = "4.5"; i.dispatchEvent(new Event("change")); });
  R.height = await p.evaluate(() => ed.els.get([...ed.sel][0]).height);
  const [dl] = await Promise.all([p.waitForEvent("download"), p.evaluate(() => ed.exportGlb())]);
  R.glb = require("fs").readFileSync(await dl.path()).slice(0, 4).toString();
  R.errors = errs;
  console.log(JSON.stringify(R)); await b.close();
})();
"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    root = subprocess.run([node, "-e", "console.log(require.resolve('playwright'))"], capture_output=True, text=True,
                          env={**os.environ, "NODE_PATH": subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()})
    if root.returncode != 0:
        pytest.skip("playwright (node) not installed")
    d = tmp_path_factory.mktemp("ed3d")
    for f in ("editor3d.js", "style.css"):
        shutil.copy(os.path.join(WEB, f), d)
    shutil.copytree(os.path.join(WEB, "vendor"), d / "vendor")
    (d / "index.html").write_text(PAGE)
    pipe = ExtractionPipeline()
    st = pipe.process(os.path.join(DEFAULT_DIR, "01_residential_apartment.dxf"))
    (d / "data.json").write_text(json.dumps({"mesh": build_3d(st.model, pipe.cfg).to_json(), "model": st.model}))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield d, f"http://127.0.0.1:{srv.server_address[1]}/index.html", node
    srv.shutdown()


def test_editor_like_a_user(site):
    d, url, node = site
    (d / "drive.js").write_text(SCRIPT)
    env = {**os.environ, "NODE_PATH": subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()}
    r = subprocess.run([node, str(d / "drive.js"), url], capture_output=True, text=True, timeout=180, env=env)
    if r.returncode != 0 and ("Executable doesn't exist" in r.stderr or "browserType.launch" in r.stderr):
        pytest.skip("Chromium for Playwright not installed")
    assert r.returncode == 0, r.stderr[-2000:]
    R = json.loads(r.stdout.strip().splitlines()[-1])
    assert len(R["sel"]) == 1
    assert abs(R["moved"][0] - R["undone"][0] - 1.0) < 1e-6 and R["moved"][1] == R["undone"][1]   # G X 1: exactly 1 m in x
    assert R["door"] and R["door"][0][0] == "door" and abs(R["door"][0][1] - 0.9) < 1e-6
    assert R["opSel"][0].startswith("op:")                                                        # clicking the hole selects the door
    assert abs(R["slid"] - R["door"][0][2] - 0.5) < 1e-6                                            # typed 0.5: along its wall
    assert R["closed"] == 0                                                                        # F closes the opening
    kinds = dict((k, z) for k, z in R["made"])
    assert "prism" in kinds and "roof" in kinds and kinds["roof"] > 2.0                             # roof on the wall tops
    assert R["height"] == 4.5 and R["glb"] == "glTF" and not R["errors"]
