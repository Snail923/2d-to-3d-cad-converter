"""A page refresh does not lose the work: the session (drawing, results, corrections, 3D edits, room checks) is
saved in the browser (IndexedDB) and restored at once from the start screen, without the engine; the engine
catches up (process + re-run) the first time a command needs it. The real app (web/) runs in headless Chromium
with a stand-in engine (a worker answering with a payload made by the Python bridge), so no Pyodide download
is needed. Skipped when node / Playwright / Chromium are missing."""
import functools
import http.server
import json
import os
import shutil
import subprocess
import threading

import pytest

from architectural_converter import web_bridge
from architectural_converter.benchmark.generate import DEFAULT_DIR

WEB = os.path.join(os.path.dirname(__file__), "..", "..", "web")
SAMPLE = "01_residential_apartment.dxf"

FAKE_WORKER = """
const bc = new BroadcastChannel("fake-engine");
let payload = null;
self.onmessage = async (ev) => {
  const { id, cmd } = ev.data;
  bc.postMessage(cmd);
  try {
    let result = null;
    if (cmd === "init") result = { samples: [{ file: "%s", title: "Apartment" }], version: "test", built: "now", sympoint: false };
    else if (cmd === "process" || cmd === "rerun") { payload = payload || await (await fetch("fake_payload.json")).text(); result = payload; }
    else if (cmd.startsWith("sp_")) throw new Error("no SymPointV2 in this test");
    self.postMessage({ id, ok: true, result });
  } catch (e) { self.postMessage({ id, ok: false, error: String(e.message) }); }
};""" % SAMPLE

SCRIPT = r"""
const { chromium } = require('playwright'); const fs = require('fs');
(async () => {
  const b = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
  const ctx = await b.newContext({ viewport: { width: 1400, height: 900 }, acceptDownloads: true });
  const p = await ctx.newPage();
  const errs = []; p.on("pageerror", (e) => errs.push(e.message)); p.on("dialog", (d) => d.dismiss());
  await p.route("**/worker.js", (r) => r.fulfill({ contentType: "text/javascript", body: fs.readFileSync(process.argv[3], "utf8") }));
  const listen = () => p.evaluate(() => { window.__cmds = []; const c = new BroadcastChannel("fake-engine"); c.onmessage = (e) => window.__cmds.push(e.data); });
  const scene = async () => { const [dl] = await Promise.all([p.waitForEvent("download"), p.click(".ed3d-io-btn:nth-child(2)")]);
    const j = JSON.parse(fs.readFileSync(await dl.path(), "utf8")); return j.elements.reduce((n, e) => n + (e.openings || []).length, 0); };
  const R = {};
  await p.goto(process.argv[2]); await listen();
  await p.waitForFunction(() => /Ready/.test(document.querySelector("#status").textContent), null, { timeout: 60000 });
  R.bannerFirst = await p.evaluate(() => !!document.querySelector("#restoreSession"));
  await p.selectOption("#sample", { index: 1 }); await p.click("#loadSample");
  await p.waitForFunction(() => /approach 1/.test(document.querySelector("#status").textContent), null, { timeout: 60000 });
  await p.click('nav button[data-tab="tab3d"]'); await p.waitForTimeout(300);
  R.openings0 = await scene();
  const box = await (await p.$("#view3d canvas")).boundingBox();
  for (const [fx, fy] of [[0.5, 0.5], [0.45, 0.55], [0.55, 0.45], [0.4, 0.4], [0.6, 0.6], [0.35, 0.5], [0.65, 0.5]]) {
    await p.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
    if (/selected/.test(await p.evaluate(() => document.querySelector(".ed3d-status").textContent))) { await p.keyboard.press("x"); break; }
  }
  R.edited = await scene();                                  // also clears "unsaved": the session must still keep the edit
  await p.waitForTimeout(2500);                              // debounced save
  await p.reload(); await listen();
  await p.waitForFunction(() => document.querySelector("#restoreSession"), null, { timeout: 60000 });
  R.banner = await p.evaluate(() => document.querySelector("#empty2d").innerText);
  await p.evaluate(() => { window.__cmds = []; });
  await p.click("#restoreSession");
  await p.waitForFunction(() => /Restored/.test(document.querySelector("#status").textContent), null, { timeout: 60000 });
  R.cmdsRestore = await p.evaluate(() => window.__cmds.slice());
  R.shapes = await p.evaluate(() => document.querySelectorAll("#svg path, #svg polygon").length);
  await p.click('nav button[data-tab="tab3d"]'); await p.waitForTimeout(300);
  R.restored = await scene();
  await p.click('nav button[data-tab="tab2d"]');
  await p.evaluate(() => { window.__cmds = []; });
  await p.click("#rerun");
  await p.waitForFunction(() => /Re-run with/.test(document.querySelector("#status").textContent), null, { timeout: 60000 });
  R.cmdsRerun = await p.evaluate(() => window.__cmds.slice());
  await p.reload(); await listen();
  await p.waitForFunction(() => document.querySelector("#discardSession"), null, { timeout: 60000 });
  await p.click("#discardSession"); await p.reload();
  await p.waitForFunction(() => /Ready/.test(document.querySelector("#status").textContent), null, { timeout: 60000 });
  R.afterDiscard = await p.evaluate(() => !!document.querySelector("#restoreSession"));
  R.errors = errs;
  console.log(JSON.stringify(R)); await b.close();
})();
"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    npm_root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    if subprocess.run([node, "-e", "require.resolve('playwright')"], capture_output=True, env={**os.environ, "NODE_PATH": npm_root}).returncode:
        pytest.skip("playwright (node) not installed")
    d = tmp_path_factory.mktemp("site")
    shutil.copytree(WEB, d, dirs_exist_ok=True)
    os.makedirs(d / "samples", exist_ok=True)
    shutil.copy(os.path.join(DEFAULT_DIR, SAMPLE), d / "samples")
    (d / "fake_payload.json").write_text(web_bridge.process(os.path.join(DEFAULT_DIR, SAMPLE)))
    (d / "fake_worker.js").write_text(FAKE_WORKER)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield d, f"http://127.0.0.1:{srv.server_address[1]}/index.html", node, npm_root
    srv.shutdown()


def test_refresh_restores_the_session(site):
    d, url, node, npm_root = site
    (d / "drive.js").write_text(SCRIPT)
    r = subprocess.run([node, str(d / "drive.js"), url, str(d / "fake_worker.js")], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "NODE_PATH": npm_root})
    if r.returncode != 0 and ("Executable doesn't exist" in r.stderr or "browserType.launch" in r.stderr):
        pytest.skip("Chromium for Playwright not installed")
    assert r.returncode == 0, r.stderr[-2000:]
    R = json.loads(r.stdout.strip().splitlines()[-1])
    assert R["bannerFirst"] is False
    assert R["edited"] < R["openings0"]                              # an element deleted in 3D (with its openings)
    assert SAMPLE in R["banner"] and "edited 3D scene" in R["banner"]
    assert R["cmdsRestore"] == [] and R["shapes"] > 50               # restored without the engine: 2D drawn
    assert R["restored"] == R["edited"]                              # the 3D edit survived the refresh
    assert R["cmdsRerun"] == ["process", "rerun"]                    # the engine catches up on the first re-run
    assert R["afterDiscard"] is False and not R["errors"]
