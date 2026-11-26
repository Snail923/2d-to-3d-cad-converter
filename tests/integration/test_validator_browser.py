"""Room validation by a local vision model (web/validator.js), with LM Studio replaced by a fake server in the
browser: what the model receives (guideline as system prompt, reference + result images, key, metadata, strict
answer schema), how its answer is checked (unknown tags refused), confidence-based auto-apply, the review list,
undo and the JSON export. Skipped when node / Playwright / Chromium are missing."""
import functools
import http.server
import json
import os
import shutil
import subprocess
import threading

import pytest

from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.reconstruction3d import build_3d
from architectural_converter.visualization.feature_overlay import view_payload

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
WEB = os.path.join(ROOT, "web")

PAGE = """<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="style.css">
<style>html,body{margin:0;height:100%} #view3d{position:relative;width:100%;height:100%}</style>
<script type="importmap">{"imports": {"three": "./vendor/three.module.min.js"}}</script></head>
<body><div id="view3d"></div><script type="module">
import { Editor3D } from "./editor3d.js";
import { Validator } from "./validator.js";
const ed = new Editor3D(document.getElementById("view3d"), () => {}); window.ed = ed;
const d = await (await fetch("data.json")).json(); window.D = d; ed.setData(d.mesh, d.model);
window.val = new Validator(ed, () => ({ model: d.model, view: d.view })); window.ready = true;
</script></body></html>"""

SCRIPT = r"""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
  const p = await b.newPage({ viewport: { width: 1400, height: 900 }, acceptDownloads: true });
  const errs = []; p.on("pageerror", (e) => errs.push(e.message));
  const seen = [];
  await p.route(/^http:\/\/(localhost|127\.0\.0\.1):1234\//, async (route) => {
    const u = route.request().url();
    if (u.endsWith("/api/v0/models")) return route.fulfill({ json: { data: [{ id: "vision-model", type: "vlm", state: "loaded" }] } });
    const body = JSON.parse(route.request().postData()), user = body.messages[1].content, text = user[0].text;
    seen.push({ hasFormat: !!body.response_format, schemaProps: body.response_format && body.response_format.json_schema.schema.properties,
      system: body.messages[0].content, text, images: user.filter((c) => c.type === "image_url").length,
      schema: body.response_format && body.response_format.json_schema.schema.properties.edits.items.properties.action.enum });
    const meta = JSON.parse(text.split("METADATA (the result as data, JSON):\n")[1].split("\n\nImage 1")[0]);
    const w = meta.walls.find((x) => !x.openings.length) || meta.walls[0], w2 = meta.walls.find((x) => x !== w) || w;
    const edits = [
      { action: "add_door", target: w.tag, position: 0.5, width: 0, value: "", confidence: 0.95, reason: "swing arc" },
      { action: "set_room_type", target: "R1", position: 0, width: 0, value: "bedroom", confidence: 0.7, reason: "bed" },
      { action: "delete_wall", target: w2.tag, position: 0, width: 0, value: "", confidence: 0.6, reason: "counter" },
      { action: "delete_wall", target: "W99", position: 0, width: 0, value: "", confidence: 0.9, reason: "made up" },
      { action: "comment", target: "R1", position: 0, width: 0, value: "", confidence: 0.4, reason: "note" } ];
    const ans = { room: "R1", analysis: "Access: no door in the reference either side.", summary: "ok", edits };
    if (!body.response_format)              // free mode: text first, JSON block at the end
      return route.fulfill({ json: { choices: [{ message: { content: "Looking at R1: the reference shows a swing arc.\n```json\n" + JSON.stringify(ans) + "\n```" } }] } });
    return route.fulfill({ json: { choices: [{ message: { content: JSON.stringify(ans) } }] } });
  });
  await p.goto(process.argv[2]); await p.waitForFunction(() => window.ready, null, { timeout: 60000 });
  const R = {};
  await p.click("canvas"); await p.keyboard.press("c");
  R.open = await p.evaluate(() => !val.panel.hidden);
  await p.click('.ed3d-validator [data-do="models"]'); await p.waitForTimeout(200);
  R.model = await p.evaluate(() => val.s.model);
  R.room = await p.evaluate(() => { const r = D.model.rooms.filter((x) => x.status !== "REJECTED" && !x.is_hypothesis).sort((a, b) => b.area - a.area)[3]; ed.setSelection([r.id], false); val.render(); return r.id; });
  await p.click('.ed3d-validator [data-do="runsel"]');
  await p.waitForFunction(() => !val.busy && val.results.length, null, { timeout: 60000 });
  if (!seen.length) { console.log(JSON.stringify({ debug: await p.evaluate(() => val.results.map((r) => r.error)), errs })); process.exit(3); }
  R.req = { guideline: seen[0].system.startsWith("# Room validation guideline"), images: seen[0].images, actions: seen[0].schema,
    tags: (seen[0].text.match(/^Tags: (.*)$/m) || [])[1] };
  R.first = await p.evaluate(() => val.results[0].edits.map((e) => [e.action, e.status]));
  R.undo1 = await p.evaluate(() => ed.undoStack.map((u) => u.label));
  await p.click('.ed3d-validator [data-do="applyall"]');
  R.after = await p.evaluate(() => val.results[0].edits.map((e) => e.status));
  R.type = await p.evaluate((id) => ed.els.get(id).roomType, R.room);
  const [dl] = await Promise.all([p.waitForEvent("download"), p.click('.ed3d-validator [data-do="export"]')]);
  R.export = JSON.parse(require("fs").readFileSync(await dl.path(), "utf8")).format;
  R.analysis1 = await p.evaluate(() => val.results[0].analysis);
  R.schemaFirst = Object.keys(seen[0].schemaProps || {});
  // free mode: no forced schema, the JSON block at the end of a text answer is used
  await p.uncheck('.ed3d-validator [data-s="strict"]');
  await p.evaluate(() => { val.results = []; val.render(); });
  await p.click('.ed3d-validator [data-do="runsel"]');
  await p.waitForFunction(() => !val.busy && val.results.length, null, { timeout: 60000 });
  R.free = await p.evaluate(() => ({ strict: val.s.strict, analysis: val.results[0].analysis, edits: val.results[0].edits.length, error: val.results[0].error || null }));
  R.freeSchema = !!seen[seen.length - 1].hasFormat;
  const [rq] = await Promise.all([p.waitForEvent("download"), p.click('.ed3d-validator [data-req]')]);
  const saved = JSON.parse(require("fs").readFileSync(await rq.path(), "utf8"));
  R.saved = { images: saved.request.messages[1].content.filter((c) => c.type === "image_url").length, answer: saved.answer.startsWith("Looking at R1") };
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
    d = tmp_path_factory.mktemp("val")
    for f in ("editor3d.js", "validator.js", "validation_guideline.md", "style.css"):
        shutil.copy(os.path.join(WEB, f), d)
    shutil.copytree(os.path.join(WEB, "vendor"), d / "vendor")
    (d / "index.html").write_text(PAGE)
    pipe = ExtractionPipeline()
    st = pipe.process(os.path.join(ROOT, "samples", "real", "apartment-building-plan.dxf"))
    (d / "data.json").write_text(json.dumps({"mesh": build_3d(st.model, pipe.cfg).to_json(), "model": st.model, "view": view_payload(st)}))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield d, f"http://127.0.0.1:{srv.server_address[1]}/index.html", node, npm_root
    srv.shutdown()


def test_room_check_with_a_fake_lm_studio(site):
    d, url, node, npm_root = site
    (d / "drive.js").write_text(SCRIPT)
    r = subprocess.run([node, str(d / "drive.js"), url], capture_output=True, text=True, timeout=240, env={**os.environ, "NODE_PATH": npm_root})
    if r.returncode != 0 and ("Executable doesn't exist" in r.stderr or "browserType.launch" in r.stderr):
        pytest.skip("Chromium for Playwright not installed")
    assert r.returncode == 0, r.stderr[-2000:]
    R = json.loads(r.stdout.strip().splitlines()[-1])
    assert R["open"] and R["model"] == "vision-model"
    assert R["req"]["guideline"] and R["req"]["images"] == 2 and "add_door" in R["req"]["actions"]
    assert R["req"]["tags"].startswith(f"R1 = {R['room']}, W1 = ")
    assert R["first"] == [["add_door", "applied"], ["set_room_type", "pending"], ["delete_wall", "pending"],
                          ["delete_wall", "invalid"], ["comment", "comment"]]          # only >= 0.85 applied at once
    assert len(R["undo1"]) == 1 and R["undo1"][0].startswith("AI (95%): add door")
    assert R["after"][:3] == ["applied", "applied", "applied"] and R["type"] == "bedroom"
    assert R["export"] == "cad2d3d-room-check" and not R["errors"]
    assert R["schemaFirst"][:2] == ["room", "analysis"] and R["analysis1"].startswith("Access")   # the model looks first
    assert R["free"] == {"strict": False, "analysis": "Access: no door in the reference either side.", "edits": 5, "error": None}
    assert R["freeSchema"] is False and R["saved"] == {"images": 2, "answer": True}
