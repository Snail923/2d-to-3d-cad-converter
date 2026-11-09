/* Runs the Python extraction engine (Pyodide) off the UI thread. */
const PYODIDE_VERSION = "0.28.3";
// optional override (?pyodide=<base url>) for self-hosted / offline Pyodide
const _override = new URLSearchParams(self.location.search).get("pyodide");
const INDEX_URL = _override || `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
importScripts(INDEX_URL + "pyodide.js");

let py = null;
let bridge = null;      // approach 1 (rules): architectural_converter.web_bridge
let bridge2 = null;     // approach 2 (SymPointV2): sympoint_converter.web_bridge, loaded on first use

function progress(msg) { self.postMessage({ type: "progress", msg }); }
// called from Python (architectural_converter.web_bridge._progress) while a drawing is processed
self.pyProgress = (msg) => progress(String(msg));

async function init(base) {
  progress("Loading the Python runtime (Pyodide " + PYODIDE_VERSION + ")…");
  py = await loadPyodide({ indexURL: INDEX_URL });
  progress("Loading numpy, shapely, sqlite3…");
  await py.loadPackage(["numpy", "shapely", "micropip", "sqlite3"]);
  const manifest = await (await fetch(base + "build_manifest.json", { cache: "no-cache" })).json();
  progress("Installing ezdxf, networkx, trimesh…");
  const urls = manifest.wheels.map((w) => new URL(base + "wheels/" + w, self.location.href).href);
  py.globals.set("_wheel_urls", py.toPy(urls));
  await py.runPythonAsync("import micropip\nawait micropip.install(list(_wheel_urls), deps=False)");
  progress("Loading the architectural extraction engine…");
  const zip = await (await fetch(base + manifest.package, { cache: "no-cache" })).arrayBuffer();
  py.unpackArchive(zip, "zip", { extractDir: "/home/pyodide/engine" });
  py.runPython("import sys\nsys.path.insert(0, '/home/pyodide/engine')");
  bridge = py.pyimport("architectural_converter.web_bridge");
  return { version: manifest.version, built: manifest.built, samples: manifest.samples,
           sympoint: !!manifest.sympoint };
}

function sp() {
  if (!bridge2) bridge2 = py.pyimport("sympoint_converter.web_bridge");
  return bridge2;
}

function writeInput(args) {
  const name = (args.name || "input.dxf").replace(/[^A-Za-z0-9._-]/g, "_");
  const path = "/tmp/" + name;
  if (args.bytes) py.FS.writeFile(path, new Uint8Array(args.bytes));
  return path;
}

function bytesOf(pyBytes) {
  const out = pyBytes.toJs();
  pyBytes.destroy();
  return out;
}

self.onmessage = async (ev) => {
  const { id, cmd, args } = ev.data;
  try {
    let result = null;
    const transfer = [];
    if (cmd === "init") {
      result = await init(args.base);
    } else if (cmd === "process") {
      const path = writeInput(args);
      result = bridge.process(path, args.corrections ? JSON.stringify(args.corrections) : null,
                              args.config ? JSON.stringify(args.config) : null);
    } else if (cmd === "sp_request") {
      result = sp().request(writeInput(args), args.config ? JSON.stringify(args.config) : null);
    } else if (cmd === "sp_process") {
      result = sp().process(writeInput(args), args.response, args.corrections ? JSON.stringify(args.corrections) : null,
                            args.config ? JSON.stringify(args.config) : null);
    } else if (cmd === "sp_rerun") {
      result = sp().rerun(JSON.stringify(args.corrections));
    } else if (cmd === "sp_remember") {
      sp().remember(args.method);
    } else if (cmd === "sp_activate") {
      result = sp().activate(args.method);
    } else if (cmd === "sp_reset") {
      sp().reset();
    } else if (cmd === "sp_compare") {
      result = sp().compare();
    } else if (cmd === "sp_evaluate_both") {
      result = sp().evaluate_both(args.reference);
    } else if (cmd === "rerun") {
      result = bridge.rerun(JSON.stringify(args.corrections));
    } else if (cmd === "export_json") {
      result = bridge.export_model_json();
    } else if (cmd === "export_sqlite") {
      result = bytesOf(bridge.export_sqlite_bytes());
      transfer.push(result.buffer);
    } else if (cmd === "export_glb") {
      result = bytesOf(bridge.export_glb_bytes());
      transfer.push(result.buffer);
    } else if (cmd === "export_report") {
      result = bridge.export_report_html();
    } else if (cmd === "evaluate") {
      result = bridge.evaluate(args.reference);
    } else {
      throw new Error("unknown command " + cmd);
    }
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (e) {
    self.postMessage({ id, ok: false, error: String(e && e.message ? e.message : e) });
  }
};
