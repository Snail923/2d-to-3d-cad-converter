"""SymPointV2 inference service for approach 2 of the 2D -> 3D CAD Converter (runs on a CUDA machine).

Copy it into a SymPointV2 checkout (https://github.com/nicehuster/SymPointV2) and run it there with
the released weights and the config that came with them:

    python sympoint_server.py --config svg_pointT_release.yaml --checkpoint weights.pth [--port 8790]

API (JSON, CORS enabled):

    GET  /health   -> {"ok": true, "model": "SymPointV2", "device": "...", ...}
    POST /predict  <- {"tiles": [...]} made by sympoint_converter.sympoint_io.export_tiles
                   -> {"model": "SymPointV2", "tiles": [{"id", "semantic": [[class, score], ...],
                                                           "instances": [{"label", "score", "members"}]}]}

Offline use (no server):  python sympoint_server.py --config ... --checkpoint ... --in request.json --out resp.json

Each tile goes through the repository's OWN data loader (svgnet.data.svg3.SVGDataset.load) and the
same steps as its tools/inference.py, so the input is exactly what the model was trained on."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.getcwd())
# The released weights are a full training checkpoint (pickle); PyTorch >= 2.6 only loads plain
# tensors by default. The file is the one you downloaded yourself, so allow the classic loader.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch  # noqa: E402
import yaml  # noqa: E402
from munch import Munch  # noqa: E402


def _training_only_shims():
    """SymPointV2 imports detectron2 for two training helpers only (get_world_size in the loss,
    gradient clipping in the optimiser). Inference never calls them, so when detectron2 is not
    installed (it needs its own CUDA build) a minimal stand-in module is registered instead."""
    import types
    try:
        import detectron2  # noqa: F401
        return False
    except ImportError:
        pass
    root = types.ModuleType("detectron2")
    utils, comm = types.ModuleType("detectron2.utils"), types.ModuleType("detectron2.utils.comm")
    solver, build = types.ModuleType("detectron2.solver"), types.ModuleType("detectron2.solver.build")
    config = types.ModuleType("detectron2.config")
    comm.get_world_size = lambda: 1
    build.maybe_add_gradient_clipping = lambda cfg, optim: optim

    class CfgNode(dict):
        pass
    config.CfgNode = CfgNode
    root.utils, utils.comm, root.solver, solver.build, root.config = utils, comm, solver, build, config
    sys.modules.update({"detectron2": root, "detectron2.utils": utils, "detectron2.utils.comm": comm,
                        "detectron2.solver": solver, "detectron2.solver.build": build, "detectron2.config": config})
    if not hasattr(torch.cuda, "amp") or not hasattr(torch.cuda.amp, "autocast"):  # removed in future PyTorch
        import torch.cuda.amp as _amp
        _amp.autocast = lambda enabled=True, **kw: torch.autocast("cuda", enabled=enabled, **kw)
    return True


SHIMMED = _training_only_shims()

from svgnet.data.svg3 import SVGDataset  # noqa: E402
from svgnet.model.svgnet import SVGNet  # noqa: E402
from svgnet.util import get_root_logger, load_checkpoint  # noqa: E402


def _check_weights(checkpoint, model):
    """load_checkpoint is non-strict: a wrong config / checkpoint pair would load silently and give
    random labels. Count what really matches and stop if most of the model stays uninitialised."""
    sd = torch.load(checkpoint, map_location="cpu")
    src = sd.get("net", sd) if isinstance(sd, dict) else {}
    tgt = model.state_dict()
    ok = sum(1 for k, v in src.items() if k in tgt and tuple(v.shape) == tuple(tgt[k].shape))
    share = ok / max(len(tgt), 1)
    print(f"checkpoint: {ok}/{len(tgt)} model tensors loaded ({100 * share:.0f}%)")
    if share < 0.9:
        raise SystemExit("The checkpoint does not match this config (less than 90% of the model was loaded). "
                         "Use the .yaml that came with the weights.")
    return share


class Runner:
    def __init__(self, config, checkpoint):
        if not torch.cuda.is_available():
            raise SystemExit("PyTorch does not see a CUDA GPU (torch.cuda.is_available() is False).")
        cfg = Munch.fromDict(yaml.safe_load(open(config)))
        self.cfg = cfg
        self.model = SVGNet(cfg.model).cuda().eval()
        self.loaded = _check_weights(checkpoint, self.model)
        load_checkpoint(checkpoint, get_root_logger(), self.model)
        self.lock = threading.Lock()
        self.info = {"config": os.path.abspath(config), "checkpoint": os.path.abspath(checkpoint),
                     "torch": torch.__version__, "cuda": torch.version.cuda,
                     "device": torch.cuda.get_device_name(0),
                     "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                     "weights_loaded": round(self.loaded, 3), "detectron2_stub": SHIMMED}

    @torch.no_grad()
    def predict_tile(self, tile):
        sp = tile["sympoint"]
        n = len(sp["args"])
        if n == 0:
            return {"id": tile["id"], "semantic": [], "instances": []}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tile_s2.json")
            with open(path, "w") as fh:
                json.dump(sp, fh)
            coords, feats, labels, lengths, layer_ids = SVGDataset.load(path, idx=1)
        coords -= np.mean(coords, 0)
        batch = (torch.FloatTensor(coords), torch.FloatTensor(feats), torch.LongTensor(labels),
                 torch.IntTensor([coords.shape[0]]), torch.FloatTensor(lengths), torch.LongTensor(layer_ids))
        torch.cuda.empty_cache()
        with torch.autocast("cuda", enabled=bool(self.cfg.get("fp16", False))):
            res = self.model(batch, return_loss=False)
        sem = res["semantic_scores"].float().cpu().numpy()[:n]          # (n, 35)
        cls = sem.argmax(1)
        score = sem[np.arange(n), cls]
        insts = []
        for ins in res["instances"]:
            mem = np.nonzero(np.asarray(ins["masks"])[:n])[0]
            if len(mem):
                insts.append({"label": int(ins["labels"]), "score": round(float(ins["scores"]), 4),
                              "members": mem.tolist()})
        return {"id": tile["id"], "semantic": [[int(c), round(float(s), 4)] for c, s in zip(cls, score)],
                "instances": insts}

    def predict(self, request):
        t0 = time.time()
        with self.lock:
            tiles = [self.predict_tile(t) for t in request.get("tiles", [])]
        return {"model": "SymPointV2", "tiles": tiles, "seconds": round(time.time() - t0, 2)}


def serve(runner, host, port):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            out = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, fmt, *args):  # one short line per request
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

        def do_OPTIONS(self):
            self._send(204, {})

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "model": "SymPointV2", **runner.info})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/predict":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, runner.predict(req))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._send(500, {"error": "GPU out of memory: lower max_primitives_per_tile or tile_m "
                                          "in sympoint_converter/config.json"})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    print(f"SymPointV2 service on http://{host}:{port}  (GPU: {runner.info['device']}, "
          f"torch {runner.info['torch']}, CUDA {runner.info['cuda']})")
    ThreadingHTTPServer((host, port), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the .yaml that came with the weights")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--in", dest="inp", help="offline: request JSON")
    ap.add_argument("--out", help="offline: response JSON")
    a = ap.parse_args()
    r = Runner(a.config, a.checkpoint)
    if a.inp:
        res = r.predict(json.load(open(a.inp)))
        json.dump(res, open(a.out or "resp.json", "w"))
        print(f"{len(res['tiles'])} tile(s) in {res['seconds']} s -> {a.out or 'resp.json'}")
    else:
        serve(r, a.host, a.port)
