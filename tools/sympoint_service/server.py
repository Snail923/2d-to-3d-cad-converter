"""SymPointV2 inference service for the 2D -> 3D CAD Converter (runs on a CUDA machine).

Run it from inside a SymPointV2 checkout (https://github.com/nicehuster/SymPointV2) with the
released weights:

    python server.py --config configs/svg/svg_pointT.yaml --checkpoint <weights.pth> [--port 8790]

API (JSON, CORS enabled so the web app on GitHub Pages / localhost can call it):

    GET  /health   -> {"ok": true, "model": "SymPointV2", "device": "...", "classes": [...]}
    POST /predict  <- request made by architectural_converter.learned.export_tiles
                   -> {"model": "SymPointV2", "tiles": [{"id", "semantic": [[class, score], ...],
                                                           "instances": [{"label", "score", "members"}]}]}

Offline use (no server):  python server.py --config ... --checkpoint ... --in request.json --out labels.json

Each tile goes through the repository's OWN data loader (svgnet.data.svg3.SVGDataset.load) and the
same steps as tools/inference.py, so the input is exactly what the model was trained on."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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

from svgnet.data.svg3 import SVGDataset  # noqa: E402
from svgnet.model.svgnet import SVGNet  # noqa: E402
from svgnet.util import get_root_logger, load_checkpoint  # noqa: E402


class Runner:
    def __init__(self, config, checkpoint):
        cfg = Munch.fromDict(yaml.safe_load(open(config)))
        self.cfg = cfg
        self.model = SVGNet(cfg.model).cuda().eval()
        load_checkpoint(checkpoint, get_root_logger(), self.model)
        self.lock = __import__("threading").Lock()

    @torch.no_grad()
    def predict_tile(self, tile):
        sp = tile["sympoint"]
        n = len(sp["args"])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tile_s2.json")
            with open(path, "w") as fh:
                json.dump(sp, fh)
            coords, feats, labels, lengths, layer_ids = SVGDataset.load(path, idx=1)
        coords -= np.mean(coords, 0)
        batch = (torch.FloatTensor(coords), torch.FloatTensor(feats), torch.LongTensor(labels),
                 torch.IntTensor([coords.shape[0]]), torch.FloatTensor(lengths), torch.LongTensor(layer_ids))
        torch.cuda.empty_cache()
        with torch.cuda.amp.autocast(enabled=bool(self.cfg.get("fp16", False))):
            res = self.model(batch, return_loss=False)
        sem = res["semantic_scores"].float().cpu().numpy()[:n]          # (n, 35)
        cls = sem.argmax(1)
        score = sem[np.arange(n), cls]
        insts = []
        for ins in res["instances"]:
            mask = np.asarray(ins["masks"])[:n]
            mem = np.nonzero(mask)[0]
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

        def do_OPTIONS(self):
            self._send(204, {})

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "model": "SymPointV2", "device": torch.cuda.get_device_name(0)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/predict":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, runner.predict(req))
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    print(f"SymPointV2 service on http://{host}:{port}  (GPU: {torch.cuda.get_device_name(0)})")
    ThreadingHTTPServer((host, port), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out")
    a = ap.parse_args()
    r = Runner(a.config, a.checkpoint)
    if a.inp:
        res = r.predict(json.load(open(a.inp)))
        json.dump(res, open(a.out or "labels.json", "w"))
        print(f"{len(res['tiles'])} tile(s) in {res['seconds']} s -> {a.out or 'labels.json'}")
    else:
        serve(r, a.host, a.port)
