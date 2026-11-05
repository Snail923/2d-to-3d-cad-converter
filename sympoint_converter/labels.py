"""SymPointV2 labels for a drawing: from the service (started automatically), an offline response
file, or the local cache (same DXF bytes + same tiling = same request, so the answer is reused)."""
from __future__ import annotations

import hashlib
import json
import os

from .client import ServiceError, predict
from .service_manager import ensure_running, start, state_dir, stop
from .sympoint_io import export_tiles, labels_from_response, primitives

CACHE_VERSION = 2
TILE_KEYS = ("tile_m", "overlap_m", "max_primitives_per_tile", "min_primitive_length_m", "include_hatch_boundaries")


def cache_path(path: str, cfg: dict) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    h.update(json.dumps({k: cfg[k] for k in TILE_KEYS}, sort_keys=True).encode())
    h.update(str(CACHE_VERSION).encode())
    return os.path.join(state_dir(), "labels", h.hexdigest() + ".json")


def _predict(url, req, cfg, progress, managed):
    try:
        return predict(url, req, cfg["request_tiles_per_call"], cfg["timeout_s"], progress)
    except ServiceError as e:
        # after a CUDA error the service stops itself: restart it once (only the one we manage)
        if not managed or not cfg["service"].get("autostart", True) or \
                ("stops now" not in str(e) and "Cannot reach" not in str(e)):
            raise
        progress(f"{e}\nrestarting the SymPointV2 service once ...")
        stop(cfg)
        start(cfg, progress)
        return predict(url, req, cfg["request_tiles_per_call"], cfg["timeout_s"], progress)


def get_labels(drawing, path: str, cfg: dict, url: str | None = None, response: dict | None = None,
               progress=print, use_cache: bool | None = None):
    """(labels, primitives, request, response, source). ``url`` None = configured service (autostart)."""
    prims = primitives(drawing, cfg)
    req = export_tiles(drawing, cfg, prims)
    use_cache = cfg.get("cache_labels", True) if use_cache is None else use_cache
    cp = cache_path(path, cfg) if path and os.path.exists(path) else None
    source = "response file"
    if response is None and use_cache and cp and os.path.exists(cp):
        with open(cp, encoding="utf-8") as fh:
            response = json.load(fh)
        source = "cache"
        progress(f"SymPointV2 labels from cache ({cp})")
    if response is None:
        managed = url is None
        if managed:
            url = ensure_running(cfg, progress)
        progress(f"{len(prims):,} primitives in {len(req['tiles'])} tile(s) of {cfg['tile_m']:g} m -> SymPointV2")
        response = _predict(url, req, cfg, progress, managed)
        source = "service"
        if cp:
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(response, fh)
    labels = labels_from_response(req, response, cfg["min_label_score"], cfg["min_label_mass"])
    return labels, prims, req, response, source
