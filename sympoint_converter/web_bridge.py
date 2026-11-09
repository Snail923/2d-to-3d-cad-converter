"""Browser API of approach 2 (runs inside Pyodide next to approach 1's ``web_bridge``).

The browser cannot start the GPU service, so the page talks to it directly:

    request(path)        -> SymPointV2 request JSON for the drawing (tiles of primitives)
    (the page POSTs the tiles to the local SymPointV2 service and collects the answer)
    process(path, resp)  -> the same payload as approach 1's web_bridge.process, built by approach 2

Both approaches keep their own pipeline state; ``remember`` / ``activate`` switch which one approach
1's bridge functions (rerun, exports, evaluate) act on, so the page can show either result, and
``compare`` returns counts and agreement of the two. Approach 1's modules are only called, never
changed."""
from __future__ import annotations

import json

from architectural_converter import web_bridge as wb1

from . import load_config
from .classes import CLASSES, COLORS, GROUP_OF
from .pipeline import SymPointPipeline
from .sympoint_io import export_tiles, load_drawing

_SP: dict = {}
_STORE: dict = {}
_CUR: dict = {"method": None}


def _cfg(config_json: str | None):
    over = json.loads(config_json) if config_json else {}
    over["cache_labels"] = False                  # no persistent cache inside the browser
    return load_config(over)


def reset() -> None:
    _STORE.clear()
    _SP.clear()
    _CUR["method"] = None


def request(path: str, config_json: str | None = None) -> str:
    cfg = _cfg(config_json)
    wb1._progress("SymPointV2: reading the drawing and cutting it into tiles")
    req = export_tiles(load_drawing(path), cfg)
    _SP.update(path=path, cfg=cfg)
    return json.dumps(req, separators=(",", ":"))


def process(path: str, response_json: str, corrections_json: str | None = None, config_json: str | None = None) -> str:
    cfg = _cfg(config_json)               # must be the tiling the response was made for
    resp = json.loads(response_json)
    pipe = SymPointPipeline(cfg, response=resp, progress=wb1._progress, labels_progress=wb1._progress)
    st = pipe.process(path, corrections=json.loads(corrections_json) if corrections_json else None)
    wb1._S.update(pipe=pipe, st=st, cfg=pipe.cfg)
    remember("sympoint")
    return _with_labels(wb1._payload(st, pipe.cfg), st)


def rerun(corrections_json: str) -> str:
    """Approach 1's rerun on the active state, plus the label data when it is approach 2."""
    out = wb1.rerun(corrections_json)
    _STORE[_CUR["method"]] = dict(wb1._S)
    st = wb1._S["st"]
    return _with_labels(out, st) if getattr(st, "sp_labels", None) is not None else out


def _with_labels(payload: str, st) -> str:
    lab = st.sp_labels
    ents = {rid: [CLASSES.index(e["label"]), round(e["score"], 3)] for rid, e in lab["entities"].items()}
    log = {k: v for k, v in st.logs.get("detect", {}).items() if k != "approach"}
    from .viewer import summarize
    summ = summarize(st.sp_prims, lab)
    extra = {"entity_class": ents, "classes": CLASSES, "colors": [list(COLORS[c]) for c in CLASSES],
             "groups": [GROUP_OF.get(c, "other") for c in CLASSES], "summary": summ, "extract": log,
             "model_seconds": (st.sp_response or {}).get("seconds")}
    return payload[:-1] + ',"sympoint":' + json.dumps(extra, separators=(",", ":")) + "}"


def remember(method: str) -> None:
    """Store the pipeline state approach 1's bridge currently holds as ``method``'s result."""
    _STORE[method] = dict(wb1._S)
    _CUR["method"] = method


def activate(method: str) -> bool:
    """Make ``method``'s result the one rerun / exports / evaluate act on."""
    if _CUR["method"] and _CUR["method"] in _STORE:
        _STORE[_CUR["method"]] = dict(wb1._S)
    if method not in _STORE:
        return False
    wb1._S.clear()
    wb1._S.update(_STORE[method])
    _CUR["method"] = method
    return True


def compare() -> str:
    from architectural_converter.benchmark.metrics import evaluate
    from .compare import counts, model_as_reference
    if "rules" not in _STORE or "sympoint" not in _STORE:
        return json.dumps({"error": "run both methods first"})
    m1, m2 = _STORE["rules"]["st"].model, _STORE["sympoint"]["st"].model
    agree = evaluate(m2, model_as_reference(m1))
    out = {"rules": counts(m1), "sympoint": counts(m2),
           "timings": {"rules": sum(v for k, v in _STORE["rules"]["st"].timings.items() if ":" not in k),
                       "sympoint": sum(v for k, v in _STORE["sympoint"]["st"].timings.items() if ":" not in k)},
           "agreement": {"wall length of approach 2 also in approach 1": agree["wall_precision"],
                         "wall length of approach 1 found by approach 2": agree["wall_recall"],
                         "rooms matching": min(agree["room_precision"], agree["room_recall"]),
                         "doors matching": min(agree["door_precision"], agree["door_recall"]),
                         "windows matching": min(agree["window_precision"], agree["window_recall"])}}
    return json.dumps(out)


def evaluate_both(reference_json: str) -> str:
    """Benchmark scores of every stored result against a reference (benchmark samples)."""
    from architectural_converter.benchmark.metrics import evaluate
    ref = json.loads(reference_json)
    return json.dumps({k: evaluate(v["st"].model, ref) for k, v in _STORE.items()},
                      default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
