"""HTTP client for the SymPointV2 service (``service/server.py``)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class ServiceError(RuntimeError):
    pass


def _call(url: str, data: dict | None = None, timeout: float = 30.0) -> dict:
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.load(fh)
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e).get("error", str(e))
        except Exception:  # noqa: BLE001
            msg = str(e)
        raise ServiceError(f"SymPointV2 service error ({e.code}): {msg}") from None
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise ServiceError(f"Cannot reach the SymPointV2 service at {url}: {e}. "
                           "Is sympoint_server.py running (see sympoint_converter/README.md)?") from None


def health(base_url: str) -> dict:
    return _call(base_url.rstrip("/") + "/health", timeout=10)


def predict(base_url: str, request: dict, tiles_per_call: int = 8, timeout: float = 1800, progress=print) -> dict:
    """Send the tiles in small batches (progress + bounded request size); merge the answers."""
    tiles = request.get("tiles", [])
    out, t0 = [], time.time()
    for i in range(0, len(tiles), max(1, tiles_per_call)):
        part = tiles[i:i + tiles_per_call]
        res = _call(base_url.rstrip("/") + "/predict", {"tiles": part}, timeout=timeout)
        out.extend(res.get("tiles", []))
        if progress:
            progress(f"  SymPointV2: {min(i + len(part), len(tiles))}/{len(tiles)} tiles "
                     f"({time.time() - t0:.1f} s)")
    return {"model": "SymPointV2", "tiles": out, "seconds": round(time.time() - t0, 2)}
