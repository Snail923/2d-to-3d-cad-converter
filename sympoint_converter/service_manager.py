"""Start / stop the local SymPointV2 service automatically.

The service runs inside its own conda env (it needs PyTorch + the compiled point ops) while the
converter runs in a normal Python. The manager:

* finds the SymPointV2 checkout (``SYMPOINT_HOME`` or config ``service.home``) and the python of
  its conda env (``<conda root>/envs/<env>/bin/python``; ``SYMPOINT_PYTHON`` / ``service.python``
  override it),
* copies the current ``service/server.py`` to ``<home>/sympoint_server.py`` when it changed,
* finds the weights and their config (``service.checkpoint`` / ``service.config`` candidates),
* starts the service in the background (log: ``~/.cache/sympoint_converter/server.log``,
  ``SYMPOINT_STATE_DIR`` moves it) and waits
  until ``/health`` answers; it keeps running for the next drawings until ``server stop``."""
from __future__ import annotations

import filecmp
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

from .client import ServiceError, health

def state_dir() -> str:
    return os.path.expanduser(os.environ.get("SYMPOINT_STATE_DIR", "~/.cache/sympoint_converter"))


def pid_file() -> str:
    return os.path.join(state_dir(), "server.pid")


def log_file() -> str:
    return os.path.join(state_dir(), "server.log")


SERVER_SRC = os.path.join(os.path.dirname(__file__), "service", "server.py")


class ServiceSetupError(RuntimeError):
    pass


def home_dir(scfg) -> str:
    return os.path.expanduser(os.environ.get("SYMPOINT_HOME") or scfg["home"])


def _conda_roots():
    roots = []
    for var in ("CONDA_EXE", "CONDA_PYTHON_EXE"):
        p = os.environ.get(var)
        if p:
            roots.append(os.path.dirname(os.path.dirname(p)))
    if os.environ.get("CONDA_PREFIX"):
        cp = os.environ["CONDA_PREFIX"]
        roots.append(os.path.dirname(os.path.dirname(cp)) if os.path.basename(os.path.dirname(cp)) == "envs" else cp)
    c = shutil.which("conda")
    if c:
        roots.append(os.path.dirname(os.path.dirname(os.path.realpath(c))))
    for name in ("miniforge3", "mambaforge", "miniconda3", "anaconda3", "micromamba"):
        roots.append(os.path.expanduser(f"~/{name}"))
    seen, out = set(), []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def find_python(scfg) -> str:
    explicit = os.environ.get("SYMPOINT_PYTHON") or scfg.get("python")
    if explicit:
        return os.path.expanduser(explicit)
    for root in _conda_roots():
        p = os.path.join(root, "envs", scfg["env"], "bin", "python")
        if os.path.exists(p):
            return p
    raise ServiceSetupError(f"conda env '{scfg['env']}' not found (looked in {', '.join(_conda_roots())}). "
                            "Run sympoint_converter/service/setup.sh, or set SYMPOINT_PYTHON to its python.")


def _first(home, cands, kind):
    for c in cands:
        p = os.path.join(home, c)
        if os.path.exists(p):
            return p
    ext = "*.pth" if kind == "checkpoint" else "*.yaml"
    found = sorted(glob.glob(os.path.join(home, ext)))
    if len(found) == 1:
        return found[0]
    raise ServiceSetupError(f"no {kind} found in {home} (tried {', '.join(cands)}). See sympoint_converter/README.md.")


def sync_server(home) -> bool:
    """Copy service/server.py into the SymPointV2 checkout when it changed (only into a real checkout)."""
    if not os.path.isdir(os.path.join(home, "svgnet")):
        return False
    dst = os.path.join(home, "sympoint_server.py")
    if os.path.exists(dst) and filecmp.cmp(SERVER_SRC, dst, shallow=False):
        return False
    shutil.copyfile(SERVER_SRC, dst)
    return True


def _pid():
    try:
        with open(pid_file()) as fh:
            pid = int(fh.read().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def status(url) -> dict:
    try:
        h = health(url)
        return {"running": True, "url": url, **h, "pid": _pid()}
    except ServiceError as e:
        return {"running": False, "url": url, "pid": _pid(), "error": str(e)}


def start(cfg, progress=print, wait=True) -> dict:
    scfg = cfg["service"]
    url = scfg["url"]
    st = status(url)
    if st["running"]:
        return st
    home = home_dir(scfg)
    if not os.path.isdir(home):
        raise ServiceSetupError(f"SymPointV2 checkout not found at {home}. Run sympoint_converter/service/setup.sh "
                                "or set SYMPOINT_HOME.")
    py = find_python(scfg)
    if scfg.get("sync_server", True) and sync_server(home):
        progress(f"updated {os.path.join(home, 'sympoint_server.py')}")
    server = os.path.join(home, "sympoint_server.py")
    if not os.path.exists(server):
        raise ServiceSetupError(f"{server} is missing (copy sympoint_converter/service/server.py there).")
    conf = _first(home, scfg["config"], "config")
    ckpt = _first(home, scfg["checkpoint"], "checkpoint")
    u = urlparse(url)
    cmd = [py, server, "--config", conf, "--checkpoint", ckpt, "--host", u.hostname or "127.0.0.1",
           "--port", str(u.port or 8790)]
    os.makedirs(state_dir(), exist_ok=True)
    log = open(log_file(), "ab")
    log.write(f"\n==== {time.ctime()}: {' '.join(cmd)}\n".encode())
    log.flush()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for k in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(k, None)                       # the converter's venv must not leak into the GPU env
    p = subprocess.Popen(cmd, cwd=home, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=env, start_new_session=True)
    with open(pid_file(), "w") as fh:
        fh.write(str(p.pid))
    progress(f"starting the SymPointV2 service (pid {p.pid}, log {log_file()}) ...")
    if not wait:
        return {"running": False, "pid": p.pid, "starting": True}
    t0, timeout = time.time(), float(scfg.get("start_timeout_s", 240))
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            raise ServiceSetupError(f"the SymPointV2 service stopped while starting (exit code {p.returncode}). "
                                    f"Last lines of {log_file()}:\n{tail(12)}")
        st = status(url)
        if st["running"]:
            progress(f"SymPointV2 service ready on {url} ({st.get('device', '?')}, {time.time() - t0:.0f} s)")
            return st
        time.sleep(1.0)
    raise ServiceSetupError(f"the SymPointV2 service did not answer within {timeout:.0f} s. Log: {log_file()}\n{tail(12)}")


def stop(cfg=None) -> bool:
    pid = _pid()
    if pid is None:
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(50):
        try:
            os.kill(pid, 0)
            time.sleep(0.1)
        except OSError:
            break
    try:
        os.remove(pid_file())
    except OSError:
        pass
    return True


def tail(n=30) -> str:
    try:
        with open(log_file(), "rb") as fh:
            return b"\n".join(fh.read().splitlines()[-n:]).decode(errors="replace")
    except OSError:
        return "(no log)"


def ensure_running(cfg, progress=print) -> str:
    """URL of a running service: the configured one if it answers, else start it (when allowed)."""
    url = cfg["service"]["url"]
    if status(url)["running"]:
        return url
    if not cfg["service"].get("autostart", True):
        raise ServiceError(f"The SymPointV2 service is not running at {url} (autostart is off): "
                           "python -m sympoint_converter server start")
    start(cfg, progress)
    return url


if __name__ == "__main__":  # pragma: no cover
    from . import load_config
    print(json.dumps(status(load_config()["service"]["url"]), indent=1))
    sys.exit(0)
