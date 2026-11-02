"""The web app's scripts must parse as ES modules (a top-level `return`, for example, only fails there)."""
import os
import shutil
import subprocess
import tempfile

import pytest

WEB = os.path.join(os.path.dirname(__file__), "..", "..", "web")


@pytest.mark.parametrize("name,module", [("app.js", True), ("verifier.js", True), ("viewer3d.js", True),
                                         ("worker.js", False)])
def test_web_script_parses(name, module):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    d = tempfile.mkdtemp()
    dst = os.path.join(d, name.replace(".js", ".mjs" if module else ".cjs"))
    shutil.copy(os.path.join(WEB, name), dst)
    r = subprocess.run([node, "--check", dst], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
