"""DWG -> DXF conversion through the ODA File Converter (external, optional).

ODA File Converter command line:
    ODAFileConverter <in_dir> <out_dir> <out_version> <out_type> <recurse> <audit> [filter]
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile

DEFAULT_PATHS = [
    "ODAFileConverter",
    "/usr/bin/ODAFileConverter",
    "/usr/local/bin/ODAFileConverter",
    "/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter",
    *glob.glob(r"C:\Program Files\ODA\ODAFileConverter*\ODAFileConverter.exe"),
]


class DWGConversionError(RuntimeError):
    pass


def find_oda_converter(explicit: str | None = None) -> str | None:
    candidates = [explicit] if explicit else []
    env = os.environ.get("ODA_FILE_CONVERTER")
    if env:
        candidates.append(env)
    candidates += DEFAULT_PATHS
    for c in candidates:
        if not c:
            continue
        found = shutil.which(c) or (c if os.path.isfile(c) else None)
        if found:
            return found
    return None


def convert_dwg_to_dxf(dwg_path: str, out_dir: str | None = None, converter: str | None = None,
                       version: str = "ACAD2018", audit: bool = True, timeout: int = 300,
                       runner=subprocess.run) -> str:
    """Convert one DWG file and return the path of the produced DXF."""
    exe = find_oda_converter(converter)
    if exe is None:
        raise DWGConversionError(
            "ODA File Converter not found. Install it (https://www.opendesign.com/guestfiles/oda_file_converter) "
            "and put it on PATH or set ODA_FILE_CONVERTER, or convert the DWG to DXF manually.")
    dwg_path = os.path.abspath(dwg_path)
    in_dir = tempfile.mkdtemp(prefix="dwg_in_")
    out_dir = out_dir or tempfile.mkdtemp(prefix="dwg_out_")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(dwg_path)
    shutil.copy2(dwg_path, os.path.join(in_dir, name))
    cmd = [exe, in_dir, out_dir, version, "DXF", "0", "1" if audit else "0", name]
    proc = runner(cmd, capture_output=True, text=True, timeout=timeout)
    dxf_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".dxf")
    if getattr(proc, "returncode", 1) != 0 or not os.path.isfile(dxf_path):
        raise DWGConversionError(f"ODA conversion failed ({getattr(proc, 'returncode', '?')}): "
                                 f"{getattr(proc, 'stderr', '')}")
    return dxf_path
