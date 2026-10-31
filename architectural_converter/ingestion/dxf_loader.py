"""Load a DXF document robustly (ezdxf recover mode) and report audit problems."""
from __future__ import annotations

import os

import ezdxf
from ezdxf import recover

from .dwg_converter import convert_dwg_to_dxf


def load_document(path: str, dwg_converter: str | None = None):
    """Return (doc, issues, dxf_path). DWG input is converted with ODA first."""
    issues: list[dict] = []
    if path.lower().endswith(".dwg"):
        path = convert_dwg_to_dxf(path, converter=dwg_converter)
        issues.append({"stage": "ingestion", "severity": "info", "message": f"DWG converted to {path}"})
    try:
        doc, auditor = recover.readfile(path)
    except IOError as exc:
        raise ValueError(f"Not a DXF file or file not readable: {exc}") from exc
    except ezdxf.DXFStructureError as exc:
        raise ValueError(f"Invalid or corrupted DXF file: {exc}") from exc
    for err in getattr(auditor, "errors", [])[:200]:
        issues.append({"stage": "ingestion", "severity": "warning", "message": f"DXF audit error: {err.message}"})
    for fix in getattr(auditor, "fixes", [])[:200]:
        issues.append({"stage": "ingestion", "severity": "info", "message": f"DXF audit fix: {fix.message}"})
    return doc, issues, os.path.abspath(path)
