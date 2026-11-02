"""Synthetic multi-storey permit set (several floor plans side by side on sheets).

Mirrors a typical permit-drawing model space: every floor is drawn on its own sheet with the same
site context (plot limit, dashed setback line, contour lines, road), a level marker
("708.75 F.F.L"), room tags carrying the level ("Lobby GF", "Entrance F1-2"), a stair core that is
identical on every floor, plus an elevation sheet (several level markers) and a duplicate
structural plan of the ground floor - views that must NOT become floors."""
from __future__ import annotations

import math

import ezdxf
import numpy as np

from .dxf_writer import Style, _Writer
from .plan_spec import PlanSpec
from .plans import apartment

LEVELS = [("B1", "BASEMENT 1", 705.40), ("GF", "GROUND FLOOR", 708.75), ("F1", "FIRST FLOOR", 712.10),
          ("F2", "SECOND FLOOR", 715.45)]
SHEET_W, SHEET_H = 70.0, 45.0
BUILDING_AT = (20.0, 12.0)          # building origin inside each sheet (metres)


def _floor_spec(tag, kind):
    s = PlanSpec(f"floor_{tag}", tag)
    x0, y0 = BUILDING_AT
    if kind == "parking":
        s.chain([(x0, y0), (x0 + 24, y0), (x0 + 24, y0 + 9), (x0, y0 + 9)], 0.3, closed=True)
        for k, x in enumerate((6, 12, 18)):
            s.columns.append((x0 + x, y0 + 4.5, 0.4, 0.4, "rect"))
        s.label(f"{tag} PARKING", (x0 + 6, y0 + 2))
    else:
        apartment(s, x0, y0, prefix="", furniture=False)
        apartment(s, x0 + 12, y0, mirror=True, prefix="", exterior=False, entrance=False, furniture=False)
        s.chain([(x0 + 12, y0), (x0 + 24, y0), (x0 + 24, y0 + 9), (x0 + 12, y0 + 9)], 0.3)
        s.label(f"Lobby {tag}", (x0 + 12.5, y0 + 4.25))
        s.label(f"Entrance {tag}-1", (x0 + 8.5, y0 + 4.25))
        s.label(f"Entrance {tag}-2", (x0 + 15.5, y0 + 4.25))
    # stair core: identical on every floor (walls + treads), north of the building
    cx, cy = x0 + 10.5, y0 + 9
    s.chain([(cx, cy), (cx, cy + 4.5), (cx + 3, cy + 4.5), (cx + 3, cy)], 0.2)
    return s


def _site(msp, ox, oy, k):
    W = lambda x, y: ((ox + x) * 1000, (oy + y) * 1000)  # noqa: E731  (mm)
    msp.add_lwpolyline([W(0, 0), W(SHEET_W, 0), W(SHEET_W, SHEET_H), W(0, SHEET_H)], close=True, dxfattribs={"layer": "SHEET"})
    msp.add_lwpolyline([W(12, 5), W(54, 5), W(56, 33), W(10, 34)], close=True, dxfattribs={"layer": "PLOT"})
    sb = msp.add_lwpolyline([W(15, 8), W(51, 8), W(52, 30), W(14, 31)], close=True, dxfattribs={"layer": "SETBACK"})
    sb.dxf.linetype = "DASHED"
    for j in range(3):
        pts = [W(2 + 6 * i, 2 + 6 * j + 1.5 * math.sin(i / 2)) for i in range(11)]
        msp.add_spline(pts, dxfattribs={"layer": "CONTOURS"})
    msp.add_line(W(0, 40), W(SHEET_W - 12, 41), dxfattribs={"layer": "ROAD"})


def _stairs(msp, ox, oy):
    x0, y0 = BUILDING_AT
    cx, cy = x0 + 10.5, y0 + 9
    for i in range(12):
        y = cy + 0.4 + i * 0.3
        msp.add_line(((ox + cx + 0.15) * 1000, (oy + y) * 1000), ((ox + cx + 1.4) * 1000, (oy + y) * 1000),
                     dxfattribs={"layer": "A-FLOR-STRS"})
    msp.add_text("UP", dxfattribs={"height": 200, "insert": ((ox + cx + 0.6) * 1000, (oy + cy + 2) * 1000)})


def build_permit_set():
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    ref = {"levels": []}
    for k, (tag, title, ffl) in enumerate(LEVELS):
        ox, oy = k * (SHEET_W + 10), 0.0
        spec = _floor_spec(tag, "parking" if tag.startswith("B") else "apartments")
        w = _Writer(spec, Style(units="mm", layers="aia", offset=(ox, oy), door_repr="block", window_repr="block", dims=False))
        w.doc, w.msp = doc, msp
        for name in set(w.L.values()):
            if name not in doc.layers:
                doc.layers.add(name)
        w.walls(); w.openings(); w.columns(); w.labels()
        _site(msp, ox, oy, k)
        _stairs(msp, ox, oy)
        x0, y0 = BUILDING_AT
        msp.add_text(f"{ffl:.2f} F.F.L", dxfattribs={"height": 250, "insert": ((ox + x0 + 4) * 1000, (oy + y0 + 6) * 1000)})
        msp.add_text(f"{title} PLAN", dxfattribs={"height": 400, "insert": ((ox + 30) * 1000, (oy + 2.5) * 1000)})
        ref["levels"].append({"tag": tag, "elevation": ffl, "sheet_offset": [ox, oy]})
    # distractor 1: duplicate structural plan of the ground floor
    ox = len(LEVELS) * (SHEET_W + 10)
    spec = _floor_spec("GF", "apartments")
    w = _Writer(spec, Style(units="mm", layers="aia", offset=(ox, 0.0), dims=False))
    w.doc, w.msp = doc, msp
    w.walls(); w.labels()
    _site(msp, ox, 0, 99)
    msp.add_text("708.75 F.F.L", dxfattribs={"height": 250, "insert": ((ox + 24) * 1000, 18000)})
    msp.add_text("STRUCTURAL PLAN", dxfattribs={"height": 400, "insert": ((ox + 30) * 1000, 2500)})
    # distractor 2: elevation view (stacked slabs with several level markers)
    ox = (len(LEVELS) + 1) * (SHEET_W + 10)
    for i, (tag, title, ffl) in enumerate(LEVELS):
        y = 8 + i * 3.35
        for dy in (0.0, 0.2):
            msp.add_line(((ox + 15) * 1000, (y + dy) * 1000), ((ox + 39) * 1000, (y + dy) * 1000))
        msp.add_text(f"{tag} {ffl:.2f} F.F.L", dxfattribs={"height": 250, "insert": ((ox + 41) * 1000, y * 1000)})
    for x in (15, 39):
        for dx in (0.0, 0.3):
            msp.add_line(((ox + x + dx) * 1000, 8000), ((ox + x + dx) * 1000, 21000))
    msp.add_text("ELEVATION 1", dxfattribs={"height": 400, "insert": ((ox + 25) * 1000, 4000)})
    return doc, ref
