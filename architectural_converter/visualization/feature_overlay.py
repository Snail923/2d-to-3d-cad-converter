"""Overlay of extracted features (SVG) and the JSON view payload used by the web review UI."""
from __future__ import annotations

from shapely.geometry import LineString, Polygon

from ..geometry.geometric_utils import round_pt
from ..models.base_entity import Status
from .drawing_renderer import SvgCanvas, drawing_bounds, render_drawing

WALL_COLORS = {"exterior_wall": "#1d4f91", "interior_wall": "#2f7d32", "partition_wall": "#7cb342",
               None: "#555"}
STATUS_STYLE = {Status.ACCEPTED: 0.55, Status.NEEDS_REVIEW: 0.35, Status.REJECTED: 0.12}


def render_overlay(st, show_rejected=True, width_px=1400) -> str:
    b = st.building
    cv = SvgCanvas(drawing_bounds(st.drawing), width_px=width_px)
    for r in b.rooms.values():
        if r.status == Status.REJECTED:
            continue
        cv.polygon(r.polygon, fill="#ffe9a8", opacity=0.45, stroke="#e0b100", width=0.6,
                   title=f"{r.id} {r.name or ''} {r.classification} {r.area:.1f}m² conf {r.confidence:.2f}")
        c = r.polygon.representative_point()
        cv.text((c.x, c.y), f"{r.name or r.classification or ''} {r.area:.1f}m²", size=10, color="#8a6d00")
    render_drawing(st.drawing, cv, color="#c9ced6", width=0.6)
    for w in b.walls.values():
        if w.status == Status.REJECTED and not show_rejected:
            continue
        p = w.polygon(0.03)
        col = "#d32f2f" if w.status == Status.REJECTED else WALL_COLORS.get(w.classification, "#555")
        dash = "4,3" if w.status == Status.NEEDS_REVIEW else None
        title = f"{w.id} {w.classification} {w.status.value} conf {w.confidence:.2f} t={w.thickness}"
        if p is not None:
            cv.polygon(p, fill=col, opacity=STATUS_STYLE[w.status], stroke=col, width=0.8, title=title)
        cv.path(list(w.axis.coords), stroke=col, width=1.2, dash=dash, title=title)
    for h in b.hypotheses.values():
        cv.path(list(h.axis.coords), stroke="#ff00aa", width=2.5, dash="6,4", title=f"{h.id} hypothesis")
    for o in b.openings.values():
        col = {"door": "#ef6c00", "window": "#0288d1", "opening": "#8e24aa"}.get(o.type, "#000")
        if o.status == Status.REJECTED:
            col = "#bbb"
        cv.path([o.p0, o.p1], stroke=col, width=4, title=f"{o.id} {o.type} {o.subtype} {o.status.value} conf {o.confidence:.2f}")
    for c in b.columns.values():
        col = "#6a1b9a" if c.status != Status.REJECTED else "#ccc"
        cv.polygon(c.polygon, fill=col, opacity=0.6, stroke=col, width=1, title=f"{c.id} {c.status.value}")
    if st.report:
        for i in st.report.issues:
            if i.location and i.severity in ("warning", "error"):
                cv.circle(i.location, 4, fill="#ff1744" if i.severity == "error" else "#ff9100",
                          title=f"{i.code}: {i.description}")
    return cv.svg()


# ---------------------------------------------------------------------- web payload
def _pts(arr, nd=3):
    return [[round(float(x), nd), round(float(y), nd)] for x, y in arr]


def view_payload(st, max_entities: int = 60000) -> dict:
    """Original linework (normalised metres) with raw ids for highlighting in the UI."""
    d = st.drawing
    lines = []
    for g in d.entities:
        if not g.valid or g.points is None:
            continue
        if g.kind == "polyline" or g.kind == "mline":
            continue  # their line / arc children are drawn
        if g.kind in ("line", "arc", "curve"):
            lines.append({"r": g.raw_id, "l": g.layer, "k": g.role, "p": _pts(g.points)})
        elif g.kind in ("circle", "polygon"):
            pts = list(g.points) + [g.points[0]]
            lines.append({"r": g.raw_id, "l": g.layer, "k": g.role, "p": _pts(pts)})
        if len(lines) >= max_entities:
            break
    hatches = []
    for g in d.entities:
        if g.kind == "hatch" and g.valid:
            hatches.append({"r": g.raw_id, "l": g.layer, "loops": [_pts(lp) for lp in g.data.get("loops", [])][:20]})
    texts = []
    for g in d.entities:
        if g.kind == "text" and g.valid:
            texts.append({"r": g.raw_id, "l": g.layer, "t": str(g.data.get("text", ""))[:80],
                          "p": round_pt(g.points[0], 3), "h": round(float(g.data.get("height", 0.2) or 0.2), 3)})
    raw = {}
    for rid, r in d.raw_entities.items():
        raw[rid] = {"h": r.handle, "t": r.dxftype, "l": r.layer, "b": r.block_name}
    b = drawing_bounds(d)
    return {"bounds": [round(x, 3) for x in b], "lines": lines, "hatches": hatches, "texts": texts, "raw": raw,
            "layers": {k: v.to_dict() for k, v in st.ctx.layers.items()}}
