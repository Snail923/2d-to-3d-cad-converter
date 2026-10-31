"""SVG rendering of the normalised drawing and of extraction overlays (no extra deps)."""
from __future__ import annotations

import html
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, Polygon

ROLE_COLORS = {"linework": "#9aa3ad", "door": "#d08c2a", "window": "#2a8fd0", "column": "#7d4bc2",
               "furniture": "#b5b5b5", "symbol": "#c4c4c4"}


class SvgCanvas:
    def __init__(self, bounds, width_px=1400, pad=1.0):
        minx, miny, maxx, maxy = bounds
        self.minx, self.miny = minx - pad, miny - pad
        self.w = (maxx - minx) + 2 * pad
        self.h = (maxy - miny) + 2 * pad
        self.scale = width_px / max(self.w, 1e-9)
        self.width_px = width_px
        self.height_px = int(self.h * self.scale)
        self.items: list[str] = []

    def xy(self, p):
        return ((p[0] - self.minx) * self.scale, (self.miny + self.h - p[1]) * self.scale)

    def path(self, pts, closed=False, stroke="#000", width=1.0, fill="none", opacity=1.0, cls="", title=None, dash=None):
        if len(pts) < 2:
            return
        d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in (self.xy(p) for p in pts))
        if closed:
            d += " Z"
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        t = f"<title>{html.escape(title)}</title>" if title else ""
        self.items.append(f'<path class="{cls}" d="{d}" stroke="{stroke}" stroke-width="{width}" fill="{fill}" '
                          f'fill-opacity="{opacity}" stroke-linejoin="round"{extra}>{t}</path>')

    def polygon(self, poly: Polygon, **kw):
        if poly is None or poly.is_empty:
            return
        polys = [poly] if isinstance(poly, Polygon) else list(getattr(poly, "geoms", []))
        for p in polys:
            if not isinstance(p, Polygon):
                continue
            rings = [p.exterior, *p.interiors]
            d = ""
            for r in rings:
                d += "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in (self.xy(c) for c in r.coords)) + " Z "
            t = f"<title>{html.escape(kw.get('title') or '')}</title>" if kw.get("title") else ""
            self.items.append(f'<path class="{kw.get("cls", "")}" d="{d}" fill-rule="evenodd" stroke="{kw.get("stroke", "#000")}" '
                              f'stroke-width="{kw.get("width", 1)}" fill="{kw.get("fill", "none")}" '
                              f'fill-opacity="{kw.get("opacity", 1)}">{t}</path>')

    def circle(self, p, r_px, fill="#f00", stroke="none", title=None):
        x, y = self.xy(p)
        t = f"<title>{html.escape(title)}</title>" if title else ""
        self.items.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r_px}" fill="{fill}" stroke="{stroke}">{t}</circle>')

    def text(self, p, s, size=11, color="#333"):
        x, y = self.xy(p)
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
                          f'font-family="sans-serif" text-anchor="middle">{html.escape(s)}</text>')

    def svg(self, background="#ffffff") -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width_px}" height="{self.height_px}" '
                f'viewBox="0 0 {self.width_px} {self.height_px}"><rect width="100%" height="100%" fill="{background}"/>'
                + "".join(self.items) + "</svg>")


def drawing_bounds(drawing):
    pts = [g.points for g in drawing.entities if g.points is not None and g.kind not in ("text", "dimension")]
    if not pts:
        return (0, 0, 1, 1)
    P = np.vstack(pts)
    lo, hi = np.percentile(P, 0.5, axis=0), np.percentile(P, 99.5, axis=0)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def render_drawing(drawing, canvas: SvgCanvas | None = None, color=None, width=0.8):
    cv = canvas or SvgCanvas(drawing_bounds(drawing))
    for g in drawing.entities:
        if not g.valid:
            continue
        col = color or ROLE_COLORS.get(g.role, "#888")
        if g.kind in ("line", "arc", "curve") and g.parent is None or (g.kind in ("line", "arc") and g.parent):
            if g.kind == "polyline":
                continue
            cv.path([tuple(p) for p in g.points], stroke=col, width=width)
        elif g.kind == "circle" or (g.kind == "polygon"):
            cv.path([tuple(p) for p in g.points], closed=True, stroke=col, width=width)
        elif g.kind == "hatch":
            for lp in g.data.get("loops", []):
                cv.path([tuple(p) for p in lp], closed=True, stroke="none", fill="#c9b79c", opacity=0.35, width=0)
        elif g.kind == "text":
            cv.text(tuple(g.points[0]), str(g.data.get("text", "")).splitlines()[0][:30], size=9, color="#666")
    return cv
