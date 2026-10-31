"""Render a ``PlanSpec`` as a DXF drawing in a chosen CAD convention.

The same plan can be drawn as double lines, closed polylines, hatches, centre lines,
MLINEs or wide polylines; doors and windows as blocks or loose linework; in any unit,
with any layer naming scheme, fragmentation, duplicates and nested blocks. The
reference geometry comes from the spec, so it is exact by construction."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import ezdxf
import numpy as np
from shapely.geometry import Point, Polygon

from ..geometry.geometric_utils import polygons_of
from .plan_spec import PlanSpec

UNIT_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}
INSUNITS = {"mm": 4, "cm": 5, "m": 6, "in": 1, "ft": 2}

LAYER_SCHEMES = {
    "aia": {"wall": "A-WALL", "door": "A-DOOR", "window": "A-GLAZ", "column": "S-COLS", "furniture": "I-FURN",
            "text": "A-AREA-IDEN", "dim": "A-ANNO-DIMS", "hatch": "A-WALL-PATT", "grid": "S-GRID", "floor": "A-FLOR-PATT"},
    "french": {"wall": "MURS", "door": "PORTES", "window": "FENETRES", "column": "POTEAUX", "furniture": "MOBILIER",
               "text": "TEXTES", "dim": "COTATION", "hatch": "MURS-HACHURES", "grid": "AXES", "floor": "SOL"},
    "generic": {"wall": "WALLS", "door": "DOORS", "window": "WINDOWS", "column": "COLUMNS", "furniture": "FURNITURE",
                "text": "TEXT", "dim": "DIMENSIONS", "hatch": "HATCH", "grid": "GRID", "floor": "FLOOR"},
    "poor": {"wall": "0", "door": "0", "window": "0", "column": "0", "furniture": "0", "text": "0", "dim": "0",
             "hatch": "0", "grid": "0", "floor": "0"},
    "cryptic": {"wall": "L01", "door": "L07", "window": "L08", "column": "L03", "furniture": "L12", "text": "L20",
                "dim": "L21", "hatch": "L02", "grid": "L30", "floor": "L40"},
}


@dataclass
class Style:
    units: str = "mm"
    header_units: str | None = "auto"     # "auto" = matching code, None = unitless ($INSUNITS 0)
    offset: tuple[float, float] = (0.0, 0.0)
    layers: str = "aia"
    wall_repr: str = "lines"               # lines | polylines | centerline | mline | wide_polyline | hatch_only
    wall_hatch: bool = False
    door_repr: str = "block"               # block | loose
    window_repr: str = "block"             # block | loose
    window_lines: int = 3
    column_repr: str = "polyline_hatch"    # polyline_hatch | block | lines | circle_solid
    furniture_repr: str = "polyline"       # polyline | block
    fragment: float = 0.0
    duplicates: float = 0.0
    jitter: float = 0.0
    labels: str = "text"                   # text | mtext_area | none
    dims: bool = True
    floor_hatch: bool = False
    assemblies: list = field(default_factory=list)
    seed: int = 7


class _Writer:
    def __init__(self, spec: PlanSpec, style: Style):
        self.spec = spec
        self.st = style
        self.k = 1.0 / UNIT_TO_M[style.units]
        self.rng = random.Random(style.seed)
        self.doc = ezdxf.new("R2018", setup=True)
        if style.header_units is None:
            self.doc.header["$INSUNITS"] = 0
        else:
            self.doc.header["$INSUNITS"] = INSUNITS[style.units if style.header_units == "auto" else style.header_units]
        self.msp = self.doc.modelspace()
        self.L = LAYER_SCHEMES[style.layers]
        for name in set(self.L.values()):
            if name not in self.doc.layers:
                self.doc.layers.add(name)
        self._blocks: dict[str, object] = {}

    # ------------------------------------------------------------- coordinates
    def W(self, p):
        return ((p[0] + self.st.offset[0]) * self.k, (p[1] + self.st.offset[1]) * self.k)

    def lay(self, role):
        return self.L[role]

    # target container for an item located at world point p (assembly support)
    def target(self, p):
        for a in self.st.assemblies:
            if a["polygon"].contains(Point(p)):
                return a
        return None

    def _local(self, a, pw):
        """world-metres point -> assembly local (block) coordinates (drawing units)."""
        x, y = pw[0] - a["insert"][0], pw[1] - a["insert"][1]
        th = -math.radians(a["rotation"])
        xr = x * math.cos(th) - y * math.sin(th)
        yr = x * math.sin(th) + y * math.cos(th)
        if a.get("mirror"):
            xr = -xr
        return (xr * self.k, yr * self.k)

    def space_and_xy(self, pw):
        a = self.target(pw)
        if a is None:
            return self.msp, self.W(pw), None
        return a["block"], self._local(a, pw), a

    # ------------------------------------------------------------- primitives
    def line(self, a, b, role="wall"):
        sp, pa, asm = self.space_and_xy(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        if asm is not None:
            pieces = [(self._local(asm, a), self._local(asm, b))]
        else:
            pieces = [(self.W(a), self.W(b))]
        if self.st.fragment > 0 and role == "wall":
            pieces = self._fragment(pieces[0])
        for p, q in pieces:
            sp.add_line(p, q, dxfattribs={"layer": self.lay(role)})
            if self.st.duplicates > 0 and role == "wall" and self.rng.random() < self.st.duplicates:
                if self.rng.random() < 0.6:
                    sp.add_line(p, q, dxfattribs={"layer": self.lay(role)})
                else:  # partial overlapping duplicate
                    t0, t1 = sorted((self.rng.uniform(0, 0.5), self.rng.uniform(0.5, 1.0)))
                    sp.add_line(_lerp(p, q, t0), _lerp(p, q, t1), dxfattribs={"layer": self.lay(role)})

    def _fragment(self, seg):
        p, q = np.array(seg[0]), np.array(seg[1])
        L = np.hypot(*(q - p))
        if self.rng.random() > self.st.fragment or L * UNIT_TO_M[self.st.units] < 0.6:
            return [self._jit(p, q)]
        n = self.rng.choice([2, 2, 3, 4])
        cuts = sorted(self.rng.uniform(0.15, 0.85) for _ in range(n - 1))
        u = (q - p) / L
        pts = [p] + [p + (q - p) * c for c in cuts] + [q]
        out = []
        for a, b in zip(pts[:-1], pts[1:]):
            gap = self.rng.uniform(-0.003, 0.003) * self.k   # +-3 mm gap/overlap
            out.append(self._jit(a, b + u * gap * 0.5))
        return out

    def _jit(self, a, b):
        if self.st.jitter <= 0:
            return (tuple(a), tuple(b))
        j = self.st.jitter * self.k
        return (tuple(np.array(a) + [self.rng.uniform(-j, j), self.rng.uniform(-j, j)]),
                tuple(np.array(b) + [self.rng.uniform(-j, j), self.rng.uniform(-j, j)]))

    def arc(self, c, r, a0, a1, role="wall"):
        sp, pc, asm = self.space_and_xy(c)
        if asm is not None:
            rot = asm["rotation"]
            if asm.get("mirror"):
                a0, a1 = 180 - (a1 - rot), 180 - (a0 - rot)
            else:
                a0, a1 = a0 - rot, a1 - rot
        sp.add_arc(pc, r * self.k, a0, a1, dxfattribs={"layer": self.lay(role)})

    # ------------------------------------------------------------- walls
    def arc_circles(self):
        out = []
        for w in self.spec.walls:
            if w.arc:
                cx, cy, r, _, _ = w.arc
                out.append((cx, cy, r - w.t / 2))
                out.append((cx, cy, r + w.t / 2))
        return out

    def ring_edges(self, ring):
        pts = list(ring.coords)[:-1]
        circles = self.arc_circles()
        n = len(pts)

        def on(p):
            for k, (cx, cy, R) in enumerate(circles):
                if abs(math.hypot(p[0] - cx, p[1] - cy) - R) < 1e-6:
                    return k
            return None

        tags = [on(p) for p in pts]
        edges = []
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            ta, tb = tags[i], tags[(i + 1) % n]
            if ta is not None and ta == tb:
                cx, cy, R = circles[ta]
                da = math.atan2(a[1] - cy, a[0] - cx)
                db = math.atan2(b[1] - cy, b[0] - cx)
                step = abs((db - da + math.pi) % (2 * math.pi) - math.pi)
                if step < math.radians(5):
                    edges.append(("arc", ta, a, b))
                    continue
            edges.append(("line", None, a, b))
        # group consecutive arc edges on the same circle; merge collinear lines
        out = []
        for e in edges:
            if out and e[0] == "arc" and out[-1][0] == "arc" and out[-1][1] == e[1] and out[-1][3] == e[2]:
                out[-1] = ("arc", e[1], out[-1][2], e[3])
            elif out and e[0] == "line" and out[-1][0] == "line" and _collinear(out[-1][2], out[-1][3], e[3]):
                out[-1] = ("line", None, out[-1][2], e[3])
            else:
                out.append(e)
        if len(out) > 1 and out[0][0] == out[-1][0]:
            f, l = out[0], out[-1]
            if f[0] == "line" and _collinear(l[2], l[3], f[3]):
                out[0] = ("line", None, l[2], f[3])
                out.pop()
            elif f[0] == "arc" and f[1] == l[1]:
                out[0] = ("arc", f[1], l[2], f[3])
                out.pop()
        return out, circles

    def emit_edges(self, polys, role="wall"):
        for poly in polys:
            for ring in [poly.exterior, *poly.interiors]:
                edges, circles = self.ring_edges(ring)
                for kind, ci, a, b in edges:
                    if kind == "line":
                        self.line(a, b, role)
                    else:
                        cx, cy, R = circles[ci]
                        da = math.degrees(math.atan2(a[1] - cy, a[0] - cx))
                        db = math.degrees(math.atan2(b[1] - cy, b[0] - cx))
                        # ring orientation decides direction; draw CCW between the two
                        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                        cr = (a[0] - cx) * (b[1] - cy) - (a[1] - cy) * (b[0] - cx)
                        if cr >= 0:
                            self.arc((cx, cy), R, da, db, role)
                        else:
                            self.arc((cx, cy), R, db, da, role)

    def walls(self):
        sp = self.spec
        style = self.st.wall_repr
        solid = sp.solid(with_openings=True)
        polys = polygons_of(solid)
        if style in ("lines",):
            self.emit_edges(polys)
        elif style == "polylines":
            for poly in polys:
                for ring in [poly.exterior, *poly.interiors]:
                    pts = [self.W(p) for p in list(ring.coords)[:-1]]
                    self.msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": self.lay("wall")})
        elif style in ("centerline", "mline", "wide_polyline"):
            for k, w in enumerate(sp.walls):
                cuts = sorted(o.interval() for o in sp.openings if o.wall == k)
                if w.arc:
                    cx, cy, r, a0, a1 = w.arc
                    if style == "centerline":
                        self.arc((cx, cy), r, a0, a1)
                    else:
                        pts = list(w.axis().coords)
                        self._centre_piece(pts, w.t, style)
                    continue
                s = 0.0
                pieces = []
                for c0, c1 in cuts:
                    pieces.append((s, c0))
                    s = c1
                pieces.append((s, w.length))
                for a, b in pieces:
                    if b - a <= 1e-6:
                        continue
                    pa = tuple(np.array(w.p0) + w.u * a)
                    pb = tuple(np.array(w.p0) + w.u * b)
                    self._centre_piece([pa, pb], w.t, style)
        if self.st.wall_hatch or style == "hatch_only":
            for poly in polys:
                h = self.msp.add_hatch(color=8, dxfattribs={"layer": self.lay("hatch")})
                h.set_pattern_fill("ANSI31", scale=0.02 * self.k) if self.st.units != "m" else h.set_pattern_fill("ANSI31", scale=0.02)
                h.paths.add_polyline_path([self.W(p) for p in list(poly.exterior.coords)[:-1]], is_closed=True, flags=1)
                for r in poly.interiors:
                    h.paths.add_polyline_path([self.W(p) for p in list(r.coords)[:-1]], is_closed=True, flags=0)

    def _centre_piece(self, pts, t, style):
        if style == "centerline":
            for a, b in zip(pts[:-1], pts[1:]):
                self.line(a, b)
        elif style == "wide_polyline":
            pl = self.msp.add_lwpolyline([self.W(p) for p in pts], dxfattribs={"layer": self.lay("wall")})
            pl.dxf.const_width = t * self.k
        elif style == "mline":
            name = "WALLSTYLE"
            if name not in self.doc.mline_styles:
                ms = self.doc.mline_styles.new(name)
                ms.elements.append(0.5, 7, "BYLAYER")
                ms.elements.append(-0.5, 7, "BYLAYER")
            ml = self.msp.add_mline([self.W(p) for p in pts], dxfattribs={"layer": self.lay("wall"), "style_name": name})
            ml.dxf.scale_factor = t * self.k
            ml.dxf.justification = 1
            ml.update_geometry()

    # ------------------------------------------------------------- blocks
    def block(self, name, builder):
        if name not in self._blocks:
            b = self.doc.blocks.new(name)
            builder(b)
            self._blocks[name] = b
        return name

    def insert(self, name, p_world, xdir, ydir, sx, sy, layer_role):
        """Insert block with local x->xdir (unit) scaled sx and local y->ydir scaled sy."""
        sp, pt, asm = self.space_and_xy(p_world)
        xd = np.array(xdir, float)
        yd = np.array(ydir, float)
        if asm is not None:
            th = -math.radians(asm["rotation"])
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            xd, yd = R @ xd, R @ yd
            if asm.get("mirror"):
                xd[0] = -xd[0]
                yd[0] = -yd[0]
        rot = math.degrees(math.atan2(xd[1], xd[0]))
        handed = xd[0] * yd[1] - xd[1] * yd[0]
        sp.add_blockref(name, pt, dxfattribs={"layer": self.lay(layer_role), "rotation": rot,
                                               "xscale": sx * self.k, "yscale": sy * self.k * (1 if handed > 0 else -1)})

    # ------------------------------------------------------------- openings
    def openings(self):
        for o in self.spec.openings:
            w = self.spec.walls[o.wall]
            u, n = w.u, w.n
            s0, s1 = o.interval()
            base = np.array(w.p0)
            face = n * (w.t / 2) * o.swing
            if o.kind == "door":
                if o.hinge == "start":
                    hinge, xdir = base + u * s0 + face, u
                else:
                    hinge, xdir = base + u * s1 + face, -u
                ydir = n * o.swing
                self._door(hinge, xdir, ydir, o.width)
            elif o.kind == "double_door":
                half = o.width / 2
                ydir = n * o.swing
                self._door(base + u * s0 + face, u, ydir, half)
                self._door(base + u * s1 + face, -u, ydir, half)
            elif o.kind == "sliding":
                def b_sl(b):
                    for y0, y1, x0, x1 in ((-0.35, -0.05, 0.0, 0.55), (0.05, 0.35, 0.45, 1.0)):
                        b.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], close=True)
                self.block("SLIDING_DOOR", b_sl)
                self.insert("SLIDING_DOOR", base + u * s0, u, n, o.width, w.t, "door")
            elif o.kind == "window":
                ys = [-0.5, 0.0, 0.5] if self.st.window_lines == 3 else [-0.5, -1 / 6, 1 / 6, 0.5]
                if self.st.window_repr == "block":
                    nm = f"WINDOW_{self.st.window_lines}L"

                    def b_win(b, ys=ys):
                        for y in ys:
                            b.add_line((0, y), (1, y))
                        b.add_line((0, -0.5), (0, 0.5))
                        b.add_line((1, -0.5), (1, 0.5))
                    self.block(nm, b_win)
                    self.insert(nm, base + u * s0, u, n, o.width, w.t, "window")
                else:
                    for y in ys:
                        a = base + u * s0 + n * (y * w.t)
                        b = base + u * s1 + n * (y * w.t)
                        self.line(tuple(a), tuple(b), "window")

    def _door(self, hinge, xdir, ydir, width):
        if self.st.door_repr == "block":
            def b_door(b):
                b.add_line((0, 0), (0, 1))
                b.add_arc((0, 0), 1, 0, 90)
            self.block("DOOR_SWING", b_door)
            self.insert("DOOR_SWING", hinge, xdir, ydir, width, width, "door")
        else:
            tip = np.array(hinge) + np.array(ydir) * width
            self.line(tuple(hinge), tuple(tip), "door")
            a_closed = math.degrees(math.atan2(xdir[1], xdir[0]))
            a_open = math.degrees(math.atan2(ydir[1], ydir[0]))
            cr = xdir[0] * ydir[1] - xdir[1] * ydir[0]
            if cr > 0:
                self.arc(tuple(hinge), width, a_closed, a_open, "door")
            else:
                self.arc(tuple(hinge), width, a_open, a_closed, "door")

    # ------------------------------------------------------------- others
    def columns(self):
        for x, y, w, h, shape in self.spec.columns:
            if shape == "circle" or self.st.column_repr == "circle_solid":
                c = self.W((x, y))
                self.msp.add_circle(c, w / 2 * self.k, dxfattribs={"layer": self.lay("column")})
                hh = self.msp.add_hatch(dxfattribs={"layer": self.lay("column")})
                hh.paths.add_edge_path().add_arc(c, w / 2 * self.k, 0, 360)
                continue
            pts = [(x - w / 2, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)]
            if self.st.column_repr == "block":
                def b_col(b):
                    b.add_lwpolyline([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], close=True)
                    hh = b.add_hatch()
                    hh.paths.add_polyline_path([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], is_closed=True)
                self.block("COLUMN", b_col)
                self.insert("COLUMN", (x, y), (1, 0), (0, 1), w, h, "column")
            elif self.st.column_repr == "lines":
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    self.msp.add_line(self.W(a), self.W(b), dxfattribs={"layer": self.lay("column")})
            else:
                self.msp.add_lwpolyline([self.W(p) for p in pts], close=True, dxfattribs={"layer": self.lay("column")})
                hh = self.msp.add_hatch(dxfattribs={"layer": self.lay("column")})
                hh.paths.add_polyline_path([self.W(p) for p in pts], is_closed=True)

    def furniture(self):
        for x, y, w, h, ang in self.spec.furniture:
            c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            loc = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
            pts = [(x + px * c - py * s, y + px * s + py * c) for px, py in loc]
            if self.st.furniture_repr == "block":
                nm = f"FURN_{int(w * 100)}x{int(h * 100)}"

                def b_f(b, w=w, h=h):
                    b.add_lwpolyline([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)], close=True)
                    b.add_line((-w / 2 + 0.05, h / 2 - 0.3), (w / 2 - 0.05, h / 2 - 0.3))
                self.block(nm, b_f)
                sp, pt, _ = self.msp, self.W((x, y)), None
                sp.add_blockref(nm, pt, dxfattribs={"layer": self.lay("furniture"), "rotation": ang,
                                                    "xscale": self.k, "yscale": self.k})
            else:
                self.msp.add_lwpolyline([self.W(p) for p in pts], close=True, dxfattribs={"layer": self.lay("furniture")})

    def labels(self):
        if self.st.labels == "none":
            return
        ref = {r["name"]: r for r in self.spec.reference()["rooms"] if r["name"]}
        for txt, pt in self.spec.labels:
            h = 0.25 * self.k
            if self.st.labels == "mtext_area" and txt in ref:
                body = f"{txt}\\P{ref[txt]['area']:.2f} m²"
                self.msp.add_mtext(body, dxfattribs={"layer": self.lay("text"), "char_height": h,
                                                     "insert": self.W(pt), "attachment_point": 5})
            else:
                t = self.msp.add_text(txt, dxfattribs={"layer": self.lay("text"), "height": h})
                t.set_placement(self.W(pt), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)

    def dims(self):
        if not self.st.dims:
            return
        solid = self.spec.solid(with_openings=False)
        minx, miny, maxx, maxy = solid.bounds
        try:
            d = self.msp.add_linear_dim(base=self.W((minx, miny - 1.5)), p1=self.W((minx, miny)), p2=self.W((maxx, miny)),
                                        dxfattribs={"layer": self.lay("dim")})
            d.render()
            d = self.msp.add_linear_dim(base=self.W((minx - 1.5, miny)), p1=self.W((minx, miny)), p2=self.W((minx, maxy)),
                                        angle=90, dxfattribs={"layer": self.lay("dim")})
            d.render()
            # a wall-thickness dimension on the first straight wall
            for w in self.spec.walls:
                if not w.arc:
                    m = (np.array(w.p0) + np.array(w.p1)) / 2
                    a = m - w.n * w.t / 2
                    b = m + w.n * w.t / 2
                    ang = math.degrees(math.atan2(w.n[1], w.n[0]))
                    d = self.msp.add_linear_dim(base=self.W(tuple(m + w.u * 0.6)), p1=self.W(tuple(a)), p2=self.W(tuple(b)),
                                                angle=ang, dxfattribs={"layer": self.lay("dim")})
                    d.render()
                    break
        except Exception:
            pass

    def grid(self):
        if not self.spec.grid or not self.spec.columns:
            return
        xs = sorted({c[0] for c in self.spec.columns})
        ys = sorted({c[1] for c in self.spec.columns})
        minx, miny, maxx, maxy = self.spec.solid(False).bounds
        for x in xs:
            self.msp.add_line(self.W((x, miny - 2)), self.W((x, maxy + 2)), dxfattribs={"layer": self.lay("grid"), "linetype": "CENTER"})
        for y in ys:
            self.msp.add_line(self.W((minx - 2, y)), self.W((maxx + 2, y)), dxfattribs={"layer": self.lay("grid"), "linetype": "CENTER"})

    def floor(self):
        if not self.st.floor_hatch:
            return
        for r in self.spec.reference()["rooms"][:3]:
            h = self.msp.add_hatch(dxfattribs={"layer": self.lay("floor")})
            h.set_pattern_fill("ANSI37", scale=0.5 * self.k if self.st.units != "m" else 0.5)
            h.paths.add_polyline_path([self.W(p) for p in r["polygon"][:-1]], is_closed=True)

    def render(self):
        for a in self.st.assemblies:
            a["block"] = self.doc.blocks.new(a["name"])
        self.walls()
        self.openings()
        self.columns()
        self.furniture()
        self.labels()
        self.dims()
        self.grid()
        self.floor()
        for a in self.st.assemblies:
            ins = self.W(a["insert"])
            self.msp.add_blockref(a["name"], ins, dxfattribs={"rotation": a["rotation"], "xscale": -1 if a.get("mirror") else 1,
                                                             "layer": self.lay("wall")})
        return self.doc


def _lerp(p, q, t):
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def _collinear(a, b, c, tol=1e-9):
    cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    L = math.hypot(c[0] - a[0], c[1] - a[1]) or 1.0
    return abs(cr) / L < tol * 10 and ((b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])) > 0


def render_plan(spec: PlanSpec, style: Style):
    doc = _Writer(spec, style).render()
    ref = spec.reference()
    ref["frame"] = {"units": style.units, "to_m": UNIT_TO_M[style.units], "offset_m": list(style.offset),
                    "formula": "dxf = (world_m + offset_m) / to_m"}
    return doc, ref
