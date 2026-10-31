"""Benchmark plan library: the drawing categories of the brief, each with exact ground truth."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import box

from .dxf_writer import Style
from .plan_spec import PlanSpec, WallSpec


# ---------------------------------------------------------------------------- building blocks
def apartment(spec: PlanSpec, x0=0.0, y0=0.0, mirror=False, ext=0.3, part=0.12, prefix="", exterior=True,
              furniture=True, entrance=True):
    """12 x 9 m apartment with 7 rooms."""
    def P(x, y):
        return (x0 + (12 - x if mirror else x), y0 + y)

    if exterior:
        spec.chain([P(0, 0), P(12, 0), P(12, 9), P(0, 9)], ext, closed=True)
    spec.wall(P(5, 0), P(5, 9), 0.2)
    spec.wall(P(0, 3.5), P(12, 3.5), part)
    spec.wall(P(5, 5), P(12, 5), part)
    spec.wall(P(8.5, 5), P(8.5, 9), part)
    spec.wall(P(8, 0), P(8, 3.5), part)
    s = -1 if mirror else 1
    if entrance:
        spec.opening_at(P(12, 4.25), 1.0, "door", swing=-s)      # entrance
    spec.opening_at(P(2.5, 3.5), 1.2, "opening")                 # kitchen / living passage
    spec.opening_at(P(5, 4.25), 0.9, "door", swing=-s)
    spec.opening_at(P(6.5, 3.5), 0.8, "door", swing=-1)
    spec.opening_at(P(10, 3.5), 0.8, "door", swing=-1, hinge="end")
    spec.opening_at(P(6.0, 5), 0.9, "door", swing=1)
    spec.opening_at(P(10.0, 5), 0.9, "door", swing=1, hinge="end")
    spec.opening_at(P(2.5, 9), 2.0, "window")
    spec.opening_at(P(2.5, 0), 1.2, "window")
    spec.opening_at(P(6.75, 9), 1.5, "window")
    spec.opening_at(P(10.25, 9), 1.5, "window")
    spec.opening_at(P(6.5, 0), 0.6, "window")
    if not mirror:
        spec.opening_at(P(0, 6.25), 1.5, "window")
    for name, pt in (("KITCHEN", (2.5, 1.75)), ("LIVING", (2.5, 6.25)), ("BATH", (6.5, 1.75)),
                     ("STORAGE", (10, 1.75)), ("HALL", (8.5, 4.25)), ("BEDROOM", (6.75, 7.0)),
                     ("BEDROOM 2", (10.25, 7.0))):
        spec.label(prefix + name if prefix else name, P(*pt))
    if furniture:
        spec.furniture += [(*P(6.75, 7.6), 1.6, 2.0, 0), (*P(10.25, 7.6), 1.4, 2.0, 0),
                           (*P(2.5, 6.0), 1.6, 0.9, 0), (*P(0.65, 1.6), 0.6, 2.4, 0)]


def rotate_spec(spec: PlanSpec, deg: float, about=(0, 0)):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))

    def R(p):
        x, y = p[0] - about[0], p[1] - about[1]
        return (about[0] + x * c - y * s, about[1] + x * s + y * c)
    for w in spec.walls:
        w.p0, w.p1 = R(w.p0), R(w.p1)
        if w.arc:
            cx, cy, r, a0, a1 = w.arc
            ncx, ncy = R((cx, cy))
            w.arc = (ncx, ncy, r, a0 + deg, a1 + deg)
    spec.labels = [(t, R(p)) for t, p in spec.labels]
    spec.columns = [(*R((x, y)), w, h, sh) for x, y, w, h, sh in spec.columns]
    spec.furniture = [(*R((x, y)), w, h, a + deg) for x, y, w, h, a in spec.furniture]


# ---------------------------------------------------------------------------- plans
def p01_residential_apartment():
    s = PlanSpec("01_residential_apartment", "Standard apartment, AIA layers, mm, double lines, door/window blocks")
    apartment(s)
    return s, Style(units="mm", layers="aia", wall_repr="lines", door_repr="block", window_repr="block",
                    furniture_repr="block", labels="mtext_area")


def p02_multi_room_residential():
    s = PlanSpec("02_multi_room_residential", "Two mirrored apartments, French layers, cm, closed polylines, loose doors/windows")
    apartment(s, 0, 0, prefix="A ")
    # second unit shares the party wall x = 12 (its exterior walls must exist before its openings are placed)
    s.chain([(12, 0), (24, 0), (24, 9), (12, 9)], 0.3)
    apartment(s, 12, 0, mirror=True, prefix="B ", exterior=False, entrance=False)
    s.opening_at((19.75, 0), 1.0, "door", swing=1)              # unit B entrance (kitchen side)
    return s, Style(units="cm", layers="french", wall_repr="polylines", door_repr="loose", window_repr="loose",
                    window_lines=4, offset=(1000.0, 2000.0))


def p03_commercial():
    s = PlanSpec("03_commercial", "Retail unit with column grid, back-of-house rooms, storefront glazing, hatched walls")
    s.chain([(0, 0), (30, 0), (30, 20), (0, 20)], 0.3, closed=True)
    s.wall((0, 15), (30, 15), 0.2)
    for x in (8, 11, 14, 22):
        s.wall((x, 15), (x, 20), 0.15)
    for k, x in enumerate((4, 9.5, 12.5, 18, 26)):
        s.opening_at((x, 15), 0.9 if k else 1.0, "door", swing=-1)
    s.opening_at((15, 0), 1.8, "double_door", swing=1)
    for x in (4.5, 9.0, 21.0, 25.5):
        s.opening_at((x, 0), 3.0, "window")
    s.opening_at((30, 17.5), 1.2, "door", swing=1)
    for x in (6, 12, 18, 24):
        for y in (5, 10):
            s.columns.append((x, y, 0.5, 0.5, "rect"))
    for name, pt in (("SALES AREA", (15, 7.5)), ("STORAGE", (4, 17.5)), ("WC", (9.5, 17.5)), ("WC 2", (12.5, 17.5)),
                     ("OFFICE", (18, 17.5)), ("PLANT ROOM", (26, 17.5))):
        s.label(name, pt)
    s.grid = True
    return s, Style(units="m", layers="generic", wall_repr="lines", wall_hatch=True, door_repr="block",
                    window_repr="block", column_repr="polyline_hatch")


def p04_office_floor():
    s = PlanSpec("04_office_floor", "Office floor, central corridor, thin partitions, inches, double doors")
    W, H = 24.0, 14.0
    s.chain([(0, 0), (W, 0), (W, H), (0, H)], 0.3, closed=True)
    s.wall((0, 6), (W, 6), 0.15)
    s.wall((0, 8), (W, 8), 0.15)
    for x in (4, 8, 12, 16, 20):
        s.wall((x, 0), (x, 6), 0.1)
    for x in (8, 12, 16, 20):
        s.wall((x, 8), (x, H), 0.1)
    for k in range(6):
        cx = 2 + 4 * k
        s.opening_at((cx, 6), 0.9, "door", swing=-1)
        s.opening_at((cx, 0), 1.8, "window")
        s.label(f"OFFICE {k + 1}", (cx, 3))
    s.opening_at((4, 8), 0.9, "door", swing=1)
    s.label("MEETING", (4, 11))
    s.opening_at((4, H), 3.0, "window")
    for k in range(4):
        cx = 10 + 4 * k
        s.opening_at((cx, 8), 0.9, "door", swing=1, hinge="end")
        s.opening_at((cx, H), 1.8, "window")
        s.label(f"OFFICE {k + 7}", (cx, 11))
    s.opening_at((0, 7), 1.6, "double_door", swing=1)
    s.opening_at((W, 7), 1.6, "double_door", swing=-1)
    s.label("CORRIDOR", (12, 7))
    for k in range(6):
        s.furniture.append((2 + 4 * k, 4.2, 1.6, 0.8, 0))
    return s, Style(units="in", layers="aia", wall_repr="lines", door_repr="block", window_repr="block")


def p05_hotel_floor():
    s = PlanSpec("05_hotel_floor", "Hotel floor, double-loaded corridor, room modules as nested blocks (rotated/mirrored)")
    n = 6
    W = 4.0 * n
    s.chain([(0, 0), (W, 0), (W, 16), (0, 16)], 0.3, closed=True)
    s.wall((0, 7), (W, 7), 0.2)
    s.wall((0, 9), (W, 9), 0.2)
    for k in range(1, n):
        s.wall((4 * k, 0), (4 * k, 7), 0.15)
        s.wall((4 * k, 9), (4 * k, 16), 0.15)
    asm = []
    for k in range(n):
        x = 4 * k
        # bathroom near corridor: bottom rooms y 4.6..7, top rooms y 9..11.4
        s.wall((x + 2.2, 4.6), (x + 2.2, 7), 0.1)
        s.wall((x, 4.6), (x + 2.2, 4.6), 0.1)
        s.wall((x + 2.2, 9), (x + 2.2, 11.4), 0.1)
        s.wall((x, 11.4), (x + 2.2, 11.4), 0.1)
        s.opening_at((x + 3.1, 7), 0.9, "door", swing=-1)
        s.opening_at((x + 3.1, 9), 0.9, "door", swing=1)
        s.opening_at((x + 1.1, 4.6), 0.7, "door", swing=-1)
        s.opening_at((x + 1.1, 11.4), 0.7, "door", swing=1)
        s.opening_at((x + 2.0, 0), 1.6, "window")
        s.opening_at((x + 2.0, 16), 1.6, "window")
        s.label(f"ROOM {101 + k}", (x + 2.0, 2.3))
        s.label(f"ROOM {201 + k}", (x + 2.0, 13.7))
        s.label("BATH", (x + 1.1, 5.8))
        s.label("BATH", (x + 1.1, 10.2))
        s.furniture += [(x + 2.0, 1.6, 1.6, 2.0, 0), (x + 2.0, 14.4, 1.6, 2.0, 0)]
        asm.append({"name": f"ROOM_MODULE_B{k + 1}", "polygon": box(x, 0, x + 4, 7), "insert": (x, 0), "rotation": 0,
                    "mirror": k % 2 == 1})
        asm.append({"name": f"ROOM_MODULE_T{k + 1}", "polygon": box(x, 9, x + 4, 16), "insert": (x + 4, 16),
                    "rotation": 180, "mirror": False})
    s.opening_at((0, 8), 1.6, "double_door", swing=1)
    s.label("CORRIDOR", (12, 8))
    return s, Style(units="mm", layers="aia", wall_repr="lines", door_repr="block", window_repr="block",
                    assemblies=asm, dims=False)


def p06_irregular():
    s = PlanSpec("06_irregular", "Irregular L-shaped plan with chamfer and 120-degree wing, unitless header, survey offset")
    t = 0.3
    pts = [(0, 0), (14, 0), (14, 6), (11, 9), (7, 9), (7 - 4 * math.cos(math.radians(60)), 9 + 4 * math.sin(math.radians(60))),
           (0, 9 + 4 * math.sin(math.radians(60)))]
    s.chain(pts, t, closed=True)
    s.wall((7, 0), (7, 9), 0.2)
    s.wall((0, 5), (7, 5), 0.12)
    s.wall((7, 4), (14, 4), 0.12)
    s.opening_at((3.5, 5), 0.9, "door", swing=1)
    s.opening_at((7, 7), 0.9, "door", swing=1)
    s.opening_at((10.5, 4), 0.9, "door", swing=1)
    s.opening_at((3.5, 0), 1.5, "window")
    s.opening_at((10.5, 0), 1.5, "window")
    s.opening_at((12.5, 7.5), 1.2, "window")
    s.opening_at((14, 2), 1.0, "door", swing=1)
    for name, pt in (("STUDIO", (3.5, 2.5)), ("LOUNGE", (3.5, 8.5)), ("KITCHEN", (10.5, 2)), ("DINING", (10.5, 6.5))):
        s.label(name, pt)
    return s, Style(units="mm", header_units=None, layers="generic", wall_repr="lines", door_repr="block",
                    window_repr="block", offset=(350000.0, 5600000.0))


def p07_diagonal_curved():
    s = PlanSpec("07_diagonal_curved", "Plan rotated 30 degrees with a curved bay wall and a 45-degree wall")
    t = 0.25
    s.wall((0, 0), (10, 0), t)
    s.wall((10, 0), (10, 2), t)
    s.arc_wall(10, 4, 2, -90, 90, t)   # semicircular bay from (10,2) to (10,6)
    s.wall((10, 6), (10, 8), t)
    s.wall((10, 8), (0, 8), t)
    s.wall((0, 8), (0, 0), t)
    s.wall((5, 0), (5, 5), 0.15)
    s.wall((5, 5), (8, 8), 0.15)       # 45-degree wall
    s.opening_at((5, 2.5), 0.9, "door", swing=1)
    s.opening_at((2.5, 0), 1.5, "window")
    s.opening_at((2.5, 8), 1.5, "window")
    s.opening_at((0, 4), 1.0, "door", swing=1)
    s.label("STUDY", (2.5, 4.0))
    s.label("LIVING", (8.0, 3.0))
    rotate_spec(s, 30)
    return s, Style(units="m", layers="generic", wall_repr="lines", door_repr="block", window_repr="block")


def p08_fragmented():
    s = PlanSpec("08_fragmented", "Apartment with fragmented, jittered, partially duplicated linework, loose doors")
    apartment(s)
    return s, Style(units="mm", layers="aia", wall_repr="lines", door_repr="loose", window_repr="block",
                    fragment=0.6, duplicates=0.15, jitter=0.0004, seed=11)


def p09_poor_layers():
    s = PlanSpec("09_poor_layers", "Everything on layer 0 (walls, doors, windows, furniture, text), cm")
    apartment(s)
    return s, Style(units="cm", layers="poor", wall_repr="lines", door_repr="loose", window_repr="loose",
                    furniture_repr="polyline", dims=False)


def p10_blocks_hatches_overlaps():
    s = PlanSpec("10_blocks_hatches_overlaps", "Hatched walls, floor hatches, column blocks, furniture blocks, duplicates")
    apartment(s)
    s.columns += [(3.0, 8.4, 0.3, 0.3, "rect"), (11.4, 8.4, 0.3, 0.3, "rect")]
    return s, Style(units="m", layers="cryptic", wall_repr="lines", wall_hatch=True, floor_hatch=True,
                    door_repr="block", window_repr="block", column_repr="block", furniture_repr="block",
                    duplicates=0.25, seed=5)


def p11_single_line():
    s, _ = p04_office_floor()
    s.name = "11_single_line"
    s.description = "Office floor drawn with single centre lines (no thickness information)"
    return s, Style(units="m", layers="generic", wall_repr="centerline", door_repr="block", window_repr="block")


def p12_mline():
    s = PlanSpec("12_mline", "Apartment drawn with MLINE walls")
    apartment(s, furniture=False)
    return s, Style(units="mm", layers="aia", wall_repr="mline", door_repr="block", window_repr="block")


def p13_wide_polyline():
    s = PlanSpec("13_wide_polyline", "Apartment drawn with wide (constant-width) polylines")
    apartment(s, furniture=False)
    return s, Style(units="cm", layers="generic", wall_repr="wide_polyline", door_repr="block", window_repr="block")


PLANS = [p01_residential_apartment, p02_multi_room_residential, p03_commercial, p04_office_floor, p05_hotel_floor,
         p06_irregular, p07_diagonal_curved, p08_fragmented, p09_poor_layers, p10_blocks_hatches_overlaps,
         p11_single_line, p12_mline, p13_wide_polyline]
