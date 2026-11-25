"""Regular roof outlines: a drawn tile hatch (dormer notches, label boxes, a core pocket, wobbly
sides, any orientation) becomes the outline a hip roof is set out on, and the roof built on it is
made of exact planes."""
import math

import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

from architectural_converter.reconstruction3d.roof import RoofSurface, dormer_meshes, hip_roof
from architectural_converter.topology.roof_plan import dominant_axis, rectify, regularize_roof


def _dormer(x, y, width, depth, inwards):
    """An arrow-shaped notch (dormer seen in plan) opening at (x, y) on the eave line."""
    w, d = width / 2, depth
    ux, uy = inwards
    vx, vy = -uy, ux
    pts = [(-w, 0.0), (-w, d - w), (0.0, d), (w, d - w), (w, 0.0), (w, -0.5), (-w, -0.5)]
    return Polygon([(x + s * vx + t * ux, y + s * vy + t * uy) for s, t in pts])


def _drawn_roof(angle=0.0, wobble=0.02, seed=1):
    """A 20 x 17 m attic roof as a hatch would draw it: two dormers on the long sides and one on each
    short side, a label box written on it, a 6 x 6 m core pocket open to a 3.5 m entrance recess,
    sides that wander by ``wobble``. Returns (hatch, core rooms, core walls, recess) rotated by ``angle``."""
    rng = np.random.default_rng(seed)
    body = box(0, 0, 20, 17)
    pocket = box(7, 2.6, 13, 8.6)
    recess = box(8.25, -0.5, 11.75, 2.6)
    notches = [_dormer(5, 17, 1.0, 2.6, (0, -1)), _dormer(14, 17, 1.0, 2.6, (0, -1)),
               _dormer(4, 0, 0.8, 2.4, (0, 1)), _dormer(16, 0, 0.8, 2.4, (0, 1)),
               _dormer(0, 12, 1.4, 2.9, (1, 0)), _dormer(20, 12, 1.4, 2.9, (-1, 0))]
    label = box(9, 12, 11.2, 12.5)
    hatch = body.difference(pocket).difference(recess).difference(label)
    for n in notches:
        hatch = hatch.difference(n)
    # the hatch's sides wander a little
    P = np.asarray(hatch.exterior.coords)
    P[:-1] += rng.uniform(-wobble, wobble, P[:-1].shape)
    P[-1] = P[0]
    hatch = Polygon(P, [h.coords for h in hatch.interiors]).buffer(0)
    rooms = [box(7.3, 2.9, 9.4, 8.3), box(9.6, 2.9, 12.7, 5.5), box(9.6, 5.7, 12.7, 8.3)]
    walls = [box(7, 2.6, 13, 2.9), box(7, 8.3, 13, 8.6), box(7, 2.6, 7.3, 8.6), box(12.7, 2.6, 13, 8.6),
             box(9.4, 2.9, 9.6, 8.3), box(9.6, 5.5, 12.7, 5.7), box(0, 0, 0.3, 17)]
    o = (10, 8.5)
    rot = lambda g: rotate(translate(g, 100, 50), angle, origin=(o[0] + 100, o[1] + 50))  # noqa: E731
    return rot(hatch), [rot(r) for r in rooms], [rot(w) for w in walls], rot(recess)


def _axis_aligned_share(poly):
    th = dominant_axis(poly)
    P = np.asarray(poly.exterior.coords)
    E = P[1:] - P[:-1]
    a = (np.arctan2(E[:, 1], E[:, 0]) - th) % (math.pi / 2)
    ok = np.minimum(a, math.pi / 2 - a) < 1e-5
    L = np.hypot(E[:, 0], E[:, 1])
    return float(L[ok].sum() / L.sum())


@pytest.mark.parametrize("angle", [0.0, 23.0, -61.5])
def test_drawn_hatch_becomes_a_regular_roof_outline(angle):
    hatch, rooms, walls, recess = _drawn_roof(angle)
    r = regularize_roof(hatch, rooms, walls=walls)
    assert len(r["parts"]) == 1
    part = r["parts"][0]
    # dormers and the label box are closed, the sides are squared to the building in any orientation
    assert not part.interiors and _axis_aligned_share(part) == 1.0
    assert len(part.exterior.coords) - 1 == 8                     # a rectangle with the entrance recess
    d = (math.degrees(dominant_axis(part)) - angle) % 90
    assert min(d, 90 - d) < 0.5
    # six dormers, each found at its mouth on the eave line, as wide and as deep as drawn
    assert len(r["dormers"]) == 6 and r["info"]["dormers"] == 6
    assert sorted(round(d["width"], 1) for d in r["dormers"]) == [0.8, 0.8, 1.0, 1.0, 1.4, 1.4]
    assert all(2.2 < d["depth"] < 3.1 for d in r["dormers"])
    # the core pocket holds rooms: part of the roof body (an abutment); the entrance recess has
    # none and stays out of the roof (its sides are eaves)
    assert len(r["abutments"]) == 1
    core = r["abutments"][0]
    assert abs(core.area - 36.0) < 1.5 and core.difference(part).area < 1e-6
    assert part.intersection(recess).area < 0.3


def test_without_rooms_the_pocket_stays_open_and_dormers_are_optional():
    hatch, rooms, _, recess = _drawn_roof(10.0)
    r = regularize_roof(hatch, [], with_dormers=False)
    assert not r["abutments"] and not r["dormers"]
    assert abs(r["parts"][0].area - (340 - 36 - 3.5 * 2.6)) < 2.0    # the pocket and the recess stay out
    # without the walls the core is its rooms grown by a wall's thickness: still the pocket
    g = regularize_roof(hatch, rooms)
    assert len(g["abutments"]) == 1 and abs(g["abutments"][0].area - 36.0) < 2.5
    assert g["parts"][0].intersection(recess).area < 0.3


def test_rectify_squares_a_wobbly_outline_but_keeps_real_angles():
    L = Polygon([(0, 0), (12, 0.08), (12.05, 5), (5, 5.1), (5.04, 12), (0.03, 11.95)])
    q = rectify(L)[0]
    assert _axis_aligned_share(q) == 1.0 and abs(q.area - L.area) < 0.5
    # a real diagonal side (a bay) is far outside the tolerance and is kept
    bay = Polygon([(0, 0), (10, 0), (10, 6), (8, 8), (0, 8)])
    b = rectify(bay)[0]
    assert abs(b.area - bay.area) < 1e-6 and len(b.exterior.coords) == 6


def _normals(V, F):
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    a = np.linalg.norm(n, axis=1)
    keep = a > 1e-10
    return n[keep] / a[keep, None]


@pytest.mark.parametrize("ring", [
    [(0, 0), (12, 0), (12, 5), (5, 5), (5, 12), (0, 12)],                                   # L
    [(0, 0), (20, 0), (20, 17), (11.75, 17), (11.75, 14), (8.25, 14), (8.25, 17), (0, 17)],  # recess
    [(0, 0), (9, 0), (13, 4), (13, 10), (0, 10)],                                           # a bay
])
def test_every_roof_triangle_lies_on_a_roof_plane(ring):
    """The straight skeleton is meshed by its planes: no triangle straddles a hip or a valley, so
    every triangle rises at exactly the roof pitch (a flat deck only above the height limit)."""
    pitch = 35.0
    r = hip_roof(ring, pitch, thickness_m=0.0)
    V, F = _both(r["courses"])
    n = _normals(V, F)
    assert np.allclose(np.abs(n[:, 2]), math.cos(math.radians(pitch)), atol=1e-6)
    # it covers the outline exactly once
    t = V[F]
    a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    assert abs(0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() - Polygon(ring).area) < 1e-3
    assert V[:, 2].max() <= r["rise"] + 1e-9


def _both(courses):
    (V0, F0), (V1, F1) = courses
    return np.vstack([V0, V1]), np.vstack([F0, F1 + len(V0)])


def test_core_abutment_gets_a_flat_top_and_closed_sides():
    fp = {"exterior": [(0, 0), (20, 0), (20, 17), (11.75, 17), (11.75, 14), (8.25, 14), (8.25, 17), (0, 17)],
          "holes": [], "abutments": [[(7, 8.4), (13, 8.4), (13, 14), (7, 14)]]}
    r = hip_roof(fp, 30.0, max_rise=2.35, eave_z=1.0)
    s = r["surface"]
    assert abs(s.height(10, 10) - 3.35) < 1e-9                         # the core's top
    assert s.height(10, 15.5) is None                                   # the entrance recess is open air
    assert abs(s.height(3, 0.5) - (1.0 + 0.5 * math.tan(math.radians(30)))) < 1e-6
    # tiles stop at the core; its flat top is part of the deck; sides close it down to the roof
    V, F = _both(r["courses"])
    core = box(7, 8.4, 13, 14)
    cent = V[F].mean(axis=1)
    from shapely.geometry import Point
    assert not [c for c in cent if core.buffer(-1e-6).contains(Point(c[0], c[1]))]
    Vd, Fd = r["deck"]
    assert len(Fd) and abs(Vd[:, 2].max() - 3.35) < 1e-9
    Vs, Fs = r["abutment_sides"]
    assert len(Fs) and Vs[:, 2].min() >= 1.0 - 1e-9 and abs(Vs[:, 2].max() - 3.35) < 1e-9
    # a wall a few millimetres outside the eave line still counts as standing on it
    assert abs(s.height(-0.005, 5, near=0.3) - 1.0) < 1e-3 and s.height(-1.0, 5, near=0.3) is None


def test_dormer_stands_on_the_eave_and_runs_into_the_roof():
    fp = [(0, 0), (20, 0), (20, 17), (0, 17)]
    s = RoofSurface(fp, 30.0, 2.35, 1.0)
    d = {"mouth": [(4.6, 17.0), (5.4, 17.0)], "tip": (5.0, 14.5)}
    walls, roof = dormer_meshes(d, s, 1.0)
    zr = s.field(5.0, 14.5)
    assert abs(roof[0][:, 2].max() - zr) < 1e-9 and walls[0][:, 2].min() == 0.0
    xy = np.vstack([walls[0][:, :2], roof[0][:, :2]])
    assert xy[:, 0].min() >= 4.6 - 1e-9 and xy[:, 0].max() <= 5.4 + 1e-9 and xy[:, 1].min() >= 14.5 - 1e-9
    # a dormer whose tip is barely above the eave is left out
    assert dormer_meshes({"mouth": [(4.6, 17.0), (5.4, 17.0)], "tip": (5.0, 16.6)}, s, 1.0) is None
