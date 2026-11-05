"""SymPointV2 / FloorPlanCAD classes, colours and groups.

Names and colours are those of ``svgnet/data/svg3.py`` (SVG_CATEGORIES) in the SymPointV2
repository; the semantic id used by the model is the list index (0-34), 35 is background."""
from __future__ import annotations

CATEGORIES = [
    ("single door", (224, 62, 155)), ("double door", (157, 34, 101)), ("sliding door", (232, 116, 91)),
    ("folding door", (101, 54, 72)), ("revolving door", (172, 107, 133)), ("rolling door", (142, 76, 101)),
    ("window", (96, 78, 245)), ("bay window", (26, 2, 219)), ("blind window", (63, 140, 221)),
    ("opening symbol", (233, 59, 217)),
    ("sofa", (122, 181, 145)), ("bed", (94, 150, 113)), ("chair", (66, 107, 81)), ("table", (123, 181, 114)),
    ("TV cabinet", (94, 150, 83)), ("Wardrobe", (66, 107, 59)), ("cabinet", (145, 182, 112)),
    ("gas stove", (152, 147, 200)), ("sink", (113, 151, 82)), ("refrigerator", (112, 103, 178)),
    ("airconditioner", (81, 107, 58)), ("bath", (172, 183, 113)), ("bath tub", (141, 152, 83)),
    ("washing machine", (80, 72, 147)), ("squat toilet", (100, 108, 59)), ("urinal", (182, 170, 112)),
    ("toilet", (238, 124, 162)),
    ("stairs", (247, 206, 75)), ("elevator", (237, 112, 45)), ("escalator", (233, 59, 46)),
    ("row chairs", (172, 107, 151)), ("parking spot", (102, 67, 62)),
    ("wall", (167, 92, 32)), ("curtain wall", (121, 104, 178)), ("railing", (64, 52, 105)),
]
CLASSES = [n for n, _ in CATEGORIES]
COLORS = {n: c for n, c in CATEGORIES}
BG = len(CLASSES)            # 35 = background / no label

GROUPS = {
    "wall": ["wall", "curtain wall"],
    "door": ["single door", "double door", "sliding door", "folding door", "revolving door", "rolling door"],
    "window": ["window", "bay window", "blind window"],
    "opening": ["opening symbol"],
    "stair": ["stairs", "escalator"],
    "railing": ["railing"],
    "furniture": ["sofa", "bed", "chair", "table", "TV cabinet", "Wardrobe", "cabinet", "gas stove", "sink",
                  "refrigerator", "airconditioner", "bath", "bath tub", "washing machine", "squat toilet",
                  "urinal", "toilet", "row chairs"],
    "other": ["elevator", "parking spot"],
}
GROUP_OF = {c: g for g, cs in GROUPS.items() for c in cs}
