"""Text / label analysis: room names, room numbers, area annotations, tags."""
from __future__ import annotations

import re
import unicodedata

from ..models.building import TextLabel
from .base_detector import BaseDetector, register

TAG_RE = re.compile(r"^[A-Z]{1,3}[-_ ]?\d{1,4}[A-Z]?$")
NUM_RE = re.compile(r"^\d{1,4}[A-Z]?$")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s.lower()).strip()


def room_type_of(text: str, keywords: dict) -> str | None:
    t = _norm(text)
    generic = None
    for rtype, words in keywords.items():
        for w in words:
            if re.search(r"(^|[^a-z])" + re.escape(w) + r"($|[^a-z])", t):
                if rtype == "guest_room":
                    generic = generic or rtype
                    break
                return rtype
    return generic


def area_of(text: str, patterns) -> float | None:
    t = _norm(text).replace("m 2", "m2")
    for p in patterns:
        m = re.search(p, t)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                return None
    return None


@register
class TextDetector(BaseDetector):
    name = "text_detector"
    produces = "label"

    def detect(self, drawing, context):
        cc = self.cfg["classification"]
        out = []
        n = 0
        for g in drawing.entities:
            if g.kind != "text" or not g.valid:
                continue
            txt = str(g.data.get("text", "")).strip()
            if not txt:
                continue
            rtype = room_type_of(txt, cc["room_keywords"])
            area = area_of(txt, cc["area_text_patterns"])
            first = txt.splitlines()[0].strip() if txt.splitlines() else txt
            if g.role in ("door", "window", "column", "furniture", "symbol") and not rtype:
                cat = "tag"
            elif rtype:
                cat = "room_name"
            elif area is not None:
                cat = "area"
            elif NUM_RE.match(first.upper()):
                cat = "room_number"
            elif TAG_RE.match(first.upper()):
                cat = "tag"
            else:
                cat = "other"
            n += 1
            out.append(TextLabel(id=f"label_{n:04d}", raw_id=g.raw_id, text=txt,
                                 position=(float(g.points[0][0]), float(g.points[0][1])),
                                 height=float(g.data.get("height", 0) or 0), category=cat, room_type=rtype,
                                 area_value=area, layer=g.layer))
        return out
