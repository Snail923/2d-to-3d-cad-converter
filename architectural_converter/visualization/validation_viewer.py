"""Self-contained HTML report: overlay drawing + summary + validation issues."""
from __future__ import annotations

import html
import json


def html_report(st, overlay_svg: str) -> str:
    m = st.model
    counts = m["statistics"]["counts"]
    rows = []
    for i in m["validation"]["issues"]:
        src = ", ".join(f"{s['type']}#{s['handle'] or '-'}@{s['layer']}" for s in i["source_entities"][:4])
        rows.append(f"<tr class='{i['severity']}'><td>{i['id']}</td><td>{i['severity']}</td><td>{i['code']}</td>"
                    f"<td>{html.escape(str(i['entity_id'] or ''))}</td><td>{html.escape(i['description'])}</td>"
                    f"<td>{html.escape(i['suggestion'])}</td><td>{i['location']}</td><td>{html.escape(src)}</td></tr>")
    units = m["source"]
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Extraction report - {html.escape(units['file'])}</title>
<style>body{{font-family:system-ui,sans-serif;margin:20px;color:#222}} table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ddd;padding:4px 6px;vertical-align:top}} tr.warning td{{background:#fff8e1}} tr.error td{{background:#ffebee}}
.svg{{border:1px solid #ccc;overflow:auto;max-height:80vh}} code{{background:#f4f4f4;padding:1px 4px}}</style></head><body>
<h1>Architectural extraction report</h1>
<p><b>File:</b> {html.escape(units['file'])}<br><b>Units:</b> {units['dxf_units']} (confidence {units['unit_confidence']})
<br><b>Counts:</b> <code>{html.escape(json.dumps(counts))}</code><br><b>Timings (s):</b> <code>{html.escape(json.dumps(m['statistics']['timings_s']))}</code></p>
<p>Legend: blue = exterior wall, green = interior / partition, dashed = needs review, red = rejected, orange = door,
light blue = window, purple = unidentified opening, magenta dashed = missing-wall hypothesis, yellow = room.</p>
<div class="svg">{overlay_svg}</div>
<h2>Validation ({m['validation']['summary'].get('total', 0)} issues)</h2>
<table><tr><th>id</th><th>severity</th><th>code</th><th>entity</th><th>problem</th><th>suggestion</th><th>location (m)</th><th>source DXF entities</th></tr>
{''.join(rows)}</table></body></html>"""
