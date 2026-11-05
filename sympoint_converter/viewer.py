"""Raw SymPointV2 label viewer: one self-contained HTML page per drawing.

Every primitive is drawn in the colour of the class SymPointV2 gave it (grey = no label or below
the score threshold), so the model is judged on its own, before any geometry is built from it.
The page also has per-class and per-layer summaries, the instances (doors, windows, ...) and a
score slider; hovering a line shows its class, score, layer, entity type and DXF handle."""
from __future__ import annotations

import html
import json

import numpy as np

from .classes import BG, CLASSES, COLORS, GROUP_OF, GROUPS
from .sympoint_io import COMMAND_NAMES

GROUP_ORDER = ["wall", "door", "window", "opening", "stair", "railing", "furniture", "other"]


def summarize(prims, labels: dict, drawing=None) -> dict:
    """Per class / group / layer statistics of the labelled primitives (lengths in metres)."""
    lab = labels.get("primitives", {})
    tot_len = sum(p[4] for p in prims)
    cls_stats = {c: {"primitives": 0, "length_m": 0.0, "score_sum": 0.0} for c in CLASSES}
    layer_stats: dict[str, dict] = {}
    unl = {"primitives": 0, "length_m": 0.0}
    for k, p in enumerate(prims):
        L = p[4]
        ls = layer_stats.setdefault(p[1], {"length_m": 0.0, "groups": {}, "classes": {}})
        ls["length_m"] += L
        lv = lab.get(k) or lab.get(str(k))
        if lv is None:
            unl["primitives"] += 1
            unl["length_m"] += L
            ls["groups"]["unlabelled"] = ls["groups"].get("unlabelled", 0.0) + L
            continue
        c = CLASSES[int(lv[0])]
        s = cls_stats[c]
        s["primitives"] += 1
        s["length_m"] += L
        s["score_sum"] += float(lv[1]) * L
        g = GROUP_OF.get(c, "other")
        ls["groups"][g] = ls["groups"].get(g, 0.0) + L
        ls["classes"][c] = ls["classes"].get(c, 0.0) + L
    lab_len = tot_len - unl["length_m"]
    classes = []
    for c in CLASSES:
        s = cls_stats[c]
        if s["primitives"]:
            classes.append({"class": c, "group": GROUP_OF.get(c, "other"), "primitives": s["primitives"],
                            "length_m": round(s["length_m"], 2),
                            "share_of_labelled": round(s["length_m"] / lab_len, 4) if lab_len else 0.0,
                            "mean_score": round(s["score_sum"] / s["length_m"], 3)})
    groups = {}
    for g in GROUP_ORDER:
        L = sum(x["length_m"] for x in classes if x["group"] == g)
        if L:
            groups[g] = round(L / lab_len, 4) if lab_len else 0.0
    layers = []
    for name, ls in sorted(layer_stats.items(), key=lambda kv: -kv[1]["length_m"]):
        top = max(ls["classes"], key=ls["classes"].get) if ls["classes"] else None
        layers.append({"layer": name, "length_m": round(ls["length_m"], 2), "dominant_class": top,
                       "groups": {g: round(v / ls["length_m"], 3) for g, v in ls["groups"].items()}})
    inst: dict[str, list] = {}
    for i in labels.get("instances", []):
        inst.setdefault(i["label"], []).append(i["score"])
    instances = [{"class": c, "count": len(v), "mean_score": round(float(np.mean(v)), 3)}
                 for c, v in sorted(inst.items(), key=lambda kv: -len(kv[1]))]
    return {"primitives": len(prims), "labelled_primitives": len(prims) - unl["primitives"],
            "total_length_m": round(tot_len, 2),
            "labelled_length_share": round(lab_len / tot_len, 4) if tot_len else 0.0,
            "groups": groups, "classes": classes, "layers": layers, "instances": instances}


def _data(prims, labels, drawing):
    lab = labels.get("primitives", {})
    layers = sorted({p[1] for p in prims})
    lid = {ly: k for k, ly in enumerate(layers)}
    raw = getattr(drawing, "raw_entities", {}) if drawing is not None else {}
    ents, eid = [], {}
    coords, offs, cls, sc, lay, ent, cmd = [], [0], [], [], [], [], []
    for k, p in enumerate(prims):
        rid = p[0]
        if rid not in eid:
            r = raw.get(rid)
            eid[rid] = len(ents)
            ents.append([rid, (r.handle if r else None) or "", (r.dxftype if r else "") or "",
                         (r.block_name if r else None) or ""])
        D = np.asarray(p[5], float)
        coords.extend(np.round(D.reshape(-1), 3).tolist())
        offs.append(len(coords) // 2)
        lv = lab.get(k) or lab.get(str(k))
        cls.append(int(lv[0]) if lv else BG)
        sc.append(round(float(lv[1]), 3) if lv else 0.0)
        lay.append(lid[p[1]])
        ent.append(eid[rid])
        cmd.append(int(p[2]))
    inst = [{"label": i["label"], "score": round(i["score"], 3), "pidx": i.get("pidx", [])}
            for i in labels.get("instances", [])]
    return {"coords": coords, "offs": offs, "cls": cls, "score": sc, "layer": lay, "ent": ent, "cmd": cmd,
            "layers": layers, "ents": ents, "instances": inst,
            "classes": CLASSES, "colors": [list(COLORS[c]) for c in CLASSES],
            "groups": [GROUP_OF.get(c, "other") for c in CLASSES], "group_order": GROUP_ORDER,
            "commands": [COMMAND_NAMES[i] for i in range(4)]}


def render_html(prims, labels: dict, drawing=None, title: str = "SymPointV2 labels", meta: dict | None = None) -> str:
    summ = summarize(prims, labels, drawing)
    data = _data(prims, labels, drawing)
    payload = json.dumps({"data": data, "summary": summ, "meta": meta or {}}, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", payload)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfbfa;--panel:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e3df;--unl:#c9c8c3;--accent:#3b6fd8}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--panel:#1f1f1d;--fg:#ecebe6;--muted:#9c9b95;--line:#33332f;--unl:#4a4945;--accent:#7aa2f7}}
*{box-sizing:border-box}body{margin:0;font:13px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:12px 16px;border-bottom:1px solid var(--line)}h1{font-size:16px;margin:0 0 4px}
.meta{color:var(--muted)}main{display:grid;grid-template-columns:minmax(0,1fr) 380px;height:calc(100vh - 62px)}
@media (max-width:900px){main{grid-template-columns:1fr;height:auto}#view{height:70vh}}
#view{position:relative;overflow:hidden;background:var(--panel);border-right:1px solid var(--line);cursor:grab}
canvas{display:block;width:100%;height:100%}aside{overflow:auto;padding:12px 16px}
h2{font-size:13px;margin:16px 0 6px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
table{border-collapse:collapse;width:100%}td,th{padding:3px 4px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-weight:600;color:var(--muted)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
#tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);padding:6px 8px;border-radius:4px;
box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;font-size:12px;max-width:320px;white-space:normal}
.ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:6px 0}.bar{display:flex;height:8px;border-radius:2px;overflow:hidden;min-width:90px}
.bar i{display:block;height:100%}label{cursor:pointer}button{font:inherit;padding:3px 8px;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:4px;cursor:pointer}
#hint{position:absolute;left:8px;bottom:6px;color:var(--muted);font-size:11px;background:var(--panel);padding:2px 6px;border-radius:3px}
</style></head><body>
<header><h1>__TITLE__</h1><div class="meta" id="meta"></div></header>
<main><div id="view"><canvas id="cv"></canvas><div id="tip"></div><div id="hint">wheel: zoom · drag: pan · hover: details</div></div>
<aside>
<div class="ctl"><label>min score <input type="range" id="thr" min="0" max="1" step="0.05" value="0"></label><span id="thrv">0.00</span>
<button id="fit">Fit</button><label><input type="checkbox" id="dim"> dim unlabelled</label></div>
<h2>Groups (share of labelled length)</h2><div id="groups"></div>
<h2>Classes</h2><table id="classes"></table>
<h2>Instances</h2><table id="inst"></table>
<h2>Layers × groups</h2><table id="layers"></table>
</aside></main>
<script>
const P=__PAYLOAD__;const D=P.data,S=P.summary;
const $=id=>document.getElementById(id);const N=D.cls.length;
const GC={wall:[167,92,32],door:[224,62,155],window:[96,78,245],opening:[233,59,217],stair:[247,206,75],railing:[64,52,105],furniture:[94,150,113],other:[120,120,120],unlabelled:[200,200,195]};
const rgb=c=>`rgb(${c[0]},${c[1]},${c[2]})`;const pct=x=>(100*x).toFixed(1)+'%';
const hidden=new Set();let thr=0,dim=false;
const m=P.meta;$('meta').textContent=[m.file,m.units&&('units '+m.units),S.primitives.toLocaleString()+' primitives',
 pct(S.labelled_length_share)+' of length labelled',m.tiles!=null&&(m.tiles+' tiles of '+m.tile_m+' m'),m.seconds!=null&&('model '+m.seconds+' s')].filter(Boolean).join(' · ');
// groups bar
let gh='<div class="bar" style="height:14px;margin-bottom:6px">';for(const g of D.group_order){const v=S.groups[g];if(v)gh+=`<i style="width:${100*v}%;background:${rgb(GC[g])}" title="${g} ${pct(v)}"></i>`}
gh+='</div><table>';for(const g of D.group_order){const v=S.groups[g];if(v)gh+=`<tr><td><span class="sw" style="background:${rgb(GC[g])}"></span>${g}</td><td class="n">${pct(v)}</td></tr>`}$('groups').innerHTML=gh+'</table>';
// classes
let ch='<tr><th></th><th>class</th><th class="n">lines</th><th class="n">length m</th><th class="n">share</th><th class="n">score</th></tr>';
for(const c of S.classes){const i=D.classes.indexOf(c.class);ch+=`<tr><td><input type="checkbox" checked data-c="${i}"></td><td><span class="sw" style="background:${rgb(D.colors[i])}"></span>${c.class}</td><td class="n">${c.primitives}</td><td class="n">${c.length_m.toFixed(1)}</td><td class="n">${pct(c.share_of_labelled)}</td><td class="n">${c.mean_score.toFixed(2)}</td></tr>`}
ch+=`<tr><td><input type="checkbox" checked data-c="${D.classes.length}"></td><td><span class="sw" style="background:var(--unl)"></span>no label</td><td class="n">${S.primitives-S.labelled_primitives}</td><td class="n"></td><td></td><td></td></tr>`;
$('classes').innerHTML=ch;$('classes').addEventListener('change',e=>{const c=+e.target.dataset.c;e.target.checked?hidden.delete(c):hidden.add(c);draw()});
let ih='<tr><th>class</th><th class="n">count</th><th class="n">mean score</th></tr>';for(const x of S.instances)ih+=`<tr><td>${x.class}</td><td class="n">${x.count}</td><td class="n">${x.mean_score.toFixed(2)}</td></tr>`;
$('inst').innerHTML=S.instances.length?ih:'<tr><td>none reported</td></tr>';
let lh='<tr><th>layer</th><th class="n">length m</th><th>groups</th><th>top class</th></tr>';
for(const l of S.layers.slice(0,60)){let b='<div class="bar">';for(const g of [...D.group_order,'unlabelled']){const v=l.groups[g];if(v)b+=`<i style="width:${100*v}%;background:${rgb(GC[g])}" title="${g} ${pct(v)}"></i>`}
lh+=`<tr><td title="${l.layer}">${l.layer.length>22?l.layer.slice(0,21)+'…':l.layer}</td><td class="n">${l.length_m.toFixed(1)}</td><td>${b}</div></td><td>${l.dominant_class||''}</td></tr>`}
$('layers').innerHTML=lh;
// canvas
const cv=$('cv'),ctx=cv.getContext('2d'),view=$('view');let W=0,H=0,dpr=1;
let minx=1e18,miny=1e18,maxx=-1e18,maxy=-1e18;for(let i=0;i<D.coords.length;i+=2){const x=D.coords[i],y=D.coords[i+1];if(x<minx)minx=x;if(x>maxx)maxx=x;if(y<miny)miny=y;if(y>maxy)maxy=y}
let sc=1,tx=0,ty=0;
function fit(){const w=Math.max(maxx-minx,1e-3),h=Math.max(maxy-miny,1e-3);sc=0.95*Math.min(W/w,H/h);tx=(W-sc*(minx+maxx))/2;ty=(H+sc*(miny+maxy))/2;draw()}
function resize(){dpr=window.devicePixelRatio||1;W=view.clientWidth;H=view.clientHeight;cv.width=W*dpr;cv.height=H*dpr}
const css=getComputedStyle(document.documentElement);
function color(i){const c=D.cls[i];if(c>=D.classes.length||D.score[i]<thr)return null;return D.colors[c]}
function draw(){ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);ctx.lineWidth=1;const unl=css.getPropertyValue('--unl').trim();
 const order=[[],[]];for(let i=0;i<N;i++){const c=color(i);(c?order[1]:order[0]).push(i)}
 for(const pass of [0,1]){for(const i of order[pass]){const c=color(i);const key=c?D.cls[i]:D.classes.length;if(hidden.has(key))continue;
  ctx.strokeStyle=c?rgb(c):unl;ctx.globalAlpha=c?1:(dim?0.25:0.8);ctx.beginPath();
  for(let k=D.offs[i];k<D.offs[i+1];k++){const x=D.coords[2*k]*sc+tx,y=ty-D.coords[2*k+1]*sc;k==D.offs[i]?ctx.moveTo(x,y):ctx.lineTo(x,y)}ctx.stroke()}}
 ctx.globalAlpha=1}
// hover: uniform grid over primitive bounding boxes
const G=new Map(),cell=Math.max(maxx-minx,maxy-miny)/200||1;
for(let i=0;i<N;i++){let a=1e18,b=1e18,c=-1e18,d=-1e18;for(let k=D.offs[i];k<D.offs[i+1];k++){const x=D.coords[2*k],y=D.coords[2*k+1];a=Math.min(a,x);b=Math.min(b,y);c=Math.max(c,x);d=Math.max(d,y)}
 for(let gx=Math.floor(a/cell);gx<=Math.floor(c/cell);gx++)for(let gy=Math.floor(b/cell);gy<=Math.floor(d/cell);gy++){const key=gx+','+gy;(G.get(key)||G.set(key,[]).get(key)).push(i)}}
function dseg(px,py,ax,ay,bx,by){const dx=bx-ax,dy=by-ay,l=dx*dx+dy*dy;let t=l?((px-ax)*dx+(py-ay)*dy)/l:0;t=Math.max(0,Math.min(1,t));const x=ax+t*dx-px,y=ay+t*dy-py;return Math.hypot(x,y)}
function pick(mx,my){const x=(mx-tx)/sc,y=(ty-my)/sc,r=6/sc;let best=-1,bd=r;
 for(let gx=Math.floor((x-r)/cell);gx<=Math.floor((x+r)/cell);gx++)for(let gy=Math.floor((y-r)/cell);gy<=Math.floor((y+r)/cell);gy++){for(const i of G.get(gx+','+gy)||[]){
  for(let k=D.offs[i];k<D.offs[i+1]-1;k++){const d=dseg(x,y,D.coords[2*k],D.coords[2*k+1],D.coords[2*k+2],D.coords[2*k+3]);if(d<bd){bd=d;best=i}}}}return best}
const tip=$('tip');let drag=null;
view.addEventListener('mousemove',e=>{const r=view.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
 if(drag){tx+=mx-drag[0];ty+=my-drag[1];drag=[mx,my];draw();return}
 const i=pick(mx,my);if(i<0){tip.style.display='none';return}const e2=D.ents[D.ent[i]],c=D.cls[i];
 tip.innerHTML=`<b>${c<D.classes.length?D.classes[c]:'no label'}</b>${c<D.classes.length?' · score '+D.score[i].toFixed(2):''}<br>layer ${D.layers[D.layer[i]]}<br>${e2[2]} ${D.commands[D.cmd[i]]}${e2[1]?' · handle '+e2[1]:''}${e2[3]?'<br>block '+e2[3]:''}`;
 tip.style.display='block';tip.style.left=Math.min(mx+14,W-240)+'px';tip.style.top=(my+14)+'px'});
view.addEventListener('mousedown',e=>{const r=view.getBoundingClientRect();drag=[e.clientX-r.left,e.clientY-r.top];view.style.cursor='grabbing'});
window.addEventListener('mouseup',()=>{drag=null;view.style.cursor='grab'});view.addEventListener('mouseleave',()=>tip.style.display='none');
view.addEventListener('wheel',e=>{e.preventDefault();const r=view.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,f=Math.exp(-e.deltaY*0.0015);
 tx=mx-(mx-tx)*f;ty=my-(my-ty)*f;sc*=f;draw()},{passive:false});
$('thr').addEventListener('input',e=>{thr=+e.target.value;$('thrv').textContent=thr.toFixed(2);draw()});
$('dim').addEventListener('change',e=>{dim=e.target.checked;draw()});$('fit').addEventListener('click',fit);
window.addEventListener('resize',()=>{resize();draw()});resize();fit();
</script></body></html>
"""
