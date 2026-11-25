"""Build the PDF report (HTML -> Chromium PDF). All numbers are read from the analysis JSON files."""
import base64, json, os, re, statistics as S, subprocess, sys, datetime

SP = "/tmp/claude-0/-home-user-2d-to-3d-cad-converter/c35bb2e6-6853-5259-a2c6-fc79a906fa3c/scratchpad"
A = "/home/user/analysis"
OUT_HTML = f"{SP}/pdf/report.html"
os.makedirs(f"{SP}/pdf", exist_ok=True)

sim = json.load(open(f"{SP}/bench.json"))
sim_nl = json.load(open(f"{SP}/bench_nolayers.json"))
sim_hy = json.load(open(f"{SP}/bench_hybrid.json"))
real = json.load(open(f"{A}/real_variants.json"))
cmp_ = json.load(open("/home/user/gpu_in/gpu_out/benchmark.compare.json"))
lconf = json.load(open(f"{A}/label_conf.json"))
apt = json.load(open(f"{A}/apt_scores.json"))

M = ["wall_precision", "wall_recall", "room_precision", "room_recall", "door_precision", "door_recall",
     "window_precision", "window_recall", "manual_corrections_required"]
HDR = ["Wall P", "Wall R", "Room P", "Room R", "Door P", "Door R", "Win P", "Win R", "Fixes/dwg"]

def mean_group(res, group, skip=()):
    per = {}
    for n, d in res.items():
        if any(n.startswith(s) for s in skip): continue
        for k, m in d.items():
            if "error" in m or re.sub(r"_s\d+$", "", k) != group: continue
            per.setdefault(n, []).append(m)
    if not per: return None
    return {x: S.mean(S.mean(mm[x] for mm in ms) for ms in per.values()) for x in M} | {"n": len(per)}

def img(path, w=None):
    data = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/jpeg;base64,{data}"' + (f' style="width:{w}"' if w else "") + ">"

def cls(v, k):
    if k == "manual_corrections_required":
        return "good" if v < 0.5 else ("warn" if v < 3 else "bad")
    return "good" if v >= 0.98 else ("warn" if v >= 0.85 else "bad")

def row(name, m, note=""):
    cells = "".join(f'<td class="{cls(m[k], k)}">{m[k]:.2f}</td>' if k != "manual_corrections_required"
                    else f'<td class="{cls(m[k], k)}">{m[k]:.1f}</td>' for k in M)
    return f"<tr><td>{name}{note}</td>{cells}</tr>"

def table(rows, first="Configuration"):
    return (f'<table><thead><tr><th>{first}</th>' + "".join(f"<th>{h}</th>" for h in HDR) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")

def bars(title, items, maxv=1.0, fmt=lambda v: f"{v:.2f}"):
    out = [f'<div class="bgroup"><div class="btitle">{title}</div>']
    for label, v, color in items:
        out.append(f'<div class="brow"><span class="blbl">{label}</span><span class="btrack"><span class="bbar" '
                   f'style="width:{max(0, min(1, v / maxv)) * 100:.1f}%;background:{color}"></span></span>'
                   f'<span class="bval">{fmt(v)}</span></div>')
    return "".join(out) + "</div>"

C1, C2, C3, C4 = "#257de0", "#f47230", "#00b894", "#f2a600"
pct = lambda v: f"{round(v * 100)}%"

# ---------- simulated benchmark (part A)
simA = [("Approach 1 (rules only)", mean_group(sim, "A1_rules")),
        ("Approach 2, perfect labels", mean_group(sim, "A2_perfect_labels", ("09",))),
        ("Approach 2, damaged labels, no rules", mean_group(sim, "A2_damaged_raw", ("09",))),
        ("Approach 2, damaged labels + rules", mean_group(sim, "A2_damaged_reasoning", ("09",)))]
simB = [("Approach 1 (rules only)", mean_group(sim_nl, "A1_rules_nolayers")),
        ("Approach 2, perfect labels", mean_group(sim_nl, "A2_perfect_nolayers", ("09",))),
        ("Approach 2, damaged, no rules", mean_group(sim_nl, "A2_damaged_raw_nolayers", ("09",))),
        ("Approach 2, damaged + rules", mean_group(sim_nl, "A2_damaged_reasoning_nolayers", ("09",))),
        ("Approach 2, damaged + rules + approach-1 fallback", mean_group(sim_hy, "A2_damaged_reasoning_plus_A1_fallback_nolayers", ("09",)))]

# ---------- real GPU (part B)
def real_mean(v, skip=("09", "11")):
    rows = [d[v] for n, d in real.items() if v in d and not n.startswith(skip)]
    return {k: S.mean(r[k] for r in rows) for k in M} | {"n": len(rows)}
a1_scores = {os.path.basename(e["file"])[:-4]: e["approach1"]["scores"] for e in cmp_}
a1_mean = {k: S.mean(a1_scores[n][k] for n in a1_scores if not n.startswith(("09", "11"))) for k in M}
realrows = [("Approach 1 (rules only)", a1_mean),
            ("Model only (rules off)", real_mean("A2_real_raw")),
            ("Model + rules (as shipped)", real_mean("A2_real_rules")),
            ("Model + rules, no layer names", real_mean("A2_real_rules_nolayers")),
            ("Model + rules + fallback, no layer names", real_mean("A2_real_rules_fallback_nolayers")),
            ("Model + rules + fallback", real_mean("A2_real_rules_fallback"))]

def share(names, t, p):
    num = sum(lconf[n].get(f"{t}|{p}", 0) for n in names)
    den = sum(v for n in names for k, v in lconf[n].items() if k.startswith(t + "|"))
    return num / den if den else 0.0
syn = [n for n in lconf if n[:2].isdigit()]
APT, PER = ["apartment-building-plan"], ["SNOOBAR_545_Permit"]
lab = {
    "wall": [share(syn, "wall", "wall"), share(APT, "wall", "wall"), share(PER, "wall", "wall")],
    "door": [share(syn, "door", "door"), share(APT, "door", "door"), share(PER, "door", "door")],
    "window": [share(syn, "window", "window"), share(APT, "window", "window"), share(PER, "window", "window")],
    "bgwall": [share(syn, "background", "wall"), share(APT, "background", "wall"), share(PER, "background", "wall")],
}
simprof = [("Wall lines left unlabelled", "12.5%", share(syn, "wall", "none"), share(APT, "wall", "none"), share(PER, "wall", "none")),
           ("Window lines left unlabelled", "65%", share(syn, "window", "none"), share(APT, "window", "none"), share(PER, "window", "none")),
           ("Door lines called wall", "15%", share(syn, "door", "wall"), share(APT, "door", "wall"), share(PER, "door", "wall")),
           ("Furniture lines called wall", "25%", share(syn, "furniture", "wall"), share(APT, "furniture", "wall"), share(PER, "furniture", "wall")),
           ("Other lines called wall", "3%", share(syn, "background", "wall"), share(APT, "background", "wall"), share(PER, "background", "wall"))]

# per-drawing appendix
app_rows = []
for e in cmp_:
    n = os.path.basename(e["file"])[:-4]
    for a, lbl in (("approach1", "1 rules"), ("approach2", "2 SymPoint")):
        s = e[a]["scores"]
        app_rows.append(f"<tr><td>{n if a == 'approach1' else ''}</td><td>{lbl}</td>" + "".join(
            (f'<td class="{cls(s[k], k)}">{s[k]:.2f}</td>' if k != "manual_corrections_required" else f'<td class="{cls(s[k], k)}">{s[k]}</td>')
            for k in M) + "</tr>")

today = datetime.date.today().strftime("%d %B %Y")
H = open(f"{A}/report_template.html").read()
fill = {
    "DATE": today,
    "SIM_A": table([row(n, m) for n, m in simA]),
    "SIM_B": table([row(n, m) for n, m in simB]),
    "REAL_T": table([row(n, m) for n, m in realrows]),
    "LAB_BARS": "".join(bars(t, [("Synthetic (12)", v[0], C1), ("Real apartment", v[1], C2), ("Real permit set", v[2], C3)], fmt=pct)
                        for t, v in (("Walls labelled wall", lab["wall"]), ("Doors labelled door", lab["door"]),
                                     ("Windows labelled window / curtain wall", lab["window"]),
                                     ("Non-architectural lines wrongly called wall", lab["bgwall"]))),
    "RULE_BARS": "".join(bars(t, [("Model only", realrows[1][1][k], C1), ("+ rules", realrows[2][1][k], C2),
                                   ("+ rules, no layer names", realrows[3][1][k], C3), ("+ fallback, no layer names", realrows[4][1][k], C4)],
                              **({"maxv": 5.5, "fmt": lambda v: f"{v:.1f}"} if k == "manual_corrections_required" else {}))
                         for t, k in (("Room recall", "room_recall"), ("Wall recall", "wall_recall"), ("Window recall", "window_recall"),
                                      ("Fixes needed per drawing", "manual_corrections_required"))),
    "APT_BARS": "".join(bars(t, [("Approach 1", apt["A1"][k], C1), ("Approach 2", apt["rules"][k], C2), ("Approach 2 + fallback", apt["fallback"][k], C3)],
                             **({"maxv": 42, "fmt": lambda v: str(v)} if k == "rooms" else {}))
                        for t, k in (("Window recall", "window_R"), ("Door precision", "door_P"), ("Door recall", "door_R"), ("Rooms found", "rooms"))),
    "APT_T": "".join(f'<tr><td>{lbl}</td><td>{apt[k]["doors"]}</td><td>{apt[k]["door_P"]:.2f}</td><td>{apt[k]["door_R"]:.2f}</td>'
                     f'<td>{apt[k]["windows"]}</td><td class="{cls(apt[k]["window_R"], "x")}">{apt[k]["window_R"]:.2f}</td><td>{apt[k]["rooms"]}</td></tr>'
                     for k, lbl in (("A1", "Approach 1"), ("rules", "Approach 2 (as shipped)"), ("raw", "Approach 2, rules off"), ("fallback", "Approach 2 + fallback"))),
    "SIMPROF": "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c:.1%}</td><td>{d:.1%}</td><td>{e:.1%}</td></tr>" for a, b, c, d, e in simprof),
    "APPENDIX": "".join(app_rows),
    "IMG_APT_A1": img(f"{SP}/report/img/apt_a1.jpg"), "IMG_APT_A2": img(f"{SP}/report/img/apt_a2.jpg"),
    "IMG_APT_FB": img(f"{SP}/report/img/apt_fb.jpg"),
    "IMG_GF_A1": img(f"{SP}/report/img/gf_a1.jpg"), "IMG_GF_A2": img(f"{SP}/report/img/gf_a2.jpg"),
    "N_REAL": str(realrows[2][1]["n"]),
}
for k, v in fill.items():
    H = H.replace("{{" + k + "}}", v)
left = re.findall(r"\{\{[A-Z_0-9]+\}\}", H)
assert not left, left
open(OUT_HTML, "w").write(H)
print("html", OUT_HTML, len(H) // 1024, "KB")
