import json, sys, statistics as S, re
r = json.load(open(sys.argv[1])); skip = set(sys.argv[2:])
M = ["wall_precision", "wall_recall", "room_precision", "room_recall", "door_precision", "door_recall", "window_precision", "window_recall", "column_recall"]
groups = {}
for n, d in r.items():
    for k, m in d.items():
        if "error" in m: print("ERR", n, k, m["error"]); continue
        g = re.sub(r"_s\d+$", "", k)
        if n in skip and g.startswith("A2"): continue
        groups.setdefault(g, {}).setdefault(n, []).append(m)
print("config".ljust(34), " ".join(x.replace("_precision", "_P").replace("_recall", "_R").rjust(8) for x in M), "  fixes/dwg")
for g, per in groups.items():
    vals = {x: S.mean(S.mean(mm[x] for mm in ms) for ms in per.values()) for x in M}
    fx = S.mean(S.mean(mm["manual_corrections_required"] for mm in ms) for ms in per.values())
    print(g.ljust(34), " ".join(f"{vals[x]:8.3f}" for x in M), f"  {fx:6.1f}   (n={len(per)})")
print()
for g in groups:
    if "damaged" in g:
        worst = sorted(((S.mean(mm["room_recall"] for mm in ms), n) for n, ms in groups[g].items()))[:4]
        print(g, "worst room recall:", [(n[:16], round(v, 2)) for v, n in worst])
