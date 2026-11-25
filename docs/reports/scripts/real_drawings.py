import json, os, sys, time
sys.path.insert(0, "/home/user/analysis"); sys.path.insert(0, "/home/user/latest")
from architectural_converter.pipeline.extraction_pipeline import ExtractionPipeline
from architectural_converter.pipeline.pipeline_config import PipelineConfig
from sympoint_converter import load_config
from sympoint_converter.pipeline import SymPointPipeline
GPU = "/home/user/gpu_in/gpu_out"; OUT = "/home/user/analysis/real_models"; os.makedirs(OUT, exist_ok=True)
name = sys.argv[1]; variants = sys.argv[2:]
path = f"/home/user/latest/samples/real/{name}.dxf"
resp = json.load(open(f"{GPU}/{name}.sympoint.response.json"))
import importlib.util
spec = importlib.util.spec_from_file_location("rv_helpers", "/home/user/analysis/rv_helpers.py"); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
for v in variants:
    t0 = time.perf_counter()
    if v == "A1":
        st = ExtractionPipeline(PipelineConfig()).process(path)
    elif v in ("rules", "raw"):
        cfg = load_config({"use_reasoning": v == "rules", "cache_labels": False})
        st = SymPointPipeline(cfg, response=resp, labels_progress=lambda m: None).process(path)
    elif v == "fallback":
        cfg = load_config({"cache_labels": False})
        a1 = json.load(open(f"{OUT}/{name}.A1.json")) if os.path.exists(f"{OUT}/{name}.A1.json") else ExtractionPipeline(PipelineConfig()).process(path).model
        ids = H.by_id(path, resp, cfg); r2, filled = H.remap(path, ids, cfg, H.a1_votes(a1)); print("filled", filled)
        st = SymPointPipeline(cfg, response=r2, labels_progress=lambda m: None).process(path)
    json.dump(st.model, open(f"{OUT}/{name}.{v}.json", "w"))
    m = st.model
    cnt = lambda k, f=lambda x: True: sum(1 for x in m[k] if x["status"] != "REJECTED" and f(x))
    print(name, v, f"{time.perf_counter()-t0:.0f}s", "walls", cnt("walls"), "rooms", cnt("rooms"), "doors", cnt("openings", lambda o: o["type"] == "door"),
          "windows", cnt("openings", lambda o: o["type"] == "window"), "other", cnt("openings", lambda o: o["type"] == "opening"), flush=True)
