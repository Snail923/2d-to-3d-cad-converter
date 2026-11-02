// Automatic verification of the engine's doubtful decisions by a LOCAL vision model.
//
// The engine (Python) selects the items and phrases each one as a closed question with the
// possible answers (model.verification.queue). For every item this module draws a crop of the
// original DXF linework around it with the element highlighted, sends the picture + the
// question + the CAD context to a model running on this computer (LM Studio by default; also
// Ollama or any OpenAI-compatible server such as llama.cpp / vLLM) and turns confident answers
// into corrections. The app applies them and re-runs automatically. The model can only choose
// between the engine's own interpretations; it never creates or moves geometry.

export const DEFAULTS = {
  enabled: false, api: "lmstudio", endpoint: "http://localhost:1234", model: "",
  minConfidence: 0.7, maxItems: 40, rounds: 2, timeoutS: 180,
};
const KEY = "cad2d3d.verifier";

export function loadSettings() {
  try { return Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem(KEY) || "{}")); }
  catch { return Object.assign({}, DEFAULTS); }
}
export function saveSettings(s) { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* private mode */ } }

// ------------------------------------------------------------------ crop rendering
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function cropSvg(view, item, size = 768) {
  const [x0, y0, x1, y1] = item.bbox;
  const w = x1 - x0, h = y1 - y0;
  const X = (x) => (x - x0).toFixed(3), Y = (y) => (y1 - y).toFixed(3);   // y up -> SVG y down
  const px = w / size;                                                      // metres per pixel
  const inside = (p) => p.some(([x, y]) => x >= x0 && x <= x1 && y >= y0 && y <= y1) ||
    (Math.min(...p.map((q) => q[0])) <= x1 && Math.max(...p.map((q) => q[0])) >= x0 &&
     Math.min(...p.map((q) => q[1])) <= y1 && Math.max(...p.map((q) => q[1])) >= y0);
  const parts = [];
  for (const ln of view.lines) {
    if (!inside(ln.p)) continue;
    parts.push(`<path d="M${ln.p.map(([x, y]) => X(x) + " " + Y(y)).join(" L")}"/>`);
  }
  const texts = [];
  for (const t of view.texts || []) {
    const [x, y] = t.p;
    if (x < x0 || x > x1 || y < y0 || y > y1 || !t.t) continue;
    const fs = Math.max(t.h || 0, w / 45);
    texts.push(`<text x="${X(x)}" y="${Y(y)}" font-size="${fs.toFixed(3)}">${esc(t.t.split("\n")[0].slice(0, 40))}</text>`);
  }
  // highlight: a red frame AROUND the element (never painted over the linework the model must read)
  const hl = [];
  const pad = Math.max(w / 60, 0.08), sw = (3 * px).toFixed(4);
  const frame = (pts) => {
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const ax = Math.min(...xs) - pad, bx = Math.max(...xs) + pad, ay = Math.min(...ys) - pad, by = Math.max(...ys) + pad;
    hl.push(`<rect x="${X(ax)}" y="${Y(by)}" width="${(bx - ax).toFixed(3)}" height="${(by - ay).toFixed(3)}" fill="rgb(255,0,0)" fill-opacity="0.07" stroke="rgb(255,0,0)" stroke-width="${sw}" stroke-dasharray="${(8 * px).toFixed(3)} ${(4 * px).toFixed(3)}"/>`);
  };
  if (item.highlight.polygon) frame(item.highlight.polygon);
  else if (item.highlight.segment) frame(item.highlight.segment);
  // 1 m scale bar
  const sb = `<g stroke="#0055cc" stroke-width="${(3 * px).toFixed(4)}"><path d="M${(w * 0.04).toFixed(3)} ${(h * 0.96).toFixed(3)} h1"/></g>
    <text x="${(w * 0.04).toFixed(3)}" y="${(h * 0.96 - 6 * px).toFixed(3)}" font-size="${(w / 40).toFixed(3)}" fill="#0055cc">1 m</text>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${Math.round(size * h / w)}" viewBox="0 0 ${w.toFixed(3)} ${h.toFixed(3)}">
    <rect x="0" y="0" width="${w}" height="${h}" fill="#fff"/>
    <g fill="none" stroke="#000" stroke-width="${(1.6 * px).toFixed(4)}" stroke-linecap="round">${parts.join("")}</g>
    <g fill="#333" font-family="Arial, sans-serif">${texts.join("")}</g>${hl.join("")}${sb}</svg>`;
}

export function rasterize(svg) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.width; c.height = img.height;
      c.getContext("2d").drawImage(img, 0, 0);
      resolve(c.toDataURL("image/png").split(",")[1]);
    };
    img.onerror = () => reject(new Error("crop rendering failed"));
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  });
}

// ------------------------------------------------------------------ model call
export function buildPrompt(item) {
  const c = item.context || {};
  const lines = [
    "You are an experienced architect checking an automatic reading of a 2D architectural CAD floor plan.",
    "The image is a crop of the plan: black lines are the original drawing, a 1 m scale bar is at the bottom left.",
    "The element in question is highlighted in translucent RED.",
    "",
    "Question: " + item.question,
    "",
    "Context read from the CAD file:",
    `- rooms on either side: ${(c.rooms_on_sides || []).join(" | ") || "unknown"}`,
    `- CAD layers: ${(c.layers || []).join(", ") || "none"}; block names: ${(c.blocks || []).join(", ") || "none"}`,
    c.width_m !== undefined ? `- width: ${c.width_m} m` : "",
    `- engine confidence: ${c.engine_confidence}; engine notes: ${(c.notes || []).join(" / ") || "none"}`,
    "",
    "Architectural logic to apply:",
    "- every room must be reachable through a door or an open passage; a window does not give access;",
    "- windows sit in walls that face the outside (or a light well); a window between two rooms is unusual;",
    "- wardrobes / closets are closed by sliding or hinged doors, often drawn as thin parallel lines or zig-zags;",
    "- a hinged door is usually drawn with a leaf line and a swing arc;",
    "- beds, wardrobes, desks, kitchen counters, shelves and other furniture are NOT walls, doors or windows;",
    "- answer \"unsure\" if the picture does not show enough.",
    "",
    `Reply with ONLY a JSON object: {"answer": one of ${JSON.stringify(item.answers)}, "confidence": number between 0 and 1, "reason": "one short sentence"}`,
  ];
  return lines.filter((l) => l !== "").join("\n");
}

function parseAnswer(text, item) {
  const m = /\{[\s\S]*\}/.exec(text || "");
  if (!m) throw new Error("no JSON in model reply: " + String(text).slice(0, 120));
  const j = JSON.parse(m[0]);
  let a = String(j.answer || j.verdict || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (!item.answers.includes(a)) a = item.answers.find((x) => a.includes(x)) || "unsure";
  const conf = Math.max(0, Math.min(1, Number(j.confidence)));
  return { answer: a, confidence: isFinite(conf) ? conf : 0, reason: String(j.reason || "").slice(0, 240) };
}

const v1 = (base) => base + (base.endsWith("/v1") ? "" : "/v1");

// Models offered by the local server. LM Studio's REST API also says which models are vision
// models ("vlm") and which are loaded; other servers only list ids.
export async function listModels(settings) {
  const base = settings.endpoint.replace(/\/+$/, "").replace(/\/v1$/, "");
  const get = async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url} answered ${r.status}`); return r.json(); };
  if (settings.api === "ollama") {
    const j = await get(base + "/api/tags");
    return (j.models || []).map((m) => ({ id: m.name, vision: null, loaded: null }));
  }
  if (settings.api === "lmstudio") {
    try {
      const j = await get(base + "/api/v0/models");
      return (j.data || []).filter((m) => m.type !== "embeddings")
        .map((m) => ({ id: m.id, vision: m.type === "vlm", loaded: m.state === "loaded" }));
    } catch { /* older LM Studio: fall back to the OpenAI listing */ }
  }
  const j = await get(v1(base) + "/models");
  return (j.data || []).map((m) => ({ id: m.id, vision: null, loaded: null }));
}

// Best default: a loaded vision model, then any vision model, then the first model.
export function pickModel(models) {
  return (models.find((m) => m.vision && m.loaded) || models.find((m) => m.vision) || models.find((m) => m.loaded) || models[0] || {}).id || "";
}

export async function askModel(settings, prompt, imageB64, answers = null) {
  if (!settings.model) throw new Error("no model selected (press “Find models”)");
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), settings.timeoutS * 1000);
  const base = settings.endpoint.replace(/\/+$/, "");
  try {
    let url, body, pick;
    if (settings.api === "ollama") {
      url = base + "/api/chat";
      body = { model: settings.model, stream: false, format: "json", options: { temperature: 0 },
        messages: [{ role: "user", content: prompt, images: imageB64 ? [imageB64] : [] }] };
      pick = (j) => j.message && j.message.content;
    } else {
      url = v1(base) + "/chat/completions";
      const content = [{ type: "text", text: prompt }];
      if (imageB64) content.push({ type: "image_url", image_url: { url: "data:image/png;base64," + imageB64 } });
      body = { model: settings.model, temperature: 0, max_tokens: 300, messages: [{ role: "user", content }] };
      if (settings.api === "lmstudio" && answers) {
        // structured output: the model can only produce one of the allowed answers
        body.response_format = { type: "json_schema", json_schema: { name: "verdict", strict: true, schema: {
          type: "object", additionalProperties: false, required: ["answer", "confidence", "reason"],
          properties: { answer: { type: "string", enum: answers }, confidence: { type: "number", minimum: 0, maximum: 1 },
            reason: { type: "string" } } } } };
      }
      pick = (j) => j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content;
    }
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctl.signal });
    if (!r.ok) throw new Error(`model server answered ${r.status}: ${(await r.text()).slice(0, 200)}`);
    return pick(await r.json());
  } finally { clearTimeout(timer); }
}

export async function testConnection(settings) {
  const probe = { answers: ["yes", "no", "unsure"], question: "Is this a connection test?", context: {} };
  const txt = await askModel(settings, 'Reply with ONLY {"answer": "yes", "confidence": 1, "reason": "test"}', null, probe.answers);
  return parseAnswer(txt, probe);
}

// ------------------------------------------------------------------ verification round
// Returns one decision per item: {id, kind, claim, answer, confidence, reason, applied, error?}
export async function verifyQueue(queue, view, settings, onProgress, isCancelled) {
  const out = [];
  const items = queue.slice(0, settings.maxItems);
  for (let i = 0; i < items.length; i++) {
    if (isCancelled && isCancelled()) break;
    const it = items[i];
    onProgress && onProgress(i, items.length, it);
    try {
      const img = await rasterize(cropSvg(view, it));
      const res = parseAnswer(await askModel(settings, buildPrompt(it), img, it.answers), it);
      out.push({ id: it.id, kind: it.kind, claim: it.claim, ...res,
        applied: res.answer !== "unsure" && res.confidence >= settings.minConfidence });
    } catch (e) {
      out.push({ id: it.id, kind: it.kind, claim: it.claim, answer: "unsure", confidence: 0, reason: "", applied: false, error: e.message });
      if (/Failed to fetch|NetworkError|abort/i.test(e.message) && i === 0) break;   // server not reachable: stop early
    }
  }
  return out;
}

// Turn applied decisions into corrections (same format as manual corrections + provenance).
export function applyDecisions(corr, decisions, answers, settings) {
  let n = 0;
  corr.provenance = corr.provenance || {};
  for (const d of decisions) {
    if (!d.applied) continue;
    const act = (answers[d.kind] || {})[d.answer];
    if (!act || !Object.keys(act).length) continue;
    if (act.type) corr.opening_type_overrides[d.id] = act.type;
    if (act.status) corr.status_overrides[d.id] = act.status;
    corr.provenance[d.id] = { by: "ai", model: settings.model, verdict: d.answer, engine_claim: d.claim,
      confidence: +d.confidence.toFixed(2), reason: d.reason };
    n++;
  }
  return n;
}

export function removeAiDecisions(corr) {
  const prov = corr.provenance || {};
  for (const [id, p] of Object.entries(prov)) {
    if (p.by !== "ai") continue;
    delete corr.status_overrides[id];
    delete corr.opening_type_overrides[id];
    delete prov[id];
  }
}
