/* The last session, kept in the browser (IndexedDB) so a page refresh does not lose the work: the drawing,
   both methods' results, the corrections, the cached SymPointV2 answers, the 3D editor's scene and the room-check
   results. Saved a moment after every change; restored on request. Everything stays on this computer. */

const DB = "cad2d3d", STORE = "session", KEY = "last";

function open() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function tx(mode, fn) {
  const db = await open();
  try {
    return await new Promise((resolve, reject) => {
      const t = db.transaction(STORE, mode), req = fn(t.objectStore(STORE));
      t.oncomplete = () => resolve(req && req.result);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error || new Error("storage aborted (quota?)"));
    });
  } finally { db.close(); }
}
export async function saveSession(s) { return tx("readwrite", (st) => st.put({ ...s, savedAt: Date.now() }, KEY)); }
export async function loadSession() { try { return await tx("readonly", (st) => st.get(KEY)); } catch { return null; } }
export async function clearSession() { try { await tx("readwrite", (st) => st.delete(KEY)); } catch { /* nothing saved */ } }

// a save a moment after the last change (changes come in bursts: an edit, a re-run, a room check)
export function debounced(fn, ms = 1500) {
  let t = null;
  return () => { clearTimeout(t); t = setTimeout(() => { t = null; fn(); }, ms); };
}
