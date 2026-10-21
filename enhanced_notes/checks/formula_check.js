// Paste into the browser console on http://localhost:3002/llm (dev build).
// Samples 20 cells in every computed block and checks: formula recomputed from inputs == stored value.
// Expected: bad = {} and skipped = {}.
(() => {
  const { state, explainCell } = window.__llmviz;
  const L = state.layout; const P = Object.getPrototypeOf(state.camera.center);
  let tested = 0, ok = 0, skipped = {}, bad = {};
  for (const b of L.cubes) {
    if (!b.deps) continue;
    for (let k = 0; k < 20; k++) {
      const idx = Object.assign(Object.create(P), { x: Math.floor(Math.random()*b.cx), y: Math.floor(Math.random()*b.cy), z: Math.floor(Math.random()*b.cz) });
      const e = explainCell(b, idx, L.tokEmbedObj, L.idxObj);
      if (e && e.masked) continue;
      const key = (b.name||'(unnamed)') + '|' + (e ? e.op : 'none');
      if (!e || e.recomputed === null || e.stored === null) { skipped[key] = (skipped[key]||0)+1; continue; }
      tested++;
      if (Math.abs(e.stored - e.recomputed) <= 1e-3*Math.max(1,Math.abs(e.stored))) ok++; else bad[key] = bad[key] || [idx.x, idx.y, e.stored, e.recomputed];
    }
  }
  return { tested, ok, skipped, bad };
})()
