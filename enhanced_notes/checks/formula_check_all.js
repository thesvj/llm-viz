// Paste into the browser console on http://localhost:3002/llm (dev build), or run via a browser tool.
// For every model (gpt2, llama, moe): checks EVERY cell of every computed block —
// the formula recomputed from its inputs must equal the stored value.
// For modern models it also compares the shown next-token probabilities with the forward pass.
// Expected: bad = {} and skipped = {} for each model, probCheck = 0.
(async () => {
  const { state, explainCell } = window.__llmviz;
  const P = Object.getPrototypeOf(state.camera.center);
  const prevMode = state.modelMode;
  const res = {};
  for (const mode of ['gpt2', 'llama', 'moe']) {
    state.modelMode = mode; state.markDirty();
    // the layout is rebuilt on the next frame; wait until it has switched
    for (let i = 0; i < 50 && (state.layout.kind === 'gpt2') !== (mode === 'gpt2'); i++) await new Promise(r => setTimeout(r, 100));
    await new Promise(r => setTimeout(r, 300));
    const L = state.layout;
    let tested = 0, ok = 0, masked = 0, skipped = {}, bad = {}, noSrc = [];
    for (const b of L.cubes) {
      if (b.access && !b.access.src) noSrc.push(b.name);
      if (!b.deps && !(b.meta && b.meta.kind === 'lens')) continue;
      for (let x = 0; x < b.cx; x++) for (let y = 0; y < b.cy; y++) for (let z = 0; z < b.cz; z++) {
        const e = explainCell(b, Object.assign(Object.create(P), { x, y, z }), L.tokEmbedObj, L.idxObj);
        if (e && e.masked) { masked++; continue; }
        const key = (b.name || '(unnamed)') + '|' + (e ? e.op : 'none');
        if (!e || e.recomputed === null || e.stored === null) { skipped[key] = (skipped[key] || 0) + 1; continue; }
        tested++;
        if (Math.abs(e.stored - e.recomputed) <= 1e-3 * Math.max(1, Math.abs(e.stored))) ok++;
        else bad[key] = bad[key] || [x, y, e.stored, e.recomputed];
      }
    }
    let probCheck = null;
    if (L.kind === 'modern') {
      const p = state.modern[mode].rec.probs, V = p.shape[p.shape.length - 1], b = L.logitsSoftmax;
      probCheck = 0;
      for (let x = 0; x < b.cx; x++) for (let y = 0; y < b.cy; y++) {
        const e = explainCell(b, Object.assign(Object.create(P), { x, y, z: 0 }), L.tokEmbedObj, L.idxObj);
        probCheck = Math.max(probCheck, Math.abs(e.stored - p.data[x * V + y]));
      }
    }
    res[mode] = { kind: L.kind, cubes: L.cubes.length, tested, ok, masked, skipped, bad, noSrc, probCheck };
  }
  state.modelMode = prevMode; state.markDirty();
  return JSON.stringify(res);
})()
