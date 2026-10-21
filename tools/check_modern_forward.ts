// Parity test: JS forward pass vs the PyTorch export, every intermediate tensor.
// Run: node tools/check_modern_forward.ts            (Node 24 strips the TypeScript types itself)
import { readFileSync } from 'node:fs';
import { decodeTensorJson, modernForward } from '../src/llm/arch/ModernForward.ts';

let failed = false;
for (const variant of ['llama', 'moe']) {
    const modelJson = JSON.parse(readFileSync(`public/tiny-${variant}-model.json`, 'utf8'));
    const checkJson = JSON.parse(readFileSync(`public/tiny-${variant}-check.json`, 'utf8'));
    const model = { config: modelJson.config, w: decodeTensorJson(modelJson) };
    const ref = decodeTensorJson(checkJson.tensors);
    const got = modernForward(model, checkJson.tokens);
    let worst = 0, worstName = '', missing: string[] = [], extra: string[] = [];
    for (const [name, r] of Object.entries(ref)) {
        const g = got[name];
        if (!g) { missing.push(name); continue; }
        if (g.shape.join() !== r.shape.join()) { console.log(`  shape mismatch ${name}: js ${g.shape} vs py ${r.shape}`); failed = true; continue; }
        for (let i = 0; i < r.data.length; i++) {
            const d = Math.abs(g.data[i] - r.data[i]) / Math.max(1, Math.abs(r.data[i]));
            if (d > worst) { worst = d; worstName = name; }
        }
    }
    for (const name of Object.keys(got)) if (!ref[name]) extra.push(name);
    const ok = worst <= 1e-4 && missing.length === 0 && extra.length === 0;
    failed ||= !ok;
    console.log(`${variant}: ${Object.keys(ref).length} tensors, worst error ${worst.toExponential(2)} (${worstName})` +
        `${missing.length ? `, MISSING in JS: ${missing.join(' ')}` : ''}${extra.length ? `, EXTRA in JS: ${extra.join(' ')}` : ''}  ${ok ? 'PASS' : 'FAIL'}`);
}
process.exit(failed ? 1 : 0);
