// Run: npx tsx tools/check_presets.ts — every preset's parameter count must be within 1% of the published one.
import { archPresets, countParams } from "../src/llm/arch/ArchPresets";
let fail = 0;
for (let p of archPresets) {
    let n = countParams(p.config);
    let err = Math.abs(n.total - p.published.total) / p.published.total;
    let errA = p.published.active ? Math.abs(n.active - p.published.active) / p.published.active : 0;
    let ok = err < 0.01 && errA < 0.01;
    if (!ok) fail++;
    console.log(`${ok ? 'OK  ' : 'FAIL'} ${p.title.padEnd(14)} total ${(n.total / 1e9).toFixed(3)}B vs ${(p.published.total / 1e9).toFixed(3)}B` +
        (p.published.active ? `, active ${(n.active / 1e9).toFixed(2)}B vs ${(p.published.active / 1e9).toFixed(2)}B` : ''));
}
process.exit(fail ? 1 : 0);
