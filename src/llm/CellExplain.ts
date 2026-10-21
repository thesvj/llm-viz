import { BlKDepSpecial, IBlkCellDep, IBlkDef } from "./GptModelLayout";
import { getBlockValueAtIdx } from "./components/DataFlow";
import { Vec3, Vec4 } from "@/src/utils/vector";

/** One input that goes into a cell, e.g. one product term of a dot product. */
export interface ICellTerm {
    label: string;
    values: (number | null)[];
}

/** Plain explanation of how one cell's value is computed, recomputed from its real inputs. */
export interface ICellExplain {
    op: string;             // short operation name, e.g. "Dot product + bias"
    formula: string;        // formula in symbols, e.g. "Σᵢ W[y,i]·x[i,t] + b[y]"
    words: string;          // one plain sentence
    terms: ICellTerm[];     // the first few inputs (for display)
    termCount: number;      // how many inputs exist in total
    termHeader: string[];   // column names for terms
    stored: number | null;  // value held by the model
    recomputed: number | null; // value recomputed here from the inputs (null if not supported)
    masked?: boolean;
    legend?: string;        // what the term columns mean
}

const maxTerms = 8;

function srcIdxFor(dep: IBlkCellDep, dest: Vec3, i: number): Vec3 {
    let v = dep.srcIdxMtx.mulVec4(new Vec4(dest.x, dest.y, dest.z, i));
    return new Vec3(v.x, v.y, v.z);
}

/** Size of the summed-over dimension for a dep that uses `i`, or 0 if the dep does not use `i`. */
function iLength(dep: IBlkCellDep, dest: Vec3): number {
    let a = srcIdxFor(dep, dest, 0);
    let b = srcIdxFor(dep, dest, 1);
    if (a.x !== b.x) return dep.src.cx;
    if (a.y !== b.y) return dep.src.cy;
    if (a.z !== b.z) return dep.src.cz;
    return 0;
}

function val(blk: IBlkDef, idx: Vec3) {
    return getBlockValueAtIdx(blk, idx);
}

function name(blk: IBlkDef) {
    return blk.name || 'input';
}

function gelu(x: number) {
    return x * 0.5 * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
}

export function explainCell(blk: IBlkDef, idx: Vec3, tokEmbedObj?: IBlkDef, idxObj?: IBlkDef): ICellExplain | null {
    let deps = blk.deps;
    let stored = val(blk, idx);
    let base = { stored, terms: [] as ICellTerm[], termCount: 0, termHeader: [] as string[] };

    if (blk.meta?.kind === 'concat') {
        let m = blk.meta;
        let part = Math.floor(idx.y / m.partSize), within = idx.y % m.partSize;
        let v = m.parts[part] ? val(m.parts[part], new Vec3(idx.x, within, idx.z)) : null;
        return {
            ...base,
            op: 'Concatenate heads',
            formula: `head ${part} output [t=${idx.x}, channel ${within}]`,
            words: `The head outputs are stacked one after another. Channel ${idx.y} is channel ${within} of head ${part}.`,
            termHeader: ['part', 'value'],
            terms: [{ label: `head ${part}`, values: [v] }],
            termCount: 1,
            recomputed: v,
        };
    }

    if (blk.meta?.kind === 'lens') {
        let m = blk.meta;
        let t = idx.x, C = m.src.cy, V = blk.cy;
        let xs = Array.from({ length: C }, (_, c) => val(m.src, new Vec3(t, c, 0)));
        let gs = Array.from({ length: C }, (_, c) => val(m.gamma, new Vec3(0, c, 0)));
        let ok = xs.every(v => v !== null) && gs.every(v => v !== null);
        let logits: number[] = [];
        if (ok) {
            let rms = Math.sqrt((xs as number[]).reduce((a, x) => a + x * x, 0) / C + m.eps);
            for (let v = 0; v < V; v++) {
                let s = 0;
                for (let c = 0; c < C; c++) s += xs[c]! / rms * gs[c]! * (val(m.lmHead, new Vec3(c, v, 0)) ?? NaN);
                logits.push(s);
            }
        }
        let mx = Math.max(...logits);
        let sum = logits.reduce((a, l) => a + Math.exp(l - mx), 0);
        let probs = logits.map(l => Math.exp(l - mx) / sum);
        return {
            ...base,
            op: 'Logit lens',
            formula: 'softmax( LM head · RMS norm(x) )',
            words: `A probe, not part of the model: apply the final RMS norm and the LM head to token ${t}'s numbers at this point. It shows what the model would guess if it stopped here.`,
            termHeader: ['token', 'logit', 'prob'],
            terms: logits.map((l, v) => ({ label: `${String.fromCharCode(65 + v)}${v === idx.y ? ' ◀' : ''}`, values: [l, probs[v]] })),
            termCount: V,
            recomputed: ok && Number.isFinite(probs[idx.y]) ? probs[idx.y] : null,
        };
    }

    if (!deps) {
        let isWeight = blk.t === 'w';
        return {
            ...base,
            op: isWeight ? 'Learned weight' : 'Input',
            formula: isWeight ? 'fixed number learned in training' : 'given',
            words: isWeight ? 'This number was learned during training. It does not depend on the input text.' : 'This value is an input to the model.',
            recomputed: null,
        };
    }

    let special = deps.special;

    if (deps.lowerTri && idx.x > idx.y) {
        return {
            ...base, masked: true,
            op: 'Causal mask',
            formula: special === BlKDepSpecial.Softmax ? 'p = 0' : 'score = −∞',
            words: `Token ${idx.y} may not look at a later token (${idx.x}), so this cell is blocked.`,
            recomputed: null,
        };
    }

    let add = deps.add ?? [];
    let modern = explainModern(blk, idx, base);
    if (modern) {
        return modern;
    }
    let bySpecial = (s: BlKDepSpecial) => add.find(d => d.src.deps?.special === s);

    // ---- input embedding: token row + position row ----
    if (special === BlKDepSpecial.InputEmbed && tokEmbedObj && idxObj) {
        let tok = val(idxObj, new Vec3(idx.x, 0, idx.z));
        let tokDep = add.find(d => d.src === tokEmbedObj);
        let posDep = add.find(d => d.src !== tokEmbedObj && d.src !== idxObj);
        let a = tokDep && tok !== null ? val(tokDep.src, new Vec3(tok, idx.y, 0)) : null;
        let b = posDep ? val(posDep.src, srcIdxFor(posDep, idx, 0)) : null;
        return {
            ...base,
            op: 'Embedding lookup',
            formula: posDep ? `tokEmbed[token=${tok}, c=${idx.y}] + posEmbed[t=${idx.x}, c=${idx.y}]` : `tokEmbed[token=${tok}, c=${idx.y}]`,
            words: posDep ? `Look up row ${tok} of the token table and row ${idx.x} of the position table, then add them.`
                : `Look up row ${tok} of the token table. No position is added here: RoPE adds position later, inside attention.`,
            termHeader: ['part', 'value'],
            terms: posDep ? [{ label: 'token part', values: [a] }, { label: 'position part', values: [b] }] : [{ label: 'token row', values: [a] }],
            termCount: posDep ? 2 : 1,
            recomputed: a !== null && (b !== null || !posDep) ? a + (b ?? 0) : null,
        };
    }

    // ---- layer norm statistics ----
    if (special === BlKDepSpecial.LayerNormMu || special === BlKDepSpecial.LayerNormSigma) {
        let dep = add[0];
        let n = iLength(dep, idx);
        let xs: number[] = [];
        for (let i = 0; i < n; i++) {
            let v = val(dep.src, srcIdxFor(dep, idx, i));
            if (v !== null) xs.push(v);
        }
        let mean = xs.reduce((a, b) => a + b, 0) / xs.length;
        let variance = xs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / xs.length;
        let isMu = special === BlKDepSpecial.LayerNormMu;
        return {
            ...base,
            op: isMu ? 'Mean (μ)' : 'Standard deviation (σ)',
            formula: isMu ? `μ = (1/${n}) Σᵢ x[i]` : `σ = √( (1/${n}) Σᵢ (x[i] − μ)² + ε )`,
            words: isMu ? `Average of the ${n} numbers in this token's column.` : `How spread out this token's ${n} numbers are around their mean.`,
            termHeader: ['i', 'x[i]'],
            terms: xs.slice(0, maxTerms).map((x, i) => ({ label: `${i}`, values: [x] })),
            termCount: n,
            recomputed: isMu ? mean : Math.sqrt(variance + 1e-5),
        };
    }

    // ---- layer norm ----
    if (special === BlKDepSpecial.LayerNorm) {
        let [xDep, muDep, sigDep, gDep, bDep] = add;
        let x = val(xDep.src, srcIdxFor(xDep, idx, 0));
        let mu = val(muDep.src, srcIdxFor(muDep, idx, 0));
        let sig = val(sigDep.src, srcIdxFor(sigDep, idx, 0));
        let g = val(gDep.src, srcIdxFor(gDep, idx, 0));
        let b = val(bDep.src, srcIdxFor(bDep, idx, 0));
        let ok = [x, mu, sig, g, b].every(v => v !== null);
        return {
            ...base,
            op: 'Layer norm',
            formula: '(x − μ) / σ · γ + β',
            words: 'Centre the number on its column mean, rescale by the spread, then apply the learned scale γ and shift β.',
            termHeader: ['part', 'value'],
            terms: [['x', x], ['μ', mu], ['σ', sig], ['γ', g], ['β', b]].map(([l, v]) => ({ label: l as string, values: [v as number | null] })),
            termCount: 5,
            recomputed: ok ? (x! - mu!) / sig! * g! + b! : null,
        };
    }

    // ---- softmax pieces ----
    if (special === BlKDepSpecial.SoftmaxAggMax || special === BlKDepSpecial.SoftmaxAggExp) {
        let dep = add.find(d => iLength(d, idx) > 0)!;
        let n = iLength(dep, idx);
        let xs: number[] = [];
        for (let i = 0; i < n; i++) {
            let v = val(dep.src, srcIdxFor(dep, idx, i));
            if (v !== null) xs.push(v);
        }
        if (dep.src.deps?.lowerTri) {
            let row = srcIdxFor(dep, idx, 0);
            let limit = (row.y ?? 0) + 1;
            xs = xs.slice(0, Math.max(1, limit));
        }
        let m = Math.max(...xs);
        let isMax = special === BlKDepSpecial.SoftmaxAggMax;
        let s = xs.reduce((a, x) => a + Math.exp(x - m), 0);
        return {
            ...base,
            op: isMax ? 'Row maximum' : 'Inverse sum of exponentials',
            formula: isMax ? 'm = maxᵢ x[i]' : '1 / Σᵢ exp(x[i] − m)',
            words: isMax ? 'The largest score in this row. Subtracting it keeps exp() from overflowing.' : 'One over the total of all exp(score − max) in this row; multiplying by it makes the row sum to 1.',
            termHeader: ['i', 'x[i]'],
            terms: xs.slice(0, maxTerms).map((x, i) => ({ label: `${i}`, values: [x] })),
            termCount: xs.length,
            recomputed: isMax ? m : 1 / s,
        };
    }

    if (special === BlKDepSpecial.Softmax) {
        let xDep = add[0];
        let expDep = bySpecial(BlKDepSpecial.SoftmaxAggExp);
        let maxDep = bySpecial(BlKDepSpecial.SoftmaxAggMax);
        let x = val(xDep.src, srcIdxFor(xDep, idx, 0));
        let inv = expDep ? val(expDep.src, srcIdxFor(expDep, idx, 0)) : null;
        let m = maxDep ? val(maxDep.src, srcIdxFor(maxDep, idx, 0)) : null;
        let ok = x !== null && inv !== null && m !== null;
        return {
            ...base,
            op: 'Softmax',
            formula: 'exp(x − m) · (1 / Σ exp)',
            words: 'Turn the score into a probability: bigger scores get more weight, and each row adds up to 1.',
            termHeader: ['part', 'value'],
            terms: [{ label: 'x', values: [x] }, { label: 'm (row max)', values: [m] }, { label: '1/Σ exp', values: [inv] }],
            termCount: 3,
            recomputed: ok ? Math.exp(x! - m!) * inv! : null,
        };
    }

    if (special === BlKDepSpecial.Gelu) {
        let dep = add[0];
        let x = val(dep.src, srcIdxFor(dep, idx, 0));
        return {
            ...base,
            op: 'GELU activation',
            formula: 'x · ½ · (1 + tanh(√(2/π) · (x + 0.044715·x³)))',
            words: 'A smooth on/off switch: large positive x passes through, negative x is squashed towards 0.',
            termHeader: ['part', 'value'],
            terms: [{ label: 'x', values: [x] }],
            termCount: 1,
            recomputed: x !== null ? gelu(x) : null,
        };
    }

    // ---- dot product (matrix multiply), optional bias ----
    if (deps.dot) {
        let [aDep, bDep] = deps.dot;
        let n = Math.min(iLength(aDep, idx) || Infinity, iLength(bDep, idx) || Infinity);
        if (!isFinite(n)) n = deps.dotLen ?? 0;
        // masked attention weights are stored as 0, so summing over every token is exact
        let sum = 0;
        let ok = true;
        let terms: ICellTerm[] = [];
        for (let i = 0; i < n; i++) {
            let a = val(aDep.src, srcIdxFor(aDep, idx, i));
            let b = val(bDep.src, srcIdxFor(bDep, idx, i));
            if (a === null || b === null) { ok = false; continue; }
            sum += a * b;
            if (terms.length < maxTerms) terms.push({ label: `${i}`, values: [a, b, a * b] });
        }
        let isAttn = special === BlKDepSpecial.Attention;
        let scale = 1;
        let biasDep = add.find(d => iLength(d, idx) === 0);
        let bias = biasDep ? val(biasDep.src, srcIdxFor(biasDep, idx, 0)) : null;
        if (isAttn) {
            // attention scores are scaled by 1/√(head size); n here is the head size
            scale = 1 / Math.sqrt(n);
        }
        let result = sum * scale + (bias ?? 0);
        let fa = name(aDep.src), fb = name(bDep.src);
        return {
            ...base,
            op: isAttn ? 'Attention score (scaled dot product)' : biasDep ? 'Dot product + bias' : 'Dot product',
            formula: isAttn ? `(Σᵢ q[i]·k[i]) / √${n}` : `Σᵢ ${fa}[i] · ${fb}[i]${biasDep ? ` + ${name(biasDep.src)}` : ''}`,
            words: isAttn
                ? `How well token ${idx.y}'s query matches token ${idx.x}'s key: multiply the ${n} pairs, add them up, divide by √${n}.`
                : `Multiply ${n} pairs of numbers from "${fa}" and "${fb}" and add them up${biasDep ? ', then add the bias' : ''}.`,
            termHeader: ['i', 'a', 'b', 'a·b'],
            legend: `a = ${fa}, b = ${fb}`,
            terms: biasDep ? [...terms, { label: 'bias', values: [null, null, bias] }] : terms,
            termCount: n,
            recomputed: ok ? result : null,
        };
    }

    // ---- plain addition (residual connection) ----
    if (add.length === 2) {
        let [aDep, bDep] = add;
        let a = val(aDep.src, srcIdxFor(aDep, idx, 0));
        let b = val(bDep.src, srcIdxFor(bDep, idx, 0));
        return {
            ...base,
            op: 'Residual add',
            formula: `${name(aDep.src)} + ${name(bDep.src)}`,
            words: 'Add the new result onto the running "residual stream", so earlier information is kept.',
            termHeader: ['part', 'value'],
            terms: [{ label: name(aDep.src), values: [a] }, { label: name(bDep.src), values: [b] }],
            termCount: 2,
            recomputed: a !== null && b !== null ? a + b : null,
        };
    }

    return { ...base, op: 'Computed', formula: '', words: '', recomputed: null };
}

function silu(x: number) {
    return x / (1 + Math.exp(-x));
}

type IBase = Pick<ICellExplain, 'stored' | 'terms' | 'termCount' | 'termHeader'>;

/** Operations of modern architectures (RMSNorm, RoPE, SwiGLU, MoE). Returns null for other blocks. */
function explainModern(blk: IBlkDef, idx: Vec3, base: IBase): ICellExplain | null {
    let deps = blk.deps!;
    let add = deps.add ?? [];
    let special = deps.special;
    let meta = blk.meta;

    if (special === BlKDepSpecial.RMSNormAgg) {
        let dep = add[0];
        let n = iLength(dep, idx);
        let xs: number[] = [];
        for (let i = 0; i < n; i++) {
            let v = val(dep.src, srcIdxFor(dep, idx, i));
            if (v !== null) xs.push(v);
        }
        let ms = xs.reduce((a, b) => a + b * b, 0) / xs.length;
        return {
            ...base,
            op: 'Root mean square (rms)',
            formula: `rms = √( (1/${n}) Σᵢ x[i]² + ε )`,
            words: `The typical size of this token's ${n} numbers. Unlike LayerNorm there is no mean to subtract.`,
            termHeader: ['i', 'x[i]'],
            terms: xs.slice(0, maxTerms).map((x, i) => ({ label: `${i}`, values: [x] })),
            termCount: n,
            recomputed: Math.sqrt(ms + 1e-5),
        };
    }

    if (special === BlKDepSpecial.RMSNorm) {
        let [xDep, rDep, gDep] = add;
        let x = val(xDep.src, srcIdxFor(xDep, idx, 0));
        let r = val(rDep.src, srcIdxFor(rDep, idx, 0));
        let g = val(gDep.src, srcIdxFor(gDep, idx, 0));
        return {
            ...base,
            op: 'RMS norm',
            formula: 'x / rms · γ',
            words: 'Divide by the token\'s typical size so every token has a similar scale, then apply the learned scale γ.',
            termHeader: ['part', 'value'],
            terms: [{ label: 'x', values: [x] }, { label: 'rms', values: [r] }, { label: 'γ', values: [g] }],
            termCount: 3,
            recomputed: x !== null && r !== null && g !== null ? x / r * g : null,
        };
    }

    if (special === BlKDepSpecial.RoPE && meta?.kind === 'rope') {
        let h = meta.half;
        let a = idx.y, t = idx.x;
        let upper = a >= h;
        let partner = upper ? a - h : a + h;
        let own = val(meta.src, new Vec3(t, a, idx.z));
        let other = val(meta.src, new Vec3(t, partner, idx.z));
        let c = val(meta.cos, new Vec3(t, a, 0));
        let sn = val(meta.sin, new Vec3(t, a, 0));
        let ok = own !== null && other !== null && c !== null && sn !== null;
        let angle = ok ? Math.atan2(sn!, c!) : 0;
        return {
            ...base,
            op: 'RoPE rotation',
            formula: upper ? `x[${a}]·cos θ + x[${partner}]·sin θ` : `x[${a}]·cos θ − x[${partner}]·sin θ`,
            words: `Channels ${Math.min(a, partner)} and ${Math.max(a, partner)} form a pair. The pair is rotated by an angle θ that grows with the position (t = ${t}), so the dot product of Q and K depends on how far apart two tokens are.`,
            termHeader: ['part', 'value'],
            terms: [{ label: `x[${a}]`, values: [own] }, { label: `x[${partner}]`, values: [other] }, { label: 'cos θ', values: [c] }, { label: 'sin θ', values: [sn] }, { label: 'θ (rad)', values: [ok ? angle : null] }],
            termCount: 5,
            recomputed: ok ? (upper ? own! * c! + other! * sn! : own! * c! - other! * sn!) : null,
        };
    }

    if (special === BlKDepSpecial.SoftmaxDirect && meta?.kind === 'softmax') {
        let alongX = meta.axis === 'x';
        let n = alongX ? meta.src.cx : meta.src.cy;
        if (meta.causal) n = Math.min(n, idx.y + 1);
        let xs: number[] = [];
        for (let i = 0; i < n; i++) {
            let v = val(meta.src, alongX ? new Vec3(i, idx.y, idx.z) : new Vec3(idx.x, i, idx.z));
            if (v !== null) xs.push(v);
        }
        let m = Math.max(...xs);
        let sum = xs.reduce((a, x) => a + Math.exp(x - m), 0);
        let own = alongX ? idx.x : idx.y;
        let x = xs[own];
        return {
            ...base,
            op: 'Softmax',
            formula: `exp(x[${own}] − max) / Σᵢ exp(x[i] − max)`,
            words: meta.causal
                ? `Turn token ${idx.y}'s ${n} scores into probabilities that add up to 1. Later tokens are blocked.`
                : `Turn the ${n} scores into probabilities that add up to 1.`,
            termHeader: ['i', 'score'],
            terms: xs.slice(0, maxTerms).map((v, i) => ({ label: i === own ? `${i} ◀` : `${i}`, values: [v] })),
            termCount: n,
            recomputed: x !== undefined ? Math.exp(x - m) / sum : null,
        };
    }

    if (special === BlKDepSpecial.SwiGLU) {
        let [gDep, uDep] = add;
        let g = val(gDep.src, srcIdxFor(gDep, idx, 0));
        let u = val(uDep.src, srcIdxFor(uDep, idx, 0));
        return {
            ...base,
            op: 'SwiGLU',
            formula: 'silu(gate) · up,   silu(g) = g / (1 + e^(−g))',
            words: 'The gate decides how much of the "up" value passes. Negative gates close it, positive gates open it.',
            termHeader: ['part', 'value'],
            terms: [{ label: 'gate', values: [g] }, { label: 'silu(gate)', values: [g !== null ? silu(g) : null] }, { label: 'up', values: [u] }],
            termCount: 3,
            recomputed: g !== null && u !== null ? silu(g) * u : null,
        };
    }

    if (special === BlKDepSpecial.MoEGate && meta?.kind === 'moeGate') {
        let probsBlk = meta.probs;
        let E = probsBlk.cy;
        let ps = Array.from({ length: E }, (_, e) => val(probsBlk, new Vec3(idx.x, e, idx.z)) ?? 0);
        let order = ps.map((p, e) => ({ p, e })).sort((a, b) => b.p - a.p);
        let top = order.slice(0, meta.topK);
        let tot = top.reduce((a, o) => a + o.p, 0);
        let chosen = top.some(o => o.e === idx.y);
        return {
            ...base,
            op: chosen ? `Chosen expert (top-${meta.topK})` : 'Expert not chosen',
            formula: chosen ? `p[${idx.y}] / (sum of the top-${meta.topK} p)` : '0',
            words: chosen
                ? `Token ${idx.x} is sent to its ${meta.topK} most likely experts. Expert ${idx.y} is one of them; its share is renormalised so the chosen weights add up to 1.`
                : `Token ${idx.x} does not use expert ${idx.y}: it is not in the top ${meta.topK}.`,
            termHeader: ['expert', 'router p'],
            terms: order.map(o => ({ label: `${o.e}${top.includes(o) ? ' ✓' : ''}`, values: [o.p] })),
            termCount: E,
            recomputed: chosen ? ps[idx.y] / tot : 0,
        };
    }

    if (special === BlKDepSpecial.MoECombine && meta?.kind === 'moeCombine') {
        let sum = 0, ok = true;
        let terms: ICellTerm[] = [];
        let weightsBlk = meta.weights;
        meta.experts.forEach((ex, e) => {
            let w = val(weightsBlk, new Vec3(idx.x, e, idx.z));
            let y = val(ex, new Vec3(idx.x, idx.y, idx.z));
            if (w === null || y === null) { ok = false; return; }
            sum += w * y;
            terms.push({ label: `${e}${w > 0 ? ' ✓' : ''}`, values: [w, y, w * y] });
        });
        return {
            ...base,
            op: 'Mix the chosen experts',
            formula: 'Σₑ w[e] · expert_e(x)',
            words: 'Add up the experts\' outputs, each times its weight. Experts that were not chosen have weight 0, so they cost nothing in a real model.',
            termHeader: ['expert', 'w', 'output', 'w·out'],
            terms,
            termCount: meta.experts.length,
            recomputed: ok ? sum : null,
        };
    }

    return null;
}
