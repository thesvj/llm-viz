// Plain forward pass for the tiny modern (Llama-style) models made by tools/train_tiny_modern.py.
// No browser or app imports, so it can be checked against the PyTorch export from Node.
// Every intermediate is recorded under the same name and shape as the Python export (batch dimension removed).

export interface IModernConfig {
    vocab_size: number;
    block_size: number;
    n_embd: number;
    n_layer: number;
    n_head: number;
    n_kv_head: number;
    n_hidden: number;
    rope_theta: number;
    norm_eps: number;
    n_expert?: number;
    top_k?: number;
    arch: 'llama' | 'moe';
    n_params?: number;
    // shape-only presets of real models (not used by the tiny trained models)
    head_dim?: number;        // when not n_embd / n_head
    tied?: boolean;           // LM head shares the token embedding matrix
    qk_norm?: boolean;        // RMS norm on each Q and K head (Qwen3)
    sliding_window?: number;  // each token attends only to this many previous tokens (Mistral)
    local_every?: number;     // with sliding_window: only some layers are local; 2 = every other layer (gpt-oss)
    bias?: boolean;           // attention, router and expert layers have bias vectors (gpt-oss)
    sinks?: boolean;
    active_note?: string;     // shown under the parameter count when the lab counts "active" differently          // one learned "sink" score per head that can soak up attention (gpt-oss)
}

export interface ITensor {
    shape: number[];
    data: Float32Array;
}

export type ITensorMap = Record<string, ITensor>;

export interface IModernWeights {
    config: IModernConfig;
    w: ITensorMap;
}

function tensor(shape: number[], data?: Float32Array): ITensor {
    let n = shape.reduce((a, b) => a * b, 1);
    return { shape, data: data ?? new Float32Array(n) };
}

/** y[t, o] = Σ_i x[t, i] · W[o, i]   (nn.Linear without bias; W is [out, in]) */
function linear(x: ITensor, W: ITensor): ITensor {
    let [T, In] = x.shape;
    let [Out, In2] = W.shape;
    if (In !== In2) throw new Error(`linear: ${In} vs ${In2}`);
    let y = tensor([T, Out]);
    for (let t = 0; t < T; t++) {
        for (let o = 0; o < Out; o++) {
            let s = 0;
            for (let i = 0; i < In; i++) s += x.data[t * In + i] * W.data[o * In + i];
            y.data[t * Out + o] = s;
        }
    }
    return y;
}

function rmsNorm(x: ITensor, weight: ITensor, eps: number) {
    let [T, C] = x.shape;
    let rms = tensor([T]);
    let out = tensor([T, C]);
    for (let t = 0; t < T; t++) {
        let ss = 0;
        for (let c = 0; c < C; c++) ss += x.data[t * C + c] ** 2;
        let r = Math.sqrt(ss / C + eps);
        rms.data[t] = r;
        for (let c = 0; c < C; c++) out.data[t * C + c] = x.data[t * C + c] / r * weight.data[c];
    }
    return { rms, out };
}

/** cos/sin tables [T, A/2]; pair j rotates channels (j, j + A/2) — "rotate half", as in Llama / HF. */
export function ropeTables(T: number, A: number, theta: number) {
    let h = A / 2;
    let cos = tensor([T, h]), sin = tensor([T, h]);
    for (let t = 0; t < T; t++) {
        for (let j = 0; j < h; j++) {
            let inv = 1 / Math.pow(theta, (2 * j) / A);
            let ang = Math.fround(t * Math.fround(inv));
            cos.data[t * h + j] = Math.cos(ang);
            sin.data[t * h + j] = Math.sin(ang);
        }
    }
    return { cos, sin };
}

/** [T, nH*A] -> [nH, T, A] */
function splitHeads(x: ITensor, nH: number, A: number): ITensor {
    let T = x.shape[0];
    let y = tensor([nH, T, A]);
    for (let h = 0; h < nH; h++)
        for (let t = 0; t < T; t++)
            for (let a = 0; a < A; a++)
                y.data[(h * T + t) * A + a] = x.data[t * nH * A + h * A + a];
    return y;
}

function applyRope(x: ITensor, cos: ITensor, sin: ITensor): ITensor {
    let [nH, T, A] = x.shape;
    let h = A / 2;
    let y = tensor([nH, T, A]);
    for (let hd = 0; hd < nH; hd++)
        for (let t = 0; t < T; t++) {
            let base = (hd * T + t) * A;
            for (let j = 0; j < h; j++) {
                let c = cos.data[t * h + j], s = sin.data[t * h + j];
                let x1 = x.data[base + j], x2 = x.data[base + j + h];
                y.data[base + j] = x1 * c - x2 * s;
                y.data[base + j + h] = x2 * c + x1 * s;
            }
        }
    return y;
}

function silu(x: number) {
    return x / (1 + Math.exp(-x));
}

function swiglu(x: ITensor, w: ITensorMap, p: string, rec: ITensorMap, recPrefix: string) {
    let gate = linear(x, w[`${p}.gate.weight`]);
    let up = linear(x, w[`${p}.up.weight`]);
    let act = tensor(gate.shape);
    for (let i = 0; i < act.data.length; i++) act.data[i] = silu(gate.data[i]) * up.data[i];
    let out = linear(act, w[`${p}.down.weight`]);
    rec[`${recPrefix}.gate`] = gate;
    rec[`${recPrefix}.up`] = up;
    rec[`${recPrefix}.act`] = act;
    rec[`${recPrefix}.out`] = out;
    return out;
}

function addInto(a: ITensor, b: ITensor): ITensor {
    let y = tensor(a.shape);
    for (let i = 0; i < y.data.length; i++) y.data[i] = a.data[i] + b.data[i];
    return y;
}

function softmaxRows(x: Float32Array, rows: number, cols: number, out: Float32Array, limit?: (r: number) => number) {
    for (let r = 0; r < rows; r++) {
        let n = limit ? limit(r) : cols;
        let m = -Infinity;
        for (let c = 0; c < n; c++) m = Math.max(m, x[r * cols + c]);
        let s = 0;
        for (let c = 0; c < n; c++) s += Math.exp(x[r * cols + c] - m);
        for (let c = 0; c < cols; c++) out[r * cols + c] = c < n ? Math.exp(x[r * cols + c] - m) / s : 0;
    }
}

/** Runs the model on `tokens` (length = block_size; pad unused positions — the causal mask keeps them out). */
export function modernForward(model: IModernWeights, tokens: number[]): ITensorMap {
    let cfg = model.config;
    let w = model.w;
    let C = cfg.n_embd, nH = cfg.n_head, nKV = cfg.n_kv_head, A = C / nH, V = cfg.vocab_size;
    let T = tokens.length;
    let rec: ITensorMap = {};

    let x = tensor([T, C]);
    let emb = w['tok_emb.weight'];
    for (let t = 0; t < T; t++) x.data.set(emb.data.subarray(tokens[t] * C, tokens[t] * C + C), t * C);
    rec['embed'] = x;

    let { cos, sin } = ropeTables(cfg.block_size, A, cfg.rope_theta);
    rec['rope.cos'] = cos;
    rec['rope.sin'] = sin;

    for (let l = 0; l < cfg.n_layer; l++) {
        let p = `blocks.${l}`, r = `l${l}`;

        let n1 = rmsNorm(x, w[`${p}.norm1.weight`], cfg.norm_eps);
        rec[`${r}.norm1.rms`] = n1.rms;
        rec[`${r}.norm1.out`] = n1.out;

        let q = splitHeads(linear(n1.out, w[`${p}.attn.wq.weight`]), nH, A);
        let k = splitHeads(linear(n1.out, w[`${p}.attn.wk.weight`]), nKV, A);
        let v = splitHeads(linear(n1.out, w[`${p}.attn.wv.weight`]), nKV, A);
        let qr = applyRope(q, cos, sin), kr = applyRope(k, cos, sin);
        rec[`${r}.attn.q`] = q; rec[`${r}.attn.k`] = k; rec[`${r}.attn.v`] = v;
        rec[`${r}.attn.q_rope`] = qr; rec[`${r}.attn.k_rope`] = kr;

        let rep = nH / nKV; // query heads that share one key/value head
        let scores = tensor([nH, T, T]);
        let probs = tensor([nH, T, T]);
        let scale = 1 / Math.sqrt(A);
        for (let h = 0; h < nH; h++) {
            let kh = Math.floor(h / rep);
            for (let tq = 0; tq < T; tq++)
                for (let tk = 0; tk <= tq; tk++) {
                    let s = 0;
                    for (let a = 0; a < A; a++) s += qr.data[(h * T + tq) * A + a] * kr.data[(kh * T + tk) * A + a];
                    scores.data[(h * T + tq) * T + tk] = s * scale;
                }
        }
        softmaxRows(scores.data, nH * T, T, probs.data, row => (row % T) + 1);
        rec[`${r}.attn.scores`] = scores;
        rec[`${r}.attn.probs`] = probs;

        let heads = tensor([T, nH * A]);
        for (let h = 0; h < nH; h++) {
            let kh = Math.floor(h / rep);
            for (let tq = 0; tq < T; tq++)
                for (let a = 0; a < A; a++) {
                    let s = 0;
                    for (let tk = 0; tk <= tq; tk++) s += probs.data[(h * T + tq) * T + tk] * v.data[(kh * T + tk) * A + a];
                    heads.data[tq * nH * A + h * A + a] = s;
                }
        }
        rec[`${r}.attn.heads`] = heads;
        let attnOut = linear(heads, w[`${p}.attn.wo.weight`]);
        rec[`${r}.attn.out`] = attnOut;
        x = addInto(x, attnOut);
        rec[`${r}.resid1`] = x;

        let n2 = rmsNorm(x, w[`${p}.norm2.weight`], cfg.norm_eps);
        rec[`${r}.norm2.rms`] = n2.rms;
        rec[`${r}.norm2.out`] = n2.out;

        let mlpOut: ITensor;
        if (cfg.n_expert) {
            let E = cfg.n_expert, K = cfg.top_k!;
            let logits = linear(n2.out, w[`${p}.mlp.router.weight`]);
            let rp = tensor([T, E]);
            softmaxRows(logits.data, T, E, rp.data);
            let weights = tensor([T, E]);
            for (let t = 0; t < T; t++) {
                let order = Array.from({ length: E }, (_, e) => e).sort((a, b) => rp.data[t * E + b] - rp.data[t * E + a]);
                let top = order.slice(0, K);
                let tot = top.reduce((s, e) => s + rp.data[t * E + e], 0);
                for (let e of top) weights.data[t * E + e] = rp.data[t * E + e] / tot;
            }
            rec[`${r}.mlp.router_logits`] = logits;
            rec[`${r}.mlp.router_probs`] = rp;
            rec[`${r}.mlp.weights`] = weights;
            mlpOut = tensor([T, C]);
            for (let e = 0; e < E; e++) {
                // every expert is evaluated for every token so each one can be inspected; only the top-k count
                let y = swiglu(n2.out, w, `${p}.mlp.experts.${e}`, rec, `${r}.mlp.e${e}`);
                for (let t = 0; t < T; t++)
                    for (let c = 0; c < C; c++) mlpOut.data[t * C + c] += weights.data[t * E + e] * y.data[t * C + c];
            }
            rec[`${r}.mlp.out`] = mlpOut;
        } else {
            mlpOut = swiglu(n2.out, w, `${p}.mlp`, rec, `${r}.mlp`);
        }
        x = addInto(x, mlpOut);
        rec[`${r}.resid2`] = x;
    }

    let nf = rmsNorm(x, w['norm_f.weight'], cfg.norm_eps);
    rec['norm_f.rms'] = nf.rms;
    rec['norm_f.out'] = nf.out;
    let logits = linear(nf.out, w['lm_head.weight']);
    rec['logits'] = logits;
    let probs = tensor([T, V]);
    softmaxRows(logits.data, T, V, probs.data);
    rec['probs'] = probs;
    return rec;
}

/** Decode the JSON written by tools/train_tiny_modern.py (base64 float32 tensors). */
export function decodeTensorJson(obj: Record<string, any>): ITensorMap {
    let out: ITensorMap = {};
    for (let [k, v] of Object.entries(obj)) {
        if (!v || typeof v !== 'object' || !('data' in v) || !('shape' in v)) continue;
        let bin = typeof atob === 'function' ? atob(v.data) : Buffer.from(v.data, 'base64').toString('binary');
        let bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        out[k] = { shape: v.shape, data: new Float32Array(bytes.buffer) };
    }
    return out;
}

/** Logit lens: what the model would predict if it stopped at this point of the residual stream.
 *  Applies the final RMS norm and the LM head to `resid` [T, C], then softmax → [T, V].
 *  A probe for the guide, not part of the model. On the last residual it equals rec['probs']. */
export function logitLens(model: IModernWeights, resid: ITensor): ITensor {
    let { w, config } = model;
    let n = rmsNorm(resid, w['norm_f.weight'], config.norm_eps);
    let logits = linear(n.out, w['lm_head.weight']);
    let [T, V] = logits.shape;
    let out = tensor([T, V]);
    softmaxRows(logits.data, T, V, out.data);
    return out;
}
