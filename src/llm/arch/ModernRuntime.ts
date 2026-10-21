import { createBufferTex, IBufferTex, writeToBufferTex } from "@/src/utils/renderPhases";
import { IModelRuntimeLink } from "../LayoutBase";
import { decodeTensorJson, IModernConfig, IModernWeights, ITensor, ITensorMap, logitLens, modernForward } from "./ModernForward";

/** A tiny modern model running in the page: weights + latest activations, each tensor in its own texture. */
export interface IModernRuntime extends IModelRuntimeLink {
    gl: WebGL2RenderingContext;
    name: string;
    model: IModernWeights;
    tokens: number[];          // length = block_size; positions >= inputLen are padding
    rec: ITensorMap;           // activations from the latest run
    tex: Record<string, IBufferTex>; // 'w:<weight name>' for weights, '<activation name>' for activations
    /** Logit lens per residual snapshot (embed, then resid1/resid2 of each layer): [T, V] probabilities. */
    lens: { name: string, probs: ITensor }[];
    /** The last generation step: how the token was picked, and whether the cached K/V stayed the same. */
    lastStep: ISampleInfo | null;
    rngState: number;
}

export interface ISampleInfo {
    pos: number;            // position whose prediction was used
    temperature: number;
    probs: number[];        // probabilities after temperature
    draw: number;           // the random number in [0, 1)
    picked: number;         // chosen token id
    cacheCols: number;      // number of earlier positions whose K/V were reused
    cacheSame: boolean;     // true when those K/V columns are bit-identical to the previous run
}

/** Small seeded random generator (mulberry32), so a run can be repeated exactly. */
function nextRandom(rt: IModernRuntime): number {
    let t = (rt.rngState = (rt.rngState + 0x6D2B79F5) | 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

/** softmax(logits / temperature) for one position. Temperature 0 = always the top token. */
export function tempProbs(rt: IModernRuntime, pos: number, temperature: number): number[] {
    let V = rt.model.config.vocab_size;
    let lg = Array.from(rt.rec['logits'].data.slice(pos * V, pos * V + V));
    if (temperature <= 0) {
        let best = lg.indexOf(Math.max(...lg));
        return lg.map((_, i) => i === best ? 1 : 0);
    }
    let m = Math.max(...lg);
    let e = lg.map(v => Math.exp((v - m) / temperature));
    let s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
}

/** Texture for a tensor: last dim = width, all other dims flattened = height. */
function texFor(rt: IModernRuntime, key: string, t: ITensor): IBufferTex {
    let width = t.shape[t.shape.length - 1];
    let height = t.data.length / width;
    let tex = rt.tex[key];
    if (!tex || tex.width !== width || tex.height !== height) {
        tex = createBufferTex(rt.gl, width, height, 1);
        rt.tex[key] = tex;
    }
    tex.localBuffer = t.data;
    writeToBufferTex(rt.gl, tex, t.data);
    return tex;
}

/** cos/sin tables widened to the full head size [T, A], so channel a uses angle pair a mod A/2. */
function widenRope(t: ITensor): ITensor {
    let [T, h] = t.shape;
    let out = new Float32Array(T * 2 * h);
    for (let r = 0; r < T; r++)
        for (let a = 0; a < 2 * h; a++) out[r * 2 * h + a] = t.data[r * h + (a % h)];
    return { shape: [T, 2 * h], data: out };
}

export function createModernRuntime(gl: WebGL2RenderingContext, name: string, json: Record<string, any>): IModernRuntime {
    let config = json.config as IModernConfig;
    let model: IModernWeights = { config, w: decodeTensorJson(json) };
    let T = config.block_size;
    let initial = [2, 1, 0, 1, 1, 2]; // "C B A B B C", same example as nano-gpt
    let tokens = Array.from({ length: T }, (_, i) => initial[i] ?? 0);
    let rt: IModernRuntime = {
        gl, name, model, tokens, rec: {}, tex: {},
        inputTokens: null!, inputLen: initial.length, sortedBuf: null,
        lens: [], lastStep: null, rngState: 12345,
    };
    for (let [k, t] of Object.entries(model.w)) {
        texFor(rt, 'w:' + k, t);
    }
    runModern(rt);
    return rt;
}

/** Run the forward pass on the current tokens and refresh every activation texture. */
export function runModern(rt: IModernRuntime) {
    let { config } = rt.model;
    let T = config.block_size, V = config.vocab_size;
    rt.rec = modernForward(rt.model, rt.tokens);
    rt.rec['rope.cosFull'] = widenRope(rt.rec['rope.cos']);
    rt.rec['rope.sinFull'] = widenRope(rt.rec['rope.sin']);
    for (let [k, t] of Object.entries(rt.rec)) {
        texFor(rt, k, t);
    }
    rt.inputTokens = texFor(rt, 'tokens', { shape: [T], data: Float32Array.from(rt.tokens) });

    // (token id, probability) pairs per position, most likely first — same format the model card reads
    let probs = rt.rec['probs'].data;
    let sorted = new Float32Array(T * V * 2);
    for (let t = 0; t < T; t++) {
        let opts = Array.from({ length: V }, (_, i) => ({ i, v: probs[t * V + i] })).sort((a, b) => b.v - a.v);
        opts.forEach((o, j) => { sorted[(t * V + j) * 2] = o.i; sorted[(t * V + j) * 2 + 1] = o.v; });
    }
    rt.sortedBuf = sorted;

    let snaps = ['embed'];
    for (let l = 0; l < config.n_layer; l++) snaps.push(`l${l}.resid1`, `l${l}.resid2`);
    rt.lens = snaps.map(name => ({ name, probs: logitLens(rt.model, rt.rec[name]) }));
}

/** One generation step: sample the next token from the last real position (with temperature), append it, re-run.
 *  Also checks the KV-cache idea: keys and values of earlier positions must not change. */
export function stepModern(rt: IModernRuntime, temperature = 0) {
    let { config } = rt.model;
    let T = config.block_size;
    if (rt.inputLen >= T) {
        return;
    }
    let pos = rt.inputLen - 1;
    let probs = tempProbs(rt, pos, temperature);
    let draw = nextRandom(rt);
    let picked = probs.length - 1;
    for (let i = 0, acc = 0; i < probs.length; i++) {
        acc += probs[i];
        if (draw < acc) { picked = i; break; }
    }

    // keep the K and V of the positions we already had
    let A = config.head_dim ?? config.n_embd / config.n_head;
    let before: Float32Array[] = [];
    for (let l = 0; l < config.n_layer; l++) {
        before.push(rt.rec[`l${l}.attn.k_rope`].data.slice(), rt.rec[`l${l}.attn.v`].data.slice());
    }

    rt.tokens[rt.inputLen] = picked;
    rt.inputLen += 1;
    runModern(rt);

    let cacheCols = rt.inputLen - 1;
    let cacheSame = true;
    for (let l = 0; l < config.n_layer; l++) {
        let after = [rt.rec[`l${l}.attn.k_rope`].data, rt.rec[`l${l}.attn.v`].data];
        for (let j = 0; j < 2; j++) {
            let a = before[2 * l + j], b = after[j];
            // layout [nKV, T, A]: compare columns t < cacheCols in every head
            for (let h = 0; h < config.n_kv_head; h++)
                for (let t = 0; t < cacheCols; t++)
                    for (let c = 0; c < A; c++) {
                        let i = (h * T + t) * A + c;
                        if (a[i] !== b[i]) cacheSame = false;
                    }
        }
    }
    rt.lastStep = { pos, temperature, probs, draw, picked, cacheCols, cacheSame };
}

/** Replace the prompt (letters A/B/C as ids 0..2) and re-run. */
export function setModernPrompt(rt: IModernRuntime, ids: number[]) {
    let T = rt.model.config.block_size;
    rt.tokens = Array.from({ length: T }, (_, i) => ids[i] ?? 0);
    rt.inputLen = Math.max(1, Math.min(ids.length, T));
    rt.lastStep = null;
    runModern(rt);
}
