import { createBufferTex, IBufferTex, writeToBufferTex } from "@/src/utils/renderPhases";
import { IModelRuntimeLink } from "../LayoutBase";
import { decodeTensorJson, IModernConfig, IModernWeights, ITensor, ITensorMap, modernForward } from "./ModernForward";

/** A tiny modern model running in the page: weights + latest activations, each tensor in its own texture. */
export interface IModernRuntime extends IModelRuntimeLink {
    gl: WebGL2RenderingContext;
    name: string;
    model: IModernWeights;
    tokens: number[];          // length = block_size; positions >= inputLen are padding
    rec: ITensorMap;           // activations from the latest run
    tex: Record<string, IBufferTex>; // 'w:<weight name>' for weights, '<activation name>' for activations
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
}

/** Append the most likely next token and re-run (one generation step). */
export function stepModern(rt: IModernRuntime) {
    let T = rt.model.config.block_size;
    if (!rt.sortedBuf || rt.inputLen >= T) {
        return;
    }
    let V = rt.model.config.vocab_size;
    rt.tokens[rt.inputLen] = rt.sortedBuf[(rt.inputLen - 1) * V * 2];
    rt.inputLen += 1;
    runModern(rt);
}

/** Replace the prompt (letters A/B/C as ids 0..2) and re-run. */
export function setModernPrompt(rt: IModernRuntime, ids: number[]) {
    let T = rt.model.config.block_size;
    rt.tokens = Array.from({ length: T }, (_, i) => ids[i] ?? 0);
    rt.inputLen = Math.max(1, Math.min(ids.length, T));
    runModern(rt);
}
