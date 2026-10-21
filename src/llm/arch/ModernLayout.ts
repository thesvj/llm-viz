import { BlKDepSpecial, IBlkAccessDefArgs, IBlkDef, IBlkDefArgs, IBlkLabel, mkBlk } from "../GptModelLayout";
import { IModelShape } from "../GptModel";
import { ILayoutBase } from "../LayoutBase";
import { DimStyle } from "../walkthrough/WalkthroughTools";
import { Vec3 } from "@/src/utils/vector";
import { IModernConfig } from "./ModernForward";
import { IModernRuntime } from "./ModernRuntime";

/** Blocks of one attention head (query side). */
export interface IModernHead {
    wq: IBlkDef; q: IBlkDef; qRope: IBlkDef; scores: IBlkDef; probs: IBlkDef; out: IBlkDef;
    kvGroup: number;
}

/** Blocks of one key/value head, shared by a group of query heads (grouped-query attention). */
export interface IModernKvHead {
    wk: IBlkDef; k: IBlkDef; kRope: IBlkDef; wv: IBlkDef; v: IBlkDef;
}

export interface IModernNorm { rms: IBlkDef; gamma: IBlkDef; out: IBlkDef; }

export interface IModernMlp {
    wGate: IBlkDef; gate: IBlkDef; wUp: IBlkDef; up: IBlkDef; act: IBlkDef; wDown: IBlkDef; down: IBlkDef;
}

export interface IModernBlock {
    norm1: IModernNorm;
    heads: IModernHead[];
    kv: IModernKvHead[];
    cos: IBlkDef; sin: IBlkDef;
    concat: IBlkDef; wo: IBlkDef; attnOut: IBlkDef; resid1: IBlkDef;
    norm2: IModernNorm;
    mlp: IModernMlp | null;            // dense SwiGLU
    moe: { wRouter: IBlkDef, logits: IBlkDef, probs: IBlkDef, weights: IBlkDef, experts: IModernMlp[], out: IBlkDef } | null;
    resid2: IBlkDef;
    labels: IBlkLabel[];
}

export interface IModernLayout extends ILayoutBase {
    kind: 'modern';
    config: IModernConfig;
    blocks: IModernBlock[];
    normF: IModernNorm;
    lmHead: IBlkDef;
    logits: IBlkDef;
    residuals: IBlkDef[]; // the residual stream, top to bottom
}

export function genModernLayout(config: IModernConfig, rt: IModernRuntime | null, offset: Vec3 = new Vec3()): IModernLayout {
    let T = config.block_size, C = config.n_embd, nH = config.n_head, nKV = config.n_kv_head;
    let A = C / nH, H = config.n_hidden, V = config.vocab_size, L = config.n_layer;
    let E = config.n_expert ?? 0, K = config.top_k ?? 0;
    let rep = nH / nKV;

    let cell = 1.5;
    let margin = Math.max(12, C / 10);
    let cubes: IBlkDef[] = [];

    // x columns: residual stream in the middle, operation outputs to its left, their weights further left
    let resHalf = T * cell / 2;
    let colOutR = -resHalf - margin;
    let colWR = colOutR - T * cell - margin;
    let zs = cell + 16; // spacing between parallel blocks (heads, experts) along z

    function mk(args: IBlkDefArgs, meta?: IBlkDef['meta']): IBlkDef {
        let b = mkBlk({ ...args, xL: args.xL !== undefined ? args.xL + offset.x : undefined, xR: args.xR !== undefined ? args.xR + offset.x : undefined, xM: args.xM !== undefined ? args.xM + offset.x : undefined, y: args.y + offset.y, zF: args.zF !== undefined ? args.zF + offset.z : undefined, zB: args.zB !== undefined ? args.zB + offset.z : undefined, zM: args.zM !== undefined ? args.zM + offset.z : undefined }, cell);
        if (meta) b.meta = meta;
        cubes.push(b);
        return b;
    }

    // access helpers (texX = column of the stored tensor, texY = row)
    function acc(key: string, x: number[], y: number[]): IBlkAccessDefArgs {
        return { src: rt?.tex[key], x, y, scale: 1 };
    }
    let tc = (key: string, rowOff = 0) => acc(key, [0, 1, 0], [1, 0, 0, rowOff]); // [.., T, X] shown x = T, y = X
    let wt = (key: string, rowOff = 0) => acc('w:' + key, [1, 0, 0], [0, 1, 0, rowOff]); // weight [Out, In] shown x = In, y = Out
    let vecX = (key: string) => acc(key, [1, 0, 0], [0, 0, 0]); // [T] shown x = T
    let vecY = (key: string) => acc(key, [0, 1, 0], [0, 0, 0]); // [C] shown y = C

    function labelOf(...bs: IBlkDef[]): IBlkLabel { return { visible: 0, cubes: bs }; }

    let y = 0;

    // ---- tokens + embedding (no learned positions: RoPE adds position inside attention) ----
    let idxObj = mk({ t: 'i', cx: T, cy: 1, cz: 1, y, xM: 0, zM: 0, access: { ...vecX('tokens'), scale: 1 / V }, dimX: DimStyle.T, dimY: DimStyle.None, name: 'Tokens' });
    y += cell + margin;
    let tokEmbedObj = mk({ t: 'w', cx: V, cy: C, cz: 1, y, xR: colOutR, zM: 0, access: acc('w:tok_emb.weight', [0, 1, 0], [1, 0, 0]), dimX: DimStyle.n_vocab, dimY: DimStyle.C, name: 'Token Embed' });
    let residual0 = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xM: 0, zM: 0, access: tc('embed'), deps: { add: [[tokEmbedObj, 'iy'], [idxObj, 'x0']], special: BlKDepSpecial.InputEmbed }, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Input Embed' });
    let labels: IBlkLabel[] = [labelOf(idxObj, tokEmbedObj, residual0)];
    y += C * cell + margin;

    function norm(src: IBlkDef, key: string, weightKey: string, name: string): IModernNorm {
        let rms = mk({ t: 'a', cx: T, cy: 1, cz: 1, y, xR: colOutR, zM: 0, access: vecX(`${key}.rms`), deps: { add: [[src, 'xi']], special: BlKDepSpecial.RMSNormAgg }, dimX: DimStyle.T, dimY: DimStyle.None, name: `${name}: rms`, small: true });
        y += cell + margin;
        let gamma = mk({ t: 'w', cx: 1, cy: C, cz: 1, y, xR: colWR, zM: 0, access: vecY('w:' + weightKey), dimX: DimStyle.None, dimY: DimStyle.C, name: 'γ', small: true });
        let out = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xR: colOutR, zM: 0, access: tc(`${key}.out`), deps: { add: [[src, 'xy'], [rms, 'x0'], [gamma, '0y']], special: BlKDepSpecial.RMSNorm }, dimX: DimStyle.T, dimY: DimStyle.C, name });
        y += C * cell + margin;
        return { rms, gamma, out };
    }

    function swiglu(src: IBlkDef, wp: string, ap: string, zGate: number, zUp: number, zMid: number, title: string): IModernMlp {
        let wGate = mk({ t: 'w', cx: C, cy: H, cz: 1, y, xR: colWR, zM: zGate, access: wt(`${wp}.gate.weight`), dimX: DimStyle.C, dimY: DimStyle.H, name: `${title}Gate W` });
        let gate = mk({ t: 'i', cx: T, cy: H, cz: 1, y, xR: colOutR, zM: zGate, access: tc(`${ap}.gate`), deps: { dot: [[wGate, 'iy'], [src, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.H, name: `${title}Gate` });
        let wUp = mk({ t: 'w', cx: C, cy: H, cz: 1, y, xR: colWR, zM: zUp, access: wt(`${wp}.up.weight`), dimX: DimStyle.C, dimY: DimStyle.H, name: `${title}Up W` });
        let up = mk({ t: 'i', cx: T, cy: H, cz: 1, y, xR: colOutR, zM: zUp, access: tc(`${ap}.up`), deps: { dot: [[wUp, 'iy'], [src, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.H, name: `${title}Up` });
        y += H * cell + margin;
        let act = mk({ t: 'i', cx: T, cy: H, cz: 1, y, xR: colOutR, zM: zMid, access: tc(`${ap}.act`), deps: { add: [[gate, 'xy'], [up, 'xy']], special: BlKDepSpecial.SwiGLU }, dimX: DimStyle.T, dimY: DimStyle.H, name: `${title}SwiGLU: silu(gate)·up` });
        y += H * cell + margin;
        let wDown = mk({ t: 'w', cx: H, cy: C, cz: 1, y, xR: colWR, zM: zMid, access: wt(`${wp}.down.weight`), dimX: DimStyle.H, dimY: DimStyle.C, name: `${title}Down W` });
        let down = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xR: colOutR, zM: zMid, access: tc(`${ap}.out`), deps: { dot: [[wDown, 'iy'], [act, 'xi']], dotLen: H }, dimX: DimStyle.T, dimY: DimStyle.C, name: `${title}Down` });
        return { wGate, gate, wUp, up, act, wDown, down };
    }

    let blocks: IModernBlock[] = [];
    let residuals: IBlkDef[] = [residual0];
    let resid = residual0;

    // z slot of each query head and each kv head: per group [q, q, …, k, v]
    let slotsPerGroup = rep + 2;
    let nSlots = nKV * slotsPerGroup;
    let zSlot = (s: number) => (s - (nSlots - 1) / 2) * zs;
    let zQ = (h: number) => zSlot(Math.floor(h / rep) * slotsPerGroup + (h % rep));
    let zK = (g: number) => zSlot(g * slotsPerGroup + rep);
    let zV = (g: number) => zSlot(g * slotsPerGroup + rep + 1);

    for (let l = 0; l < L; l++) {
        let wp = `blocks.${l}`, ap = `l${l}`;
        let blockLabels: IBlkLabel[] = [];
        let norm1 = norm(resid, `${ap}.norm1`, `${wp}.norm1.weight`, 'RMS Norm');
        blockLabels.push(labelOf(norm1.rms, norm1.gamma, norm1.out));
        let x = norm1.out;

        // Q / K / V projections
        let yQkv = y;
        let heads: IModernHead[] = [];
        let kv: IModernKvHead[] = [];
        let qBlocks: IBlkDef[] = [], wqBlocks: IBlkDef[] = [];
        for (let h = 0; h < nH; h++) {
            let wq = mk({ t: 'w', cx: C, cy: A, cz: 1, y: yQkv, xR: colWR, zM: zQ(h), access: wt(`${wp}.attn.wq.weight`, h * A), dimX: DimStyle.C, dimY: DimStyle.A, name: `Q W (head ${h})` });
            let q = mk({ t: 'i', cx: T, cy: A, cz: 1, y: yQkv, xR: colOutR, zM: zQ(h), access: tc(`${ap}.attn.q`, h * T), deps: { dot: [[wq, 'iy'], [x, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.A, name: `Q (head ${h})` });
            wqBlocks.push(wq); qBlocks.push(q);
        }
        let kBlocks: IBlkDef[] = [], vBlocks: IBlkDef[] = [], wkBlocks: IBlkDef[] = [], wvBlocks: IBlkDef[] = [];
        for (let g = 0; g < nKV; g++) {
            let wk = mk({ t: 'w', cx: C, cy: A, cz: 1, y: yQkv, xR: colWR, zM: zK(g), access: wt(`${wp}.attn.wk.weight`, g * A), dimX: DimStyle.C, dimY: DimStyle.A, name: `K W (kv ${g})` });
            let k = mk({ t: 'i', cx: T, cy: A, cz: 1, y: yQkv, xR: colOutR, zM: zK(g), access: tc(`${ap}.attn.k`, g * T), deps: { dot: [[wk, 'iy'], [x, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.A, name: `K (kv ${g})` });
            let wv = mk({ t: 'w', cx: C, cy: A, cz: 1, y: yQkv, xR: colWR, zM: zV(g), access: wt(`${wp}.attn.wv.weight`, g * A), dimX: DimStyle.C, dimY: DimStyle.A, name: `V W (kv ${g})` });
            let v = mk({ t: 'i', cx: T, cy: A, cz: 1, y: yQkv, xR: colOutR, zM: zV(g), access: tc(`${ap}.attn.v`, g * T), deps: { dot: [[wv, 'iy'], [x, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.A, name: `V (kv ${g}, shared)` });
            wkBlocks.push(wk); kBlocks.push(k); wvBlocks.push(wv); vBlocks.push(v);
        }
        y += A * cell + margin;

        // RoPE: rotate each (j, j + A/2) channel pair of Q and K by an angle that grows with the position
        let zEnd = zSlot(nSlots - 1) + zs;
        let cos = mk({ t: 'a', cx: T, cy: A, cz: 1, y, xR: colWR, zM: -zEnd, access: tc('rope.cosFull'), dimX: DimStyle.T, dimY: DimStyle.A, name: 'RoPE cos(angle)' });
        let sin = mk({ t: 'a', cx: T, cy: A, cz: 1, y, xR: colWR, zM: zEnd, access: tc('rope.sinFull'), dimX: DimStyle.T, dimY: DimStyle.A, name: 'RoPE sin(angle)' });
        let qRopes: IBlkDef[] = [], kRopes: IBlkDef[] = [];
        for (let h = 0; h < nH; h++) {
            qRopes.push(mk({ t: 'i', cx: T, cy: A, cz: 1, y, xR: colOutR, zM: zQ(h), access: tc(`${ap}.attn.q_rope`, h * T), deps: { add: [[qBlocks[h], 'xy'], [cos, 'xy'], [sin, 'xy']], special: BlKDepSpecial.RoPE }, dimX: DimStyle.T, dimY: DimStyle.A, name: `Q + RoPE (head ${h})` },
                { kind: 'rope', src: qBlocks[h], cos, sin, half: A / 2 }));
        }
        for (let g = 0; g < nKV; g++) {
            kRopes.push(mk({ t: 'i', cx: T, cy: A, cz: 1, y, xR: colOutR, zM: zK(g), access: tc(`${ap}.attn.k_rope`, g * T), deps: { add: [[kBlocks[g], 'xy'], [cos, 'xy'], [sin, 'xy']], special: BlKDepSpecial.RoPE }, dimX: DimStyle.T, dimY: DimStyle.A, name: `K + RoPE (kv ${g})` },
                { kind: 'rope', src: kBlocks[g], cos, sin, half: A / 2 }));
        }
        y += A * cell + margin;

        // scores, probabilities, weighted sum of V — per query head, reading its group's shared K and V
        let yScores = y, yProbs = y + T * cell + margin, yOut = yProbs + T * cell + margin;
        for (let h = 0; h < nH; h++) {
            let g = Math.floor(h / rep);
            let scores = mk({ t: 'i', cx: T, cy: T, cz: 1, y: yScores, xR: colOutR, zM: zQ(h), access: acc(`${ap}.attn.scores`, [1, 0, 0], [0, 1, 0, h * T]), deps: { dot: [[qRopes[h], 'yi'], [kRopes[g], 'xi']], lowerTri: true, dotLen: A, special: BlKDepSpecial.Attention }, dimX: DimStyle.T, dimY: DimStyle.T, name: `Scores (head ${h})` });
            let probs = mk({ t: 'i', cx: T, cy: T, cz: 1, y: yProbs, xR: colOutR, zM: zQ(h), access: acc(`${ap}.attn.probs`, [1, 0, 0], [0, 1, 0, h * T]), deps: { add: [[scores, 'iy']], lowerTri: true, special: BlKDepSpecial.SoftmaxDirect }, dimX: DimStyle.T, dimY: DimStyle.T, name: `Attention (head ${h})` },
                { kind: 'softmax', src: scores, axis: 'x', causal: true });
            let out = mk({ t: 'i', cx: T, cy: A, cz: 1, y: yOut, xR: colOutR, zM: zQ(h), access: acc(`${ap}.attn.heads`, [0, 1, 0, h * A], [1, 0, 0]), deps: { dot: [[vBlocks[g], 'iy'], [probs, 'ix']], dotLen: T }, dimX: DimStyle.T, dimY: DimStyle.A, name: `Head ${h} output` });
            heads.push({ wq: wqBlocks[h], q: qBlocks[h], qRope: qRopes[h], scores, probs, out, kvGroup: g });
            blockLabels.push(labelOf(wqBlocks[h], qBlocks[h], qRopes[h], scores, probs, out));
        }
        for (let g = 0; g < nKV; g++) {
            kv.push({ wk: wkBlocks[g], k: kBlocks[g], kRope: kRopes[g], wv: wvBlocks[g], v: vBlocks[g] });
            blockLabels.push(labelOf(wkBlocks[g], kBlocks[g], kRopes[g], wvBlocks[g], vBlocks[g]));
        }
        y = yOut + A * cell + margin;

        // concat heads -> output projection -> residual add
        let concat = mk({ t: 'i', cx: T, cy: nH * A, cz: 1, y, xR: colOutR, zM: 0, access: tc(`${ap}.attn.heads`), dimX: DimStyle.T, dimY: DimStyle.C, name: 'Heads (concatenated)' },
            { kind: 'concat', parts: heads.map(hd => hd.out), partSize: A });
        y += nH * A * cell + margin;
        let wo = mk({ t: 'w', cx: nH * A, cy: C, cz: 1, y, xR: colWR, zM: 0, access: wt(`${wp}.attn.wo.weight`), dimX: DimStyle.C, dimY: DimStyle.C, name: 'Output W' });
        let attnOut = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xR: colOutR, zM: 0, access: tc(`${ap}.attn.out`), deps: { dot: [[wo, 'iy'], [concat, 'xi']], dotLen: nH * A }, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Attention Output' });
        let resid1 = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xM: 0, zM: 0, access: tc(`${ap}.resid1`), deps: { add: [[attnOut, 'xy'], [resid, 'xy']] }, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Attention Residual' });
        blockLabels.push(labelOf(concat, wo, attnOut, resid1));
        residuals.push(resid1);
        y += C * cell + margin;

        let norm2 = norm(resid1, `${ap}.norm2`, `${wp}.norm2.weight`, 'RMS Norm');
        blockLabels.push(labelOf(norm2.rms, norm2.gamma, norm2.out));

        let mlp: IModernMlp | null = null;
        let moe: IModernBlock['moe'] = null;
        let mlpOut: IBlkDef;
        if (E > 0) {
            let wRouter = mk({ t: 'w', cx: C, cy: E, cz: 1, y, xR: colWR, zM: 0, access: wt(`${wp}.mlp.router.weight`), dimX: DimStyle.C, dimY: DimStyle.E, name: 'Router W' });
            let logits = mk({ t: 'i', cx: T, cy: E, cz: 1, y, xR: colOutR, zM: 0, access: tc(`${ap}.mlp.router_logits`), deps: { dot: [[wRouter, 'iy'], [norm2.out, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.E, name: 'Router scores' });
            y += E * cell + margin;
            let probs = mk({ t: 'i', cx: T, cy: E, cz: 1, y, xR: colOutR, zM: 0, access: tc(`${ap}.mlp.router_probs`), deps: { add: [[logits, 'xi']], special: BlKDepSpecial.SoftmaxDirect }, dimX: DimStyle.T, dimY: DimStyle.E, name: 'Router probabilities' },
                { kind: 'softmax', src: logits, axis: 'y', causal: false });
            y += E * cell + margin;
            let weights = mk({ t: 'i', cx: T, cy: E, cz: 1, y, xR: colOutR, zM: 0, access: tc(`${ap}.mlp.weights`), deps: { add: [[probs, 'xi']], special: BlKDepSpecial.MoEGate }, dimX: DimStyle.T, dimY: DimStyle.E, name: `Expert weights (top-${K})` },
                { kind: 'moeGate', probs, topK: K });
            y += E * cell + margin;
            let zE = (e: number) => (e - (E - 1) / 2) * zs * 2;
            let yExp = y;
            let experts: IModernMlp[] = [];
            for (let e = 0; e < E; e++) {
                y = yExp;
                experts.push(swiglu(norm2.out, `${wp}.mlp.experts.${e}`, `${ap}.mlp.e${e}`, zE(e) - zs / 2, zE(e) + zs / 2, zE(e), `E${e} `));
            }
            y += C * cell + margin;
            mlpOut = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xR: colOutR, zM: 0, access: tc(`${ap}.mlp.out`), deps: { add: [...experts.map(ex => [ex.down, 'xy'] as [IBlkDef, string]), [weights, 'xi']], special: BlKDepSpecial.MoECombine }, dimX: DimStyle.T, dimY: DimStyle.C, name: 'MoE output' },
                { kind: 'moeCombine', experts: experts.map(ex => ex.down), weights });
            moe = { wRouter, logits, probs, weights, experts, out: mlpOut };
            blockLabels.push(labelOf(wRouter, logits, probs, weights));
            experts.forEach(ex => blockLabels.push(labelOf(ex.wGate, ex.gate, ex.wUp, ex.up, ex.act, ex.wDown, ex.down)));
        } else {
            mlp = swiglu(norm2.out, `${wp}.mlp`, `${ap}.mlp`, -zs, zs, 0, '');
            mlpOut = mlp.down;
            blockLabels.push(labelOf(mlp.wGate, mlp.gate, mlp.wUp, mlp.up, mlp.act, mlp.wDown, mlp.down));
        }
        let resid2 = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xM: 0, zM: 0, access: tc(`${ap}.resid2`), deps: { add: [[mlpOut, 'xy'], [resid1, 'xy']] }, dimX: DimStyle.T, dimY: DimStyle.C, name: E > 0 ? 'MoE Residual' : 'MLP Residual' });
        residuals.push(resid2);
        y += C * cell + margin;

        blocks.push({ norm1, heads, kv, cos, sin, concat, wo, attnOut, resid1, norm2, mlp, moe, resid2, labels: blockLabels });
        labels.push(...blockLabels);
        resid = resid2;
    }

    // ---- final norm, LM head, softmax ----
    let normF = norm(resid, 'norm_f', 'norm_f.weight', 'Final RMS Norm');
    let lmHead = mk({ t: 'w', cx: C, cy: V, cz: 1, y, xR: colWR, zM: 0, access: wt('lm_head.weight'), dimX: DimStyle.C, dimY: DimStyle.n_vocab, name: 'LM Head W' });
    let logits = mk({ t: 'i', cx: T, cy: V, cz: 1, y, xR: colOutR, zM: 0, access: tc('logits'), deps: { dot: [[lmHead, 'iy'], [normF.out, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.n_vocab, name: 'Logits' });
    y += V * cell + margin;
    let logitsSoftmax = mk({ t: 'i', cx: T, cy: V, cz: 1, y, xR: colOutR, zM: 0, access: tc('probs'), deps: { add: [[logits, 'xi']], special: BlKDepSpecial.SoftmaxDirect }, dimX: DimStyle.T, dimY: DimStyle.n_vocab, name: 'Next-token probabilities' },
        { kind: 'softmax', src: logits, axis: 'y', causal: false });
    y += V * cell + margin;
    labels.push(labelOf(normF.rms, normF.gamma, normF.out), labelOf(lmHead, logits, logitsSoftmax));

    cubes.forEach((c, i) => c.idx = i);

    let shape: IModelShape = { B: 1, T, C, nHeads: nH, A, nBlocks: L, vocabSize: V };

    return {
        kind: 'modern',
        config,
        cubes, cell, margin, height: y,
        shape, labels,
        weightCount: config.n_params ?? 0,
        idxObj, tokEmbedObj, posEmbedObj: null, residual0, logitsSoftmax,
        model: rt,
        attnHeads: blocks.map(b => b.heads.map(h => h.probs)),
        tokenText: (id: number) => String.fromCharCode('A'.charCodeAt(0) + id),
        blocks, normF, lmHead, logits, residuals,
    };
}
