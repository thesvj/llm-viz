import { BlKDepSpecial, IBlkAccessDefArgs, IBlkDef, IBlkDefArgs, IBlkLabel, mkBlk } from "../GptModelLayout";
import { IModelShape } from "../GptModel";
import { ILayoutBase } from "../LayoutBase";
import { DimStyle } from "../walkthrough/WalkthroughTools";
import { Vec3 } from "@/src/utils/vector";
import { IModernConfig } from "./ModernForward";
import { countParams } from "./ArchPresets";
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
    /** Logit-lens probes, one per residual (live models only): what the model would guess if it stopped there. */
    lens: IBlkDef[];
}

/** layersPerColumn: big models wrap their layers into side-by-side columns (like GPT-2 XL in the original). */
export function genModernLayout(config: IModernConfig, rt: IModernRuntime | null, offset: Vec3 = new Vec3(), layersPerColumn = 0): IModernLayout {
    let T = config.block_size, C = config.n_embd, nH = config.n_head, nKV = config.n_kv_head;
    let A = config.head_dim ?? C / nH, H = config.n_hidden, V = config.vocab_size, L = config.n_layer;
    let E = config.n_expert ?? 0, K = config.top_k ?? 0;
    let rep = nH / nKV;

    let cell = 1.5;
    let margin = Math.max(12, C / 10);
    let cubes: IBlkDef[] = [];

    // x: the residual stream is the centre column; each operation's chain runs left to right towards it,
    // weights furthest left. Column k (0 = nearest the stream) has its right edge at colX(k, widths...).
    let resHalf = T * cell / 2;
    let x0 = -resHalf - margin;               // right edge of the column next to the stream
    let colT = (k: number) => x0 - k * (T * cell + margin); // k-th T-wide column
    // z: parallel blocks (heads, experts). Gap inside a group < gap between groups, both scale with C.
    let zIn = cell + margin + C * cell / 16;
    let zGroup = 3 * margin + C * cell / 16;

    let colShift = 0; // x shift of the current layer column
    function mk(args: IBlkDefArgs, meta?: IBlkDef['meta']): IBlkDef {
        let ox = offset.x + colShift;
        let b = mkBlk({ ...args, xL: args.xL !== undefined ? args.xL + ox : undefined, xR: args.xR !== undefined ? args.xR + ox : undefined, xM: args.xM !== undefined ? args.xM + ox : undefined, y: args.y + offset.y, zF: args.zF !== undefined ? args.zF + offset.z : undefined, zB: args.zB !== undefined ? args.zB + offset.z : undefined, zM: args.zM !== undefined ? args.zM + offset.z : undefined }, cell);
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
    let tokEmbedObj = mk({ t: 'w', cx: V, cy: C, cz: 1, y, xR: x0, zM: 0, access: acc('w:tok_emb.weight', [0, 1, 0], [1, 0, 0]), dimX: DimStyle.n_vocab, dimY: DimStyle.C, name: 'Token Embed' });
    let residual0 = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xM: 0, zM: 0, access: tc('embed'), deps: { add: [[tokEmbedObj, 'iy'], [idxObj, 'x0']], special: BlKDepSpecial.InputEmbed }, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Input Embed' });
    let labels: IBlkLabel[] = [labelOf(idxObj, tokEmbedObj, residual0)];
    y += C * cell + margin;

    // rms row (T × 1) on top, then γ and the normalised output in one row
    function norm(src: IBlkDef, key: string, weightKey: string, name: string): IModernNorm {
        let rms = mk({ t: 'a', cx: T, cy: 1, cz: 1, y, xR: colT(0), zM: 0, access: vecX(`${key}.rms`), deps: { add: [[src, 'xi']], special: BlKDepSpecial.RMSNormAgg }, dimX: DimStyle.T, dimY: DimStyle.None, name: `${name}: rms`, small: true });
        y += cell + margin / 2;
        let gamma = mk({ t: 'w', cx: 1, cy: C, cz: 1, y, xR: colT(0) - T * cell - margin / 2, zM: 0, access: vecY('w:' + weightKey), dimX: DimStyle.None, dimY: DimStyle.C, name: 'γ', small: true });
        let out = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xR: colT(0), zM: 0, access: tc(`${key}.out`), deps: { add: [[src, 'xy'], [rms, 'x0'], [gamma, '0y']], special: BlKDepSpecial.RMSNorm }, dimX: DimStyle.T, dimY: DimStyle.C, name });
        y += C * cell + margin;
        return { rms, gamma, out };
    }

    // SwiGLU in two rows: [W gate | gate] and [W up | up] side by side on z with silu(gate)·up next to them,
    // then [W down | down]. Returns with y below the first row; caller places the down row.
    function swigluTop(src: IBlkDef, wp: string, ap: string, zM: number, title: string) {
        let zG = zM - zIn / 2, zU = zM + zIn / 2;
        let wGate = mk({ t: 'w', cx: C, cy: H, cz: 1, y, xR: colT(2), zM: zG, access: wt(`${wp}.gate.weight`), dimX: DimStyle.C, dimY: DimStyle.H, name: `${title}Gate W` });
        let gate = mk({ t: 'i', cx: T, cy: H, cz: 1, y, xR: colT(1), zM: zG, access: tc(`${ap}.gate`), deps: { dot: [[wGate, 'iy'], [src, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.H, name: `${title}Gate` });
        let wUp = mk({ t: 'w', cx: C, cy: H, cz: 1, y, xR: colT(2), zM: zU, access: wt(`${wp}.up.weight`), dimX: DimStyle.C, dimY: DimStyle.H, name: `${title}Up W` });
        let up = mk({ t: 'i', cx: T, cy: H, cz: 1, y, xR: colT(1), zM: zU, access: tc(`${ap}.up`), deps: { dot: [[wUp, 'iy'], [src, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.H, name: `${title}Up` });
        let act = mk({ t: 'i', cx: T, cy: H, cz: 1, y, xR: colT(0), zM, access: tc(`${ap}.act`), deps: { add: [[gate, 'xy'], [up, 'xy']], special: BlKDepSpecial.SwiGLU }, dimX: DimStyle.T, dimY: DimStyle.H, name: `${title}SwiGLU` });
        return { wGate, gate, wUp, up, act };
    }
    function swigluDown(top: ReturnType<typeof swigluTop>, wp: string, ap: string, zM: number, title: string, yRow: number): IModernMlp {
        let wDown = mk({ t: 'w', cx: H, cy: C, cz: 1, y: yRow, xR: colT(1), zM, access: wt(`${wp}.down.weight`), dimX: DimStyle.H, dimY: DimStyle.C, name: `${title}Down W` });
        let down = mk({ t: 'i', cx: T, cy: C, cz: 1, y: yRow, xR: colT(0), zM, access: tc(`${ap}.out`), deps: { dot: [[wDown, 'iy'], [top.act, 'xi']], dotLen: H }, dimX: DimStyle.T, dimY: DimStyle.C, name: `${title}Down` });
        return { ...top, wDown, down };
    }

    let blocks: IModernBlock[] = [];
    let residuals: IBlkDef[] = [residual0];
    let resid = residual0;

    // attention z slots: per KV group [q, q, …, k, v], with a wider gap between groups
    let groupWidth = (rep + 2 - 1) * zIn;
    let groupStride = groupWidth + zGroup;
    let zGroupStart = (g: number) => g * groupStride - ((nKV - 1) * groupStride + groupWidth) / 2;
    let zQ = (h: number) => zGroupStart(Math.floor(h / rep)) + (h % rep) * zIn;
    let zK = (g: number) => zGroupStart(g) + rep * zIn;
    let zV = (g: number) => zGroupStart(g) + (rep + 1) * zIn;
    let zAttnEnd = zGroupStart(nKV - 1) + groupWidth + zGroup;

    let yLayersTop = y;
    let colWidth = 0;
    for (let l = 0; l < L; l++) {
        if (layersPerColumn > 0 && l > 0 && l % layersPerColumn === 0) {
            // start a new column to the right, at the height of the first layer
            if (colWidth === 0) {
                let xs = cubes.map(c => c.x);
                colWidth = (Math.max(...cubes.map(c => c.x + c.dx)) - Math.min(...xs)) + margin * 20;
            }
            colShift += colWidth;
            y = yLayersTop;
        }
        let wp = `blocks.${l}`, ap = `l${l}`;
        let blockLabels: IBlkLabel[] = [];
        let norm1 = norm(resid, `${ap}.norm1`, `${wp}.norm1.weight`, 'RMS Norm');
        blockLabels.push(labelOf(norm1.rms, norm1.gamma, norm1.out));
        let x = norm1.out;

        // one row: [W | Q | Q+RoPE | Scores | Attention | Head out] per query head,
        //          [W | K | K+RoPE] and [W | V] per shared key/value head
        let yRow = y;
        let heads: IModernHead[] = [];
        let kv: IModernKvHead[] = [];
        let cos = mk({ t: 'a', cx: T, cy: A, cz: 1, y: yRow, xR: colT(3), zM: -zAttnEnd, access: tc('rope.cosFull'), dimX: DimStyle.T, dimY: DimStyle.A, name: 'RoPE cos' });
        let sin = mk({ t: 'a', cx: T, cy: A, cz: 1, y: yRow, xR: colT(3), zM: zAttnEnd, access: tc('rope.sinFull'), dimX: DimStyle.T, dimY: DimStyle.A, name: 'RoPE sin' });

        let kRopes: IBlkDef[] = [], vBlocks: IBlkDef[] = [];
        for (let g = 0; g < nKV; g++) {
            let wk = mk({ t: 'w', cx: C, cy: A, cz: 1, y: yRow, xR: colT(5), zM: zK(g), access: wt(`${wp}.attn.wk.weight`, g * A), dimX: DimStyle.C, dimY: DimStyle.A, name: `K W` });
            let k = mk({ t: 'i', cx: T, cy: A, cz: 1, y: yRow, xR: colT(4), zM: zK(g), access: tc(`${ap}.attn.k`, g * T), deps: { dot: [[wk, 'iy'], [x, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.A, name: `K ${g}` });
            let kRope = mk({ t: 'i', cx: T, cy: A, cz: 1, y: yRow, xR: colT(3), zM: zK(g), access: tc(`${ap}.attn.k_rope`, g * T), deps: { add: [[k, 'xy'], [cos, 'xy'], [sin, 'xy']], special: BlKDepSpecial.RoPE }, dimX: DimStyle.T, dimY: DimStyle.A, name: `K ${g} + RoPE` },
                { kind: 'rope', src: k, cos, sin, half: A / 2 });
            let wv = mk({ t: 'w', cx: C, cy: A, cz: 1, y: yRow, xR: colT(5), zM: zV(g), access: wt(`${wp}.attn.wv.weight`, g * A), dimX: DimStyle.C, dimY: DimStyle.A, name: `V W` });
            let v = mk({ t: 'i', cx: T, cy: A, cz: 1, y: yRow, xR: colT(4), zM: zV(g), access: tc(`${ap}.attn.v`, g * T), deps: { dot: [[wv, 'iy'], [x, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.A, name: `V ${g}` });
            kv.push({ wk, k, kRope, wv, v });
            kRopes.push(kRope); vBlocks.push(v);
        }
        for (let h = 0; h < nH; h++) {
            let g = Math.floor(h / rep);
            let wq = mk({ t: 'w', cx: C, cy: A, cz: 1, y: yRow, xR: colT(5), zM: zQ(h), access: wt(`${wp}.attn.wq.weight`, h * A), dimX: DimStyle.C, dimY: DimStyle.A, name: `Q W` });
            let q = mk({ t: 'i', cx: T, cy: A, cz: 1, y: yRow, xR: colT(4), zM: zQ(h), access: tc(`${ap}.attn.q`, h * T), deps: { dot: [[wq, 'iy'], [x, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.A, name: `Q ${h}` });
            let qRope = mk({ t: 'i', cx: T, cy: A, cz: 1, y: yRow, xR: colT(3), zM: zQ(h), access: tc(`${ap}.attn.q_rope`, h * T), deps: { add: [[q, 'xy'], [cos, 'xy'], [sin, 'xy']], special: BlKDepSpecial.RoPE }, dimX: DimStyle.T, dimY: DimStyle.A, name: `Q ${h} + RoPE` },
                { kind: 'rope', src: q, cos, sin, half: A / 2 });
            let scores = mk({ t: 'i', cx: T, cy: T, cz: 1, y: yRow, xR: colT(2), zM: zQ(h), access: acc(`${ap}.attn.scores`, [1, 0, 0], [0, 1, 0, h * T]), deps: { dot: [[qRope, 'yi'], [kRopes[g], 'xi']], lowerTri: true, dotLen: A, special: BlKDepSpecial.Attention }, dimX: DimStyle.T, dimY: DimStyle.T, name: `Scores ${h}` });
            let probs = mk({ t: 'i', cx: T, cy: T, cz: 1, y: yRow, xR: colT(1), zM: zQ(h), access: acc(`${ap}.attn.probs`, [1, 0, 0], [0, 1, 0, h * T]), deps: { add: [[scores, 'iy']], lowerTri: true, special: BlKDepSpecial.SoftmaxDirect }, dimX: DimStyle.T, dimY: DimStyle.T, name: `Attention ${h}` },
                { kind: 'softmax', src: scores, axis: 'x', causal: true });
            let out = mk({ t: 'i', cx: T, cy: A, cz: 1, y: yRow, xR: colT(0), zM: zQ(h), access: acc(`${ap}.attn.heads`, [0, 1, 0, h * A], [1, 0, 0]), deps: { dot: [[vBlocks[g], 'iy'], [probs, 'ix']], dotLen: T }, dimX: DimStyle.T, dimY: DimStyle.A, name: `Head ${h} out` });
            heads.push({ wq, q, qRope, scores, probs, out, kvGroup: g });
            blockLabels.push(labelOf(wq, q, qRope, scores, probs, out));
        }
        kv.forEach(k => blockLabels.push(labelOf(k.wk, k.k, k.kRope, k.wv, k.v)));
        y = yRow + Math.max(T, A) * cell + margin;

        // one row: [heads concatenated | output W] -> attention output, added to the stream
        let concat = mk({ t: 'i', cx: T, cy: nH * A, cz: 1, y, xR: colT(1) - C * cell - margin, zM: 0, access: tc(`${ap}.attn.heads`), dimX: DimStyle.T, dimY: DimStyle.C, name: 'Heads joined' },
            { kind: 'concat', parts: heads.map(hd => hd.out), partSize: A });
        let wo = mk({ t: 'w', cx: nH * A, cy: C, cz: 1, y, xR: colT(1), zM: 0, access: wt(`${wp}.attn.wo.weight`), dimX: DimStyle.C, dimY: DimStyle.C, name: 'Output W' });
        let attnOut = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xR: colT(0), zM: 0, access: tc(`${ap}.attn.out`), deps: { dot: [[wo, 'iy'], [concat, 'xi']], dotLen: nH * A }, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Attention Output' });
        let resid1 = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xM: 0, zM: 0, access: tc(`${ap}.resid1`), deps: { add: [[attnOut, 'xy'], [resid, 'xy']] }, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Attention Residual' });
        blockLabels.push(labelOf(concat, wo, attnOut, resid1));
        residuals.push(resid1);
        y += Math.max(C, nH * A) * cell + margin;

        let norm2 = norm(resid1, `${ap}.norm2`, `${wp}.norm2.weight`, 'RMS Norm');
        blockLabels.push(labelOf(norm2.rms, norm2.gamma, norm2.out));

        let mlp: IModernMlp | null = null;
        let moe: IModernBlock['moe'] = null;
        let mlpOut: IBlkDef;
        if (E > 0) {
            // router row: [W | scores | probabilities | top-k weights]
            let wRouter = mk({ t: 'w', cx: C, cy: E, cz: 1, y, xR: colT(3), zM: 0, access: wt(`${wp}.mlp.router.weight`), dimX: DimStyle.C, dimY: DimStyle.E, name: 'Router W' });
            let logits = mk({ t: 'i', cx: T, cy: E, cz: 1, y, xR: colT(2), zM: 0, access: tc(`${ap}.mlp.router_logits`), deps: { dot: [[wRouter, 'iy'], [norm2.out, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.E, name: 'Router scores' });
            let probs = mk({ t: 'i', cx: T, cy: E, cz: 1, y, xR: colT(1), zM: 0, access: tc(`${ap}.mlp.router_probs`), deps: { add: [[logits, 'xi']], special: BlKDepSpecial.SoftmaxDirect }, dimX: DimStyle.T, dimY: DimStyle.E, name: 'Router probs' },
                { kind: 'softmax', src: logits, axis: 'y', causal: false });
            let weights = mk({ t: 'i', cx: T, cy: E, cz: 1, y, xR: colT(0), zM: 0, access: tc(`${ap}.mlp.weights`), deps: { add: [[probs, 'xi']], special: BlKDepSpecial.MoEGate }, dimX: DimStyle.T, dimY: DimStyle.E, name: `Top-${K} weights` },
                { kind: 'moeGate', probs, topK: K });
            y += E * cell + margin;
            let expStride = 2 * zIn + zGroup;
            let zE = (e: number) => (e - (E - 1) / 2) * expStride;
            let tops = Array.from({ length: E }, (_, e) => swigluTop(norm2.out, `${wp}.mlp.experts.${e}`, `${ap}.mlp.e${e}`, zE(e), `E${e} `));
            y += H * cell + margin;
            let experts = tops.map((tp, e) => swigluDown(tp, `${wp}.mlp.experts.${e}`, `${ap}.mlp.e${e}`, zE(e), `E${e} `, y));
            y += C * cell + margin;
            mlpOut = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xR: colT(0), zM: 0, access: tc(`${ap}.mlp.out`), deps: { add: [...experts.map(ex => [ex.down, 'xy'] as [IBlkDef, string]), [weights, 'xi']], special: BlKDepSpecial.MoECombine }, dimX: DimStyle.T, dimY: DimStyle.C, name: 'MoE output' },
                { kind: 'moeCombine', experts: experts.map(ex => ex.down), weights });
            moe = { wRouter, logits, probs, weights, experts, out: mlpOut };
            blockLabels.push(labelOf(wRouter, logits, probs, weights));
            experts.forEach(ex => blockLabels.push(labelOf(ex.wGate, ex.gate, ex.wUp, ex.up, ex.act, ex.wDown, ex.down)));
        } else {
            let top = swigluTop(norm2.out, `${wp}.mlp`, `${ap}.mlp`, 0, '');
            y += H * cell + margin;
            mlp = swigluDown(top, `${wp}.mlp`, `${ap}.mlp`, 0, '', y);
            mlpOut = mlp.down;
            blockLabels.push(labelOf(mlp.wGate, mlp.gate, mlp.wUp, mlp.up, mlp.act, mlp.wDown, mlp.down));
        }
        let resid2 = mk({ t: 'i', cx: T, cy: C, cz: 1, y, xM: 0, zM: 0, access: tc(`${ap}.resid2`), deps: { add: [[mlpOut, 'xy'], [resid1, 'xy']] }, dimX: DimStyle.T, dimY: DimStyle.C, name: E > 0 ? 'MoE Residual' : 'MLP Residual' });
        residuals.push(resid2);
        y += C * cell + margin * 2;

        blocks.push({ norm1, heads, kv, cos, sin, concat, wo, attnOut, resid1, norm2, mlp, moe, resid2, labels: blockLabels });
        labels.push(...blockLabels);
        resid = resid2;
    }

    // ---- final norm, LM head, softmax ----
    let normF = norm(resid, 'norm_f', 'norm_f.weight', 'Final RMS Norm');
    let lmHead = mk({ t: 'w', cx: C, cy: V, cz: 1, y, xR: colT(1), zM: 0, access: wt('lm_head.weight'), dimX: DimStyle.C, dimY: DimStyle.n_vocab, name: 'LM Head W' });
    let logits = mk({ t: 'i', cx: T, cy: V, cz: 1, y, xR: colT(0), zM: 0, access: tc('logits'), deps: { dot: [[lmHead, 'iy'], [normF.out, 'xi']], dotLen: C }, dimX: DimStyle.T, dimY: DimStyle.n_vocab, name: 'Logits' });
    let logitsSoftmax = mk({ t: 'i', cx: T, cy: V, cz: 1, y, xM: 0, zM: 0, access: tc('probs'), deps: { add: [[logits, 'xi']], special: BlKDepSpecial.SoftmaxDirect }, dimX: DimStyle.T, dimY: DimStyle.n_vocab, name: 'Next-token probabilities' },
        { kind: 'softmax', src: logits, axis: 'y', causal: false });
    y += V * cell + margin;
    labels.push(labelOf(normF.rms, normF.gamma, normF.out), labelOf(lmHead, logits, logitsSoftmax));

    // logit-lens probes to the right of the stream; made last so the other cubes keep their index
    let lens: IBlkDef[] = [];
    if (rt) {
        residuals.forEach((r, i) => {
            let where = i === 0 ? 'after embedding' : `after layer ${Math.ceil(i / 2)}${i % 2 === 1 ? ' attention' : ''}`;
            lens.push(mk({ t: 'a', cx: T, cy: V, cz: 1, y: r.y + r.dy / 2 - V * cell / 2 - offset.y, xL: resHalf + 2 * margin, zM: 0, access: tc(`lens.${i}`), dimX: DimStyle.T, dimY: DimStyle.n_vocab, name: `Guess ${where}` },
                { kind: 'lens', src: r, gamma: normF.gamma, lmHead, eps: config.norm_eps }));
        });
    }

    cubes.forEach((c, i) => c.idx = i);

    let shape: IModelShape = { B: 1, T, C, nHeads: nH, A, nBlocks: L, vocabSize: V };

    return {
        kind: 'modern',
        config,
        cubes, cell, margin, height: y,
        shape, labels,
        weightCount: countParams(config).total,
        idxObj, tokEmbedObj, posEmbedObj: null, residual0, logitsSoftmax,
        model: rt,
        attnHeads: blocks.map(b => b.heads.map(h => h.probs)),
        tokenText: (id: number) => String.fromCharCode('A'.charCodeAt(0) + id),
        blocks, normF, lmHead, logits, residuals, lens,
    };
}
