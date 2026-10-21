import { IRenderState } from "../render/modelRender";
import { IBlkDef } from "../GptModelLayout";
import { arrowKit, BlockPos, CornerMode, drawArrow } from "../components/Arrow";
import { IModernLayout, IModernMlp } from "./ModernLayout";
import { Vec3 } from "@/src/utils/vector";

/** Data-flow arrows for the modern (Llama-style) layout. Uses the same arrow style as GPT-2. */
export function drawModernArrows(state: IRenderState, layout: IModernLayout) {
    let k = arrowKit(state, layout.cell);
    let { drawVertArrow: vert, drawHorizArrow: horiz, drawArrowResidSplit: residSplit, drawArrowBetween: between, blkColor } = k;
    let width = 6;
    let normal = new Vec3(0, 0, 1);

    let midX = (b: IBlkDef) => b.x + b.dx / 2;
    let midZ = (b: IBlkDef) => b.z + b.dz / 2;

    /** One input feeding several blocks in the next row: down into the gap, left along it, then down
     *  into each block (curving in z to reach blocks that sit in front of or behind the source). */
    function bus(src: IBlkDef, dests: IBlkDef[]) {
        let shown = dests.filter(d => Math.min(src.opacity, d.opacity) > 0);
        if (shown.length === 0) return;
        let color = blkColor(src).mul(Math.max(...shown.map(d => Math.min(src.opacity, d.opacity))));
        let x0 = midX(src), z0 = midZ(src);
        let top = Math.min(...shown.map(d => d.y));
        let gapY = (src.y + src.dy + top) / 2;
        let farX = Math.min(...shown.map(midX));
        drawArrow(state, new Vec3(x0, src.y + src.dy + 2, z0), new Vec3(x0, gapY - width / 2, z0), width, normal, color, false);
        drawArrow(state, new Vec3(x0 - width / 2, gapY, z0), new Vec3(farX + width / 2, gapY, z0), width, normal, color, false, CornerMode.Left);
        for (let d of shown) {
            let c = blkColor(src).mul(Math.min(src.opacity, d.opacity));
            let isEnd = Math.abs(midX(d) - farX) < 0.01;
            drawArrow(state, new Vec3(midX(d), gapY + (isEnd ? width / 2 : 0), z0), new Vec3(midX(d), d.y - 2, midZ(d)), 4, normal, c, true, isEnd ? CornerMode.Right : CornerMode.None);
        }
    }

    /** Several blocks in one row feeding one block below and to the left: the reverse of bus(). */
    function gather(srcs: IBlkDef[], dest: IBlkDef) {
        let shown = srcs.filter(s => Math.min(s.opacity, dest.opacity) > 0);
        if (shown.length === 0) return;
        let color = blkColor(shown[0]).mul(Math.max(...shown.map(s => Math.min(s.opacity, dest.opacity))));
        let bot = Math.max(...shown.map(s => s.y + s.dy));
        let gapY = (bot + dest.y) / 2;
        let x0 = midX(shown[0]), z0 = midZ(dest), xD = midX(dest);
        for (let s of shown) {
            drawArrow(state, new Vec3(midX(s), s.y + s.dy + 2, midZ(s)), new Vec3(midX(s), gapY - width / 2, z0), 4, normal, blkColor(s).mul(Math.min(s.opacity, dest.opacity)), false);
        }
        drawArrow(state, new Vec3(x0 - width / 2, gapY, z0), new Vec3(xD + width / 2, gapY, z0), width, normal, color, false, CornerMode.Left);
        drawArrow(state, new Vec3(xD, gapY + width / 2, z0), new Vec3(xD, dest.y - 2, z0), width, normal, color, true, CornerMode.Right);
    }

    /** A shared key or value feeding a query head's block in another z row. It runs straight along its own
     *  row, then curves across in the gap just before the target, entering in the lower part of its left face. */
    function side(src: IBlkDef, dest: IBlkDef) {
        let o = Math.min(src.opacity, dest.opacity);
        if (o <= 0) return;
        let w = 4, pad = 2;
        let color = blkColor(src).mul(o);
        let y = (b: IBlkDef) => b.y + b.dy * 0.8;
        let end = new Vec3(dest.x - pad, y(dest), midZ(dest));
        let gapStart = new Vec3(dest.x - layout.margin + pad, y(src), midZ(src));
        let start = new Vec3(src.x + src.dx + pad, y(src), midZ(src));
        if (gapStart.x - start.x > 1) {
            drawArrow(state, start, gapStart, w, normal, color, false);
        }
        drawArrow(state, gapStart.x - start.x > 1 ? gapStart : start, end, w, normal, color, true);
    }

    function norm(resid: IBlkDef, n: { rms: IBlkDef, gamma: IBlkDef, out: IBlkDef }) {
        residSplit(resid, n.rms, 2);
        residSplit(resid, n.out);
        vert(n.rms, n.out, 2);
        horiz(n.gamma, n.out);
    }

    function swiglu(m: IModernMlp) {
        horiz(m.wGate, m.gate);
        horiz(m.wUp, m.up);
        between(m.gate, BlockPos.Right, m.act, BlockPos.Left);
        between(m.up, BlockPos.Right, m.act, BlockPos.Left);
        vert(m.act, m.down);
        horiz(m.wDown, m.down);
    }

    // embedding
    vert(layout.idxObj, layout.residual0);
    horiz(layout.tokEmbedObj, layout.residual0);

    let resid = layout.residual0;
    for (let b of layout.blocks) {
        // residual stream
        vert(resid, b.resid1);
        vert(b.resid1, b.resid2);

        // attention
        norm(resid, b.norm1);
        bus(b.norm1.out, [...b.heads.map(h => h.q), ...b.kv.map(kv => kv.k), ...b.kv.map(kv => kv.v)]);
        for (let kv of b.kv) {
            horiz(kv.wk, kv.k);
            horiz(kv.k, kv.kRope);
            horiz(kv.wv, kv.v);
        }
        for (let h of b.heads) {
            let kv = b.kv[h.kvGroup];
            horiz(h.wq, h.q);
            horiz(h.q, h.qRope);
            horiz(h.qRope, h.scores);
            horiz(h.scores, h.probs);
            horiz(h.probs, h.out);
            side(kv.kRope, h.scores);
            side(kv.v, h.out);
        }
        gather(b.heads.map(h => h.out), b.concat);
        horiz(b.concat, b.wo);
        horiz(b.wo, b.attnOut);
        horiz(b.attnOut, b.resid1);

        // feed-forward
        norm(b.resid1, b.norm2);
        if (b.moe) {
            let m = b.moe;
            bus(b.norm2.out, [m.logits, ...m.experts.flatMap(e => [e.gate, e.up])]);
            horiz(m.wRouter, m.logits);
            horiz(m.logits, m.probs);
            horiz(m.probs, m.weights);
            vert(m.weights, m.out);
            for (let e of m.experts) {
                swiglu(e);
                vert(e.down, m.out);
            }
            horiz(m.out, b.resid2);
        } else {
            let m = b.mlp!;
            bus(b.norm2.out, [m.gate, m.up]);
            swiglu(m);
            horiz(m.down, b.resid2);
        }
        resid = b.resid2;
    }

    // output
    norm(resid, layout.normF);
    vert(layout.normF.out, layout.logits);
    horiz(layout.lmHead, layout.logits);
    horiz(layout.logits, layout.logitsSoftmax);

    // logit-lens probes read the stream
    layout.lens.forEach((l, i) => between(layout.residuals[i], BlockPos.Right, l, BlockPos.Left, 2));

}
