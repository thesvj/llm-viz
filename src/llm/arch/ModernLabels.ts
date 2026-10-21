import { IRenderState } from "../render/modelRender";
import { drawSectionLabel } from "../components/SectionLabels";
import { IBlkDef } from "../GptModelLayout";
import { IModernLayout } from "./ModernLayout";
import { Vec3, Vec4 } from "@/src/utils/vector";

/** Section labels for the modern layout: embedding, each layer, its attention and feed-forward parts,
 *  and one bracket per key/value group naming the query heads that share it. */
export function drawModernLabels(state: IRenderState, layout: IModernLayout) {
    let m = layout.margin;
    let base = new Vec4(0.35, 0.35, 0.4, 1.0);
    let left = Math.min(...layout.cubes.map(c => c.x)) - m;

    let top = (b: IBlkDef) => b.y;
    let bot = (b: IBlkDef) => b.y + b.dy;
    let col = (o: number) => base.mul(Math.max(0.15, o));

    {
        let e = layout.tokEmbedObj;
        drawSectionLabel(state, 'Embedding', new Vec3(e.x - m, top(e), 0), new Vec3(e.x - m, bot(e), 0), { color: col(e.opacity), fontSize: 8, pad: 4 });
    }

    layout.blocks.forEach((b, i) => {
        let first = b.norm1.rms, last = b.resid2;
        let ffTop = b.norm2.rms;
        let ffBot = b.moe ? b.moe.out : b.mlp!.down;
        let opac = (bs: IBlkDef[]) => Math.max(...bs.map(x => x.opacity));

        drawSectionLabel(state, `Layer ${i + 1}`, new Vec3(left - m * 5, top(first), 0), new Vec3(left - m * 5, bot(last), 0),
            { color: col(opac([b.resid1, b.resid2])), fontSize: 18 });
        drawSectionLabel(state, 'Attention', new Vec3(left, top(first), 0), new Vec3(left, bot(b.resid1), 0),
            { color: col(opac([b.attnOut, ...b.heads.map(h => h.probs)])), fontSize: 12 });
        drawSectionLabel(state, b.moe ? 'Mixture of experts' : 'Feed-forward (SwiGLU)', new Vec3(left, top(ffTop), 0), new Vec3(left, bot(ffBot), 0),
            { color: col(opac([b.resid2, ...(b.moe ? b.moe.experts.map(e => e.act) : [b.mlp!.act])])), fontSize: 12 });

        // one bracket per key/value group, spanning its query heads and its shared K and V (along z)
        b.kv.forEach((kv, g) => {
            let qs = b.heads.filter(h => h.kvGroup === g);
            let zA = Math.min(...qs.map(h => h.wq.z)), zB = kv.wv.z + kv.wv.dz;
            let xL = kv.wk.x - m / 2;
            let yM = kv.wk.y + kv.wk.dy / 2;
            let names = qs.map(h => b.heads.indexOf(h)).join(', ');
            drawSectionLabel(state, `Q ${names} share K${g} V${g}`, new Vec3(xL, yM, zA), new Vec3(xL, yM, zB),
                { color: col(opac([kv.k, kv.v, ...qs.map(h => h.q)])), fontSize: 4, pad: 3 });
        });
        if (b.moe) {
            b.moe.experts.forEach((ex, e) => {
                let xL = ex.wGate.x - m / 2;
                let yM = ex.wGate.y + ex.wGate.dy / 2;
                drawSectionLabel(state, `Expert ${e}`, new Vec3(xL, yM, ex.wGate.z), new Vec3(xL, yM, ex.wUp.z + ex.wUp.dz),
                    { color: col(opac([ex.act])), fontSize: 6, pad: 3 });
            });
        }
    });

    {
        let n = layout.normF.rms, p = layout.logitsSoftmax;
        drawSectionLabel(state, 'Output', new Vec3(left, top(n), 0), new Vec3(left, bot(p), 0), { color: col(p.opacity), fontSize: 12 });
    }
}
