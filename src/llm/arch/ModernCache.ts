import { IRenderState } from "../render/modelRender";
import { IBlkDef } from "../GptModelLayout";
import { addLine } from "../render/lineRender";
import { measureTextWidth, writeTextToBuffer } from "../render/fontRender";
import { IModernLayout } from "./ModernLayout";
import { IModernRuntime } from "./ModernRuntime";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";

const cachedColor = new Vec4(0.15, 0.35, 0.8, 1);
const newColor = new Vec4(0.95, 0.5, 0.05, 1);

/** After a generation step, outline in every key and value block the columns reused from the cache
 *  (earlier positions, unchanged) and the one new column computed for the new token. */
export function drawCacheMarks(state: IRenderState, layout: IModernLayout, rt: IModernRuntime) {
    let step = rt.lastStep;
    if (!step || step.cacheCols <= 0) return;
    let cell = layout.cell;
    let n = new Vec3(0, 0, 1);

    function rect(b: IBlkDef, t0: number, t1: number, color: Vec4, label: string, line: number) {
        let o = b.opacity;
        if (o <= 0.05) return;
        let c = color.mul(o);
        let x0 = b.x + t0 * cell, x1 = b.x + t1 * cell;
        let y0 = b.y - 0.3, y1 = b.y + b.dy + 0.3;
        for (let z of [b.z + b.dz + 0.1]) {
            addLine(state.lineRender, 0.25, c, new Vec3(x0, y0, z), new Vec3(x1, y0, z), n);
            addLine(state.lineRender, 0.25, c, new Vec3(x1, y0, z), new Vec3(x1, y1, z), n);
            addLine(state.lineRender, 0.25, c, new Vec3(x1, y1, z), new Vec3(x0, y1, z), n);
            addLine(state.lineRender, 0.25, c, new Vec3(x0, y1, z), new Vec3(x0, y0, z), n);
        }
        let size = 2.2;
        let mtx = new Mat4f();
        mtx[14] = b.z + b.dz + 0.2;
        let tw = measureTextWidth(state.modelFontBuf, label, size);
        writeTextToBuffer(state.modelFontBuf, label, c, (x0 + x1) / 2 - tw / 2, y1 + 0.8 + line * (size + 0.4), size, mtx);
    }

    for (let b of layout.blocks) {
        for (let kv of b.kv) {
            for (let blk of [kv.kRope, kv.v]) {
                rect(blk, 0, step.cacheCols, cachedColor, 'cached', 0);
                rect(blk, step.cacheCols, step.cacheCols + 1, newColor, 'new', 1);
            }
        }
    }
}
