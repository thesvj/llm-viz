import { IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { BoundingBox3d, Vec3 } from "@/src/utils/vector";
import { clamp } from "@/src/utils/data";

/** The default viewing angle (same as nano-gpt's start view). */
export const defaultViewAngle = new Vec3(284.959, 26.501, 1);

/** Smoothly move the camera so that all the given blocks fit on screen. */
export function flyToBlocks(state: IProgramState, blocks: IBlkDef[], angle: Vec3 = defaultViewAngle, zoomScale = 1, offset: Vec3 = new Vec3(), minZoom = 0.3) {
    let visible = blocks.filter(b => b);
    if (visible.length === 0) {
        return;
    }
    let mtx = state.camera.modelMtx;
    let bb = new BoundingBox3d();
    for (let b of visible) {
        bb.addInPlace(mtx.mulVec3Proj(new Vec3(b.x, b.y, b.z).add(offset)));
        bb.addInPlace(mtx.mulVec3Proj(new Vec3(b.x + b.dx, b.y + b.dy, b.z + b.dz).add(offset)));
    }
    // same rule as the movement controls: zoom grows with the size of the box
    let zoom = clamp(bb.size().len() / 110 * zoomScale, minZoom, 1e5);
    state.camera.desiredCamera = { center: bb.center(), angle: new Vec3(angle.x, angle.y, zoom) };
    state.markDirty();
}
