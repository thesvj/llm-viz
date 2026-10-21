import { IBlkDef, IModelLayout } from "./GptModelLayout";
import { getBlockValueAtIdx } from "./components/DataFlow";
import { Vec3 } from "@/src/utils/vector";

/** User-facing view options (toolbar). */
export interface IVizSettings {
    /** 'auto': each block's colours span its own real value range (shown in the legend). 'original': hand-picked scales. */
    colorScale: 'auto' | 'original';
    /** 'diverging': blue (negative) – white (zero) – red (positive). 'classic': original black – grey – block colour. */
    palette: 'diverging' | 'classic';
    /** Dim every block except the hovered/pinned one and its direct inputs and outputs. */
    focus: boolean;
}

export function defaultVizSettings(): IVizSettings {
    return { colorScale: 'auto', palette: 'diverging', focus: true };
}

const maxSamples = 2048;

/** 99th-percentile of |value| over the block's own cells (sampled), or null when the block has no CPU-side data. */
export function blockAbsRange(blk: IBlkDef): number | null {
    if (!blk.access?.src.localBuffer) {
        return null;
    }
    let n = blk.cx * blk.cy * blk.cz;
    let step = Math.max(1, Math.floor(n / maxSamples));
    let vals: number[] = [];
    let idx = new Vec3();
    for (let i = 0; i < n; i += step) {
        idx.x = i % blk.cx;
        idx.y = Math.floor(i / blk.cx) % blk.cy;
        idx.z = Math.floor(i / (blk.cx * blk.cy));
        let v = getBlockValueAtIdx(blk, idx);
        if (v !== null && isFinite(v)) {
            vals.push(Math.abs(v));
        }
    }
    if (vals.length === 0) {
        return null;
    }
    vals.sort((a, b) => a - b);
    let p = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.99))];
    return p > 0 ? p : (vals[vals.length - 1] || null);
}

/** Store each block's real range on its access info, and (in 'auto' mode) use it as the colour scale. */
export function applyValueRanges(layout: IModelLayout, settings: IVizSettings, skip: Set<IBlkDef>) {
    for (let blk of layout.cubes) {
        if (!blk.access || skip.has(blk)) {
            continue;
        }
        let range = blockAbsRange(blk);
        blk.access.range = range ?? undefined;
        if (settings.colorScale === 'auto' && range) {
            blk.access.scale = 1 / range;
        }
    }
}

/** Blocks a block reads from (its deps). */
export function depSources(blk: IBlkDef): IBlkDef[] {
    let d = blk.deps;
    if (!d) {
        return [];
    }
    return [...(d.dot?.map(a => a.src) ?? []), ...(d.add?.map(a => a.src) ?? [])];
}

/** Focus mode: keep the target, its inputs and its outputs; dim the rest. */
export function applyFocus(layout: IModelLayout, target: IBlkDef | null) {
    if (!target) {
        return;
    }
    let keep = new Set<IBlkDef>([target, ...depSources(target)]);
    for (let blk of layout.cubes) {
        if (depSources(blk).includes(target)) {
            keep.add(blk);
        }
    }
    function dim(c: IBlkDef) {
        c.opacity *= 0.12;
        c.subs?.forEach(dim);
    }
    for (let blk of layout.cubes) {
        if (!keep.has(blk) && blk.opacity > 0) {
            dim(blk);
        }
    }
}
