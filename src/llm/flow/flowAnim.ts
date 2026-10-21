// Step playback for the modern-model guide: the step's blocks fill in cell by cell, the way the
// GPT-2 walkthrough shows them being computed. The "how this number is made" overlay follows the
// cell being computed. Nothing is hidden until the user presses Play.

import { IProgramState } from "../Program";
import { IRenderView } from "../render/modelRender";
import { IBlkDef } from "../GptModelLayout";
import { findSubBlocks, splitGrid } from "../Annotations";
import { drawDataFlow } from "../components/DataFlow";
import { drawDependences } from "../Interaction";
import { DimStyle } from "../walkthrough/WalkthroughTools";
import { Dim, Vec3 } from "@/src/utils/vector";
import { flowFocusBlocks, flowRuntime, flowSteps, goToStep } from "./flowState";

export type FlowScope = 'all' | 'token';

export interface IFlowAnim {
    t: number;          // 0 = nothing computed yet, 1 = step finished
    playing: boolean;
    speed: number;      // 1 = normal
    scope: FlowScope;   // every token, or only the followed token
    auto: boolean;      // go on to the next step when one finishes
    hold: number;       // seconds waited after playback finished (-1 = not waiting)
}

export const defaultFlowAnim = (): IFlowAnim => ({ t: 1, playing: false, speed: 1, scope: 'all', auto: false, hold: -1 });

/** Seconds per block at speed 1. */
const secsPerBlock = { all: 2.5, token: 1.2 };
/** Pause at the end of a step before auto-advancing. */
const holdSecs = 0.8;

/** Steps that show many layers at once. They do not play, and do not count as computing anything. */
const overviewSteps = new Set(['residual', 'layers']);

/** Blocks the current step computes, in compute order. Weights, plain tables and blocks that an
 *  earlier step already computed (they are this step's inputs) are not animated. */
export function animBlocks(state: IProgramState): IBlkDef[] {
    let steps = flowSteps(state);
    if (overviewSteps.has(steps[state.flow.step]?.id)) {
        return [];
    }
    let earlier = new Set(steps.slice(0, state.flow.step).filter(s => !overviewSteps.has(s.id)).flatMap(s => s.blocks));
    return flowFocusBlocks(state).filter(b => b.t !== 'w' && b.deps && !earlier.has(b)).sort((a, b) => a.idx - b.idx);
}

export function stepDuration(state: IProgramState) {
    let a = state.flow.anim;
    return Math.max(1, animBlocks(state).length) * secsPerBlock[a.scope];
}

const isSquare = (b: IBlkDef) => b.dimX === DimStyle.T && b.dimY === DimStyle.T;

/** Outer dimension = tokens. For token × token blocks (scores, weights) the outer one is the query row. */
function dims(b: IBlkDef): [Dim, Dim] {
    return isSquare(b) ? [Dim.Y, Dim.X] : [Dim.X, Dim.Y];
}

function innerCount(b: IBlkDef, outer: number) {
    return isSquare(b) ? outer + 1 : b.cy; // causal: query t only uses keys 0..t
}

/** Where playback is now: which block, which token and which cell. */
export interface IAnimCursor {
    blk: IBlkDef;
    blockIdx: number;
    nBlocks: number;
    outer: number;  // token position
    inner: number;  // row (or key position)
    idx: Vec3;      // cell index in block coords
}

export function animCursor(state: IProgramState): IAnimCursor | null {
    let rt = flowRuntime(state);
    let blocks = animBlocks(state);
    let a = state.flow.anim;
    if (!rt || blocks.length === 0 || a.t >= 1) {
        return null;
    }
    let n = blocks.length;
    let pos = Math.max(0, a.t) * n;
    let blockIdx = Math.min(n - 1, Math.floor(pos));
    let sub = pos - blockIdx;
    let blk = blocks[blockIdx];
    let nTok = rt.inputLen;
    let outer: number, inner: number;
    if (a.scope === 'token') {
        outer = Math.min(state.flow.token, nTok - 1);
        inner = Math.min(innerCount(blk, outer) - 1, Math.floor(sub * innerCount(blk, outer)));
    } else {
        let p = sub * nTok;
        outer = Math.min(nTok - 1, Math.floor(p));
        let m = innerCount(blk, outer);
        inner = Math.min(m - 1, Math.floor((p - outer) * m));
    }
    let [d0] = dims(blk);
    let idx = d0 === Dim.X ? new Vec3(outer, inner, 0) : new Vec3(inner, outer, 0);
    return { blk, blockIdx, nBlocks: n, outer, inner, idx };
}

function setAccess(b: IBlkDef, on: boolean) {
    if (b.access) {
        b.access.disable = !on;
    }
}

/** Split block b along d at index i: returns [before, at, after] sub-block lists. */
function split3(state: IProgramState, b: IBlkDef, d: Dim, i: number) {
    let mid = splitGrid(state.layout, b, d, i + 0.5, 0);
    if (!mid || mid === b) {
        return null;
    }
    return {
        before: findSubBlocks(b, d, null, i - 1),
        at: mid,
        after: findSubBlocks(b, d, i + 1, null),
    };
}

/** Advance time and apply the partial reveal. Called every frame, after the camera update. */
export function runFlowAnim(state: IProgramState, view: IRenderView) {
    let a = state.flow.anim;
    let steps = flowSteps(state);
    if (steps.length === 0) {
        return;
    }
    let dt = view.dt / 1000;

    if (a.playing) {
        a.t = Math.min(1, a.t + dt * a.speed / stepDuration(state));
        if (a.t >= 1) {
            a.playing = false;
            a.hold = 0;
        }
        view.markDirty();
    } else if (a.auto && a.t >= 1 && a.hold >= 0 && a.hold < holdSecs && state.flow.step < steps.length - 1) {
        a.hold += dt;
        if (a.hold >= holdSecs) {
            a.hold = -1; // only once per finish
            goToStep(state, state.flow.step + 1, true);
        }
        view.markDirty();
    }

    let cur = animCursor(state);
    if (!cur) {
        return;
    }
    let blocks = animBlocks(state);
    let tokenOnly = a.scope === 'token';

    // blocks after the cursor: not computed yet (all of it, or only the followed token's slice)
    for (let i = cur.blockIdx + 1; i < blocks.length; i++) {
        let b = blocks[i];
        if (tokenOnly) {
            let [d0] = dims(b);
            setAccess(b, false);
            let parts = split3(state, b, d0, cur.outer);
            if (parts) {
                [...parts.before, ...parts.after].forEach(s => setAccess(s, true));
            }
        } else {
            setAccess(b, false);
        }
    }

    // the block being computed: done part on, the current token's slice partly on, the rest off
    let b = cur.blk;
    let [d0, d1] = dims(b);
    setAccess(b, false);
    let parts = split3(state, b, d0, cur.outer);
    if (parts) {
        parts.before.forEach(s => setAccess(s, true));
        parts.after.forEach(s => setAccess(s, tokenOnly));
        let col = parts.at;
        col.highlight = 0.25;
        let cells = split3(state, col, d1, cur.inner);
        if (cells) {
            cells.before.forEach(s => setAccess(s, true));
            setAccess(cells.at, true);
            cells.at.highlight = 0.8;
        } else {
            setAccess(col, true);
        }
    }

    // show how the current number is made, unless the user is hovering or has pinned a cell
    if (!state.display.hoverTarget && !state.display.pinned) {
        drawDependences(state, b, cur.idx);
        drawDataFlow(state, b, cur.idx);
    }
}

export function playPause(state: IProgramState) {
    let a = state.flow.anim;
    if (animBlocks(state).length === 0) {
        return false;
    }
    if (a.playing) {
        a.playing = false;
    } else {
        if (a.t >= 1) a.t = 0;
        a.playing = true;
        a.hold = -1;
    }
    state.markDirty();
    return true;
}
