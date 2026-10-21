import { IProgramState } from "../Program";
import { IModernLayout } from "../arch/ModernLayout";
import { IModernRuntime } from "../arch/ModernRuntime";
import { IBlkDef } from "../GptModelLayout";
import { buildFlowSteps, IFlowStep } from "./FlowSteps";
import { flyToBlocks } from "./flowCamera";

/** The guide is shown for the live modern models (not for GPT-2, which has its own walkthrough). */
export function flowActive(state: IProgramState): boolean {
    return state.modelMode !== 'gpt2' && state.currExampleId === -1 && state.layout.kind === 'modern';
}

export function flowRuntime(state: IProgramState): IModernRuntime | null {
    return state.modelMode !== 'gpt2' ? state.modern[state.modelMode] ?? null : null;
}

export function flowSteps(state: IProgramState): IFlowStep[] {
    if (!flowActive(state)) {
        return [];
    }
    let layout = state.layout as IModernLayout;
    let layer = Math.min(state.flow.layer, layout.blocks.length - 1);
    return buildFlowSteps(layout, layer);
}

/** Blocks to keep lit for the current step. For MoE steps, experts the followed token does not use are left dim. */
export function flowFocusBlocks(state: IProgramState): IBlkDef[] {
    let steps = flowSteps(state);
    let step = steps[state.flow.step];
    if (!step) {
        return [];
    }
    let layout = state.layout as IModernLayout;
    let moe = layout.blocks[state.flow.layer]?.moe;
    if (moe && (step.id === 'experts' || step.id === 'combine')) {
        let used = expertsUsed(state);
        let unused = new Set(moe.experts.filter((_, e) => !used.includes(e)).flatMap(ex => [ex.wGate, ex.gate, ex.wUp, ex.up, ex.act, ex.wDown, ex.down]));
        return step.blocks.filter(b => !unused.has(b));
    }
    return step.blocks;
}

/** Experts (indices) the followed token uses in the current layer, by non-zero weight. */
export function expertsUsed(state: IProgramState): number[] {
    let rt = flowRuntime(state);
    let E = rt?.model.config.n_expert ?? 0;
    if (!rt || !E) {
        return [];
    }
    let w = rt.rec[`l${state.flow.layer}.mlp.weights`].data;
    let t = state.flow.token;
    return Array.from({ length: E }, (_, e) => e).filter(e => w[t * E + e] > 0);
}

export function goToStep(state: IProgramState, idx: number) {
    let steps = flowSteps(state);
    if (steps.length === 0) {
        return;
    }
    idx = Math.max(0, Math.min(steps.length - 1, idx));
    state.flow.step = idx;
    state.display.pinned = null;
    flyToBlocks(state, flowFocusBlocks(state), undefined, 1.3, undefined, 1.2);
    saveFlowView(state);
    state.markDirty();
}

const storeKey = 'llmviz.view';

/** Remember the model and guide position across page reloads (per browser; failures are ignored). */
export function saveFlowView(state: IProgramState) {
    try {
        localStorage.setItem(storeKey, JSON.stringify({ mode: state.modelMode, step: state.flow.step, layer: state.flow.layer }));
    } catch { /* storage unavailable */ }
}

export function restoreFlowView(state: IProgramState) {
    try {
        let v = JSON.parse(localStorage.getItem(storeKey) ?? 'null');
        if (v && (v.mode === 'llama' || v.mode === 'moe') && state.modern[v.mode as 'llama' | 'moe']) {
            let rt = state.modern[v.mode as 'llama' | 'moe']!;
            state.modelMode = v.mode;
            state.currExampleId = -1;
            state.flow.layer = Math.max(0, Math.min(rt.model.config.n_layer - 1, v.layer | 0));
            state.flow.token = rt.inputLen - 1;
            state.flow.step = Math.max(0, v.step | 0);
            return true;
        }
    } catch { /* storage unavailable or bad data */ }
    return false;
}
