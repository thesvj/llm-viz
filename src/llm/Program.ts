import { genModelViewMatrices, ICamera, ICameraPos, updateCamera } from "./Camera";
import { drawAllArrows } from "./components/Arrow";
import { drawBlockLabels } from "./components/SectionLabels";
import { drawModelCard } from "./components/ModelCard";
import { IGptModelLink, IGpuGptModel, IModelShape } from "./GptModel";
import { genGptModelLayout, IBlkDef, IGptModelLayout } from "./GptModelLayout";
import { ILayoutBase } from "./LayoutBase";
import { IModelMode, modelModeInfo } from "./arch/ModernModels";
import { IModernRuntime, stepModern } from "./arch/ModernRuntime";
import { IModernConfig } from "./arch/ModernForward";
import { archPresets, countParams } from "./arch/ArchPresets";
import { genModernLayout, IModernLayout } from "./arch/ModernLayout";
import { drawModernLabels } from "./arch/ModernLabels";
import { flowActive, flowFocusBlocks } from "./flow/flowState";
import { applyFocus, applyValueRanges, defaultVizSettings, IVizSettings } from "./VizSettings";
import { drawText, IFontAtlasData, IFontOpts, measureText } from "./render/fontRender";
import { initRender, IRenderState, IRenderView, renderModel, resetRenderBuffers } from "./render/modelRender";
import { beginQueryAndGetPrevMs, endQuery } from "./render/queryManager";
import { SavedState } from "./SavedState";
import { isNotNil } from "@/src/utils/data";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { initWalkthrough, runWalkthrough } from "./walkthrough/Walkthrough";
import { IColorMix } from "./Annotations";
import { Mat4f } from "@/src/utils/matrix";
import { runMouseHitTesting } from "./Interaction";
import { RenderPhase } from "./render/sharedRender";
import { drawBlockInfo } from "./components/BlockInfo";
import { NativeFunctions } from "./NativeBindings";
import { IWasmGptModel, stepWasmModel, syncWasmDataWithJsAndGpu } from "./GptModelWasm";
import { IMovementInfo, manageMovement } from "./components/MovementControls";
import { IBlockRender, initBlockRender } from "./render/blockRender";
import { ILayout } from "../utils/layout";
import { DimStyle } from "./walkthrough/WalkthroughTools";
import { Subscriptions } from "../utils/hooks";

export interface IProgramState {
    native: NativeFunctions | null;
    wasmGptModel: IWasmGptModel | null;
    stepModel: boolean;
    mouse: IMouseState;
    render: IRenderState;
    inWalkthrough: boolean;
    walkthrough: ReturnType<typeof initWalkthrough>;
    camera: ICamera;
    htmlSubs: Subscriptions;
    layout: ILayoutBase;
    mainExample: IModelExample;
    examples: IModelExample[];
    currExampleId: number;
    shape: IModelShape;
    gptGpuModel: IGpuGptModel | null;
    jsGptModel: IGptModelLink | null;
    movement: IMovementInfo;
    display: IDisplayState;
    viz: IVizSettings;
    modelMode: IModelMode;
    modern: Partial<Record<IModelMode, IModernRuntime>>;
    flow: IFlowState;
    pageLayout: ILayout;
    markDirty: () => void;
}

function paramLine(c: IModernConfig) {
    let n = countParams(c);
    let b = (x: number) => `${(x / 1e9).toFixed(2)}B`;
    return c.n_expert ? `${b(n.total)} parameters, ${b(n.active)} used per token` : `${b(n.total)} parameters`;
}

/** Step-by-step guide state (modern models). */
export interface IFlowState {
    step: number;        // index into buildFlowSteps()
    layer: number;       // which layer the guide walks through
    token: number;       // the token position the guide follows (MoE experts, logit lens, cache)
    temperature: number; // sampling temperature for "Next token"
}

export interface IModelExample {
    name: string;
    sub?: string;            // one-line description shown under the model selector
    hidden?: boolean;        // not offered in the selector (GPT-3 is huge; kept for reference)
    shape: IModelShape;
    enabled: boolean;
    layout?: ILayoutBase;
    preset?: IModernConfig;  // set for Llama-style shape-only models
    blockRender: IBlockRender;
    offset: Vec3;
    modelCardOffset: Vec3;
    camera?: ICameraPos;
}

export interface IMouseState {
    mousePos: Vec3;
    inside: boolean; // false while the pointer is over a panel or outside the canvas
}

export interface IDisplayState {
    tokenColors: IColorMix | null;
    tokenIdxColors: IColorMix | null;
    tokenOutputColors: IColorMix | null;
    tokenIdxModelOpacity?: number[];
    topOutputOpacity?: number;
    lines: string[];
    hoverTarget: IHoverTarget | null;
    pinned: { cubeIdx: number, idx: Vec3 } | null; // cell pinned by a click (inspector + data flow stay on it)
    blkIdxHover: number[] | null;
    dimHover: DimStyle | null;
}

export interface IHoverTarget {
    subCube: IBlkDef;
    mainCube: IBlkDef;
    mainIdx: Vec3;
}

export function initProgramState(canvasEl: HTMLCanvasElement, fontAtlasData: IFontAtlasData): IProgramState {

    let render = initRender(canvasEl, fontAtlasData);
    let walkthrough = initWalkthrough();

    let prevState = SavedState.state;
    let camera: ICamera = {
        angle: prevState?.camera.angle ?? new Vec3(296, 16, 13.5),
        center: prevState?.camera.center ?? new Vec3(-8.4, 0, -481.5),
        transition: {},
        modelMtx: new Mat4f(),
        viewMtx: new Mat4f(),
        lookAtMtx: new Mat4f(),
        camPos: new Vec3(),
        camPosModel: new Vec3(),
    }

    let shape: IModelShape = {
        B: 1,
        T: 11,
        C: 48,
        nHeads: 3,
        A: 48 / 3,
        nBlocks: 3,
        vocabSize: 3,
    };

    let gpt2ShapeSmall: IModelShape = {
        B: 1,
        T: 1024,
        C: 768,
        nHeads: 12,
        A: 768 / 12,
        nBlocks: 12,
        vocabSize: 50257,
    };

    let gpt2ShapeLarge: IModelShape = {
        B: 1,
        T: 1024,
        C: 1600,
        nHeads: 25,
        A: 1600 / 25,
        nBlocks: 48,
        vocabSize: 50257,
    };

    let gpt3Shape: IModelShape = {
        B: 1,
        T: 1024,
        C: 12288,
        nHeads: 96,
        A: 12288 / 96,
        nBlocks: 96,
        vocabSize: 50257,
    };

    function makeCamera(center: Vec3, angle: Vec3): ICameraPos {
        return { center, angle };
    }

    let delta = new Vec3(10000, 0, 0);

    return {
        native: null,
        wasmGptModel: null,
        render: render!,
        inWalkthrough: true,
        walkthrough,
        camera,
        shape: shape,
        layout: genGptModelLayout(shape),
        currExampleId: -1,
        mainExample: {
            name: 'nano-gpt',
            enabled: true,
            shape: shape,
            offset: new Vec3(),
            modelCardOffset: new Vec3(),
            blockRender: null!,
            camera: makeCamera(new Vec3(42.771, 0.000, -569.287), new Vec3(284.959, 26.501, 12.867)),
        },
        examples: [{
            name: 'GPT-2 (small)',
            enabled: true,
            shape: gpt2ShapeSmall,
            offset: delta.mul(-5),
            modelCardOffset: delta.mul(-2.0),
            blockRender: initBlockRender(render?.ctx ?? null),
            camera: makeCamera(new Vec3(-65141.321, 0.000, -69843.439), new Vec3(224.459, 24.501, 1574.240)),
        }, {
            name: 'GPT-2 (XL)',
            enabled: true,
            shape: gpt2ShapeLarge,
            offset: delta.mul(20),
            modelCardOffset: delta.mul(0.5),
            blockRender: initBlockRender(render?.ctx ?? null),
            camera: makeCamera(new Vec3(237902.688, 0.000, -47282.484), new Vec3(311.959, 23.501, 1382.449)),
        }, {
            name: 'GPT-3',
            enabled: false,
            shape: gpt3Shape,
            offset: delta.mul(50.0),
            modelCardOffset: delta.mul(15.0),
            blockRender: initBlockRender(render?.ctx ?? null),
            camera: makeCamera(new Vec3(837678.163, 0.000, -485242.286), new Vec3(238.959, 10.501, 12583.939)),
        },
        // Llama-family models, shape only (no weights): placed further along x, cameras set when first shown
        ...archPresets.map((p, i): IModelExample => ({
            name: p.title,
            sub: p.sub + '. ' + paramLine(p.config),
            enabled: false,
            preset: p.config,
            shape: { B: 1, T: p.config.block_size, C: p.config.n_embd, nHeads: p.config.n_head, A: p.config.head_dim ?? p.config.n_embd / p.config.n_head, nBlocks: p.config.n_layer, vocabSize: p.config.vocab_size },
            offset: delta.mul(80 + i * 40),
            modelCardOffset: new Vec3(),
            blockRender: initBlockRender(render?.ctx ?? null),
        }))],
        gptGpuModel: null,
        jsGptModel: null,
        stepModel: false,
        markDirty: () => { },
        htmlSubs: new Subscriptions(),
        mouse: {
            mousePos: new Vec3(),
            inside: false,
        },
        movement: {
            action: null,
            actionHover: null,
            target: [0, 0],
            depth: 1,
            cameraLerp: null,
         },
        viz: defaultVizSettings(),
        modelMode: 'gpt2',
        modern: {},
        flow: { step: 0, layer: 0, token: 0, temperature: 1 },
        display: {
            tokenColors: null,
            tokenIdxColors: null,
            tokenOutputColors: null,
            lines: [],
            hoverTarget: null,
            pinned: null,
            dimHover: null,
            blkIdxHover: null,
        },
        pageLayout: {
            height: 0,
            width: 0,
            isDesktop: true,
            isPhone: true,
        }
    };
}

export function runProgram(view: IRenderView, state: IProgramState) {
    let timer0 = performance.now();

    if (!state.render) {
        return;
    }

    resetRenderBuffers(state.render);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    state.display.lines = [];
    state.display.hoverTarget = null;
    state.display.tokenColors = null;
    state.display.tokenIdxColors = null;

    if (state.wasmGptModel && state.jsGptModel) {
        syncWasmDataWithJsAndGpu(state.wasmGptModel, state.jsGptModel);
    }

    let modernRt = state.modelMode !== 'gpt2' ? state.modern[state.modelMode] ?? null : null;

    if (state.stepModel && modernRt) {
        state.stepModel = false;
        stepModern(modernRt, state.flow.temperature);
    } else if (state.stepModel && state.wasmGptModel && state.jsGptModel) {
        state.stepModel = false;
        stepWasmModel(state.wasmGptModel, state.jsGptModel);
    }

    // generate the base model, incorporating the live model data if available
    if (modernRt) {
        state.layout = genModernLayout(modernRt.model.config, modernRt);
    } else {
        state.layout = genGptModelLayout(state.shape, state.jsGptModel);
    }

    // real value ranges per block (legend) and, in 'auto' mode, honest per-block colour scales
    applyValueRanges(state.layout, state.viz, new Set([state.layout.idxObj]));
    state.render.blockRender.palette = state.viz.palette === 'diverging' ? 1 : 0;

    // @TODO: handle different models in the same scene.
    // Maybe need to copy a lot of different things like the entire render state per model?
    for (let example of state.examples) {
        if (example.enabled && !example.layout) {
            example.layout = example.preset
                ? genModernLayout(example.preset, null, new Vec3(), 8)
                : genGptModelLayout(example.shape, null, example.offset);
        }
    }

    genModelViewMatrices(state, state.layout!);

    let queryRes = beginQueryAndGetPrevMs(state.render.queryManager, 'render');
    if (isNotNil(queryRes)) {
        state.render.lastGpuMs = queryRes;
    }

    state.render.renderTiming = false; // state.pageLayout.isDesktop;

    // will modify layout; view; render a few things.
    if (state.inWalkthrough && state.layout.kind === 'gpt2') {
        runWalkthrough(state, view);
    }

    updateCamera(state, view);

    // these will get modified by the walkthrough (stored where?)
    if (state.layout.kind === 'gpt2') {
        drawAllArrows(state.render, gptLayout(state));
    }

    drawModelCard(state, state.layout, modelModeInfo[state.modelMode].title, new Vec3());
    // drawTokens(state.render, state.layout, state.display);

    for (let example of state.examples) {
        if (example.enabled && example.layout) {
            drawModelCard(state, example.layout, example.name, example.offset.add(example.modelCardOffset), !!example.preset);
        }
    }

    // manageMovement(state, view);
    runMouseHitTesting(state);
    if (state.viz.focus) {
        let hover = state.display.hoverTarget as IHoverTarget | null; // set by runMouseHitTesting
        let pinnedCube = state.display.pinned ? state.layout.cubes[state.display.pinned.cubeIdx] ?? null : null;
        let target = hover?.mainCube ?? pinnedCube;
        applyFocus(state.layout, target ?? (flowActive(state) ? flowFocusBlocks(state) : null));
    }
    drawBlockInfo(state); // after focus, so dimmed blocks get no name tag
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    if (state.layout.kind === 'gpt2') {
        drawBlockLabels(state.render, gptLayout(state));
    } else {
        drawModernLabels(state.render, state.layout as IModernLayout);
    }

    let lineNo = 1;
    let tw = state.render.size.x;
    state.render.sharedRender.activePhase = RenderPhase.Overlay2D;
    for (let line of state.display.lines) {
        let opts: IFontOpts = { color: new Vec4(), size: 14 };
        let w = measureText(state.render.modelFontBuf, line, opts);
        drawText(state.render.modelFontBuf, line, tw - w - 4, lineNo * opts.size * 1.3 + 4, opts)
        lineNo++;
    }

    // render everything; i.e. here's where we actually do gl draw calls
    // up until now, we've just been putting data in cpu-side buffers
    renderModel(state);

    endQuery(state.render.queryManager, 'render');
    state.render.gl.flush();

    state.render.lastJsMs = performance.now() - timer0;
}

/** The GPT-2 layout. Only valid in Classic mode (walkthrough, arrows, section labels). */
export function gptLayout(state: IProgramState): IGptModelLayout {
    if (state.layout.kind !== 'gpt2') {
        throw new Error('gptLayout() called while a non-GPT-2 model is shown');
    }
    return state.layout as IGptModelLayout;
}
