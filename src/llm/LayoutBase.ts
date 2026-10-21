import { IBlkDef, IBlkLabel, IModelLayout } from "./GptModelLayout";
import { IModelShape } from "./GptModel";
import { IBufferTex } from "@/src/utils/renderPhases";

/** What the running model exposes to shared UI (model card, token boxes). */
export interface IModelRuntimeLink {
    inputTokens: IBufferTex;
    inputLen: number;
    sortedBuf: Float32Array | null;
}

/** The part of a model layout that shared code (render, hover, data flow, model card) may rely on. */
export interface ILayoutBase extends IModelLayout {
    kind: 'gpt2' | 'modern';
    shape: IModelShape;
    labels: IBlkLabel[];
    weightCount: number;
    idxObj: IBlkDef;
    tokEmbedObj: IBlkDef;
    posEmbedObj: IBlkDef | null; // null for RoPE models (no learned positions)
    residual0: IBlkDef;
    logitsSoftmax: IBlkDef;
    model: IModelRuntimeLink | null;
    /** Softmax attention block per [layer][head]; cell (x = key token, y = query token) = probability. */
    attnHeads: IBlkDef[][];
    /** Text for a token id. */
    tokenText: (id: number) => string;
}
