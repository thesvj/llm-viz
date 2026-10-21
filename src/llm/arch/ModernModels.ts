import { assetUrl } from "@/src/utils/basePath";
import { createModernRuntime, IModernRuntime } from "./ModernRuntime";

export type IModelMode = 'gpt2' | 'llama' | 'moe';

export const modelModeInfo: Record<IModelMode, { title: string, sub: string }> = {
    gpt2: { title: 'nano-gpt', sub: 'GPT-2 style: LayerNorm, learned positions, GELU. 85,584 parameters.' },
    llama: { title: 'tiny Llama', sub: 'Llama style: RMS norm, RoPE, grouped-query attention, SwiGLU. 76,656 parameters.' },
    moe: { title: 'tiny MoE', sub: 'Llama style with a mixture of experts: 4 experts per layer, each token uses 2. 132,528 parameters.' },
};

/** Fetch the tiny modern models (made by tools/train_tiny_modern.py) and start them on the GPU context. */
export async function loadModernModels(gl: WebGL2RenderingContext): Promise<Partial<Record<IModelMode, IModernRuntime>>> {
    let out: Partial<Record<IModelMode, IModernRuntime>> = {};
    for (let name of ['llama', 'moe'] as const) {
        let resp = await fetch(assetUrl(`tiny-${name}-model.json`));
        if (!resp.ok) {
            console.warn(`could not load tiny-${name}-model.json (${resp.status})`);
            continue;
        }
        out[name] = createModernRuntime(gl, name, await resp.json());
    }
    return out;
}
