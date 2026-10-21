import { createModernRuntime, IModernRuntime } from "./ModernRuntime";

export type IModelMode = 'gpt2' | 'llama' | 'moe';

export const modelModeInfo: Record<IModelMode, { title: string, sub: string }> = {
    gpt2: { title: 'nano-gpt', sub: 'GPT-2 style: LayerNorm, learned positions, GELU' },
    llama: { title: 'tiny-llama', sub: 'Llama style: RMSNorm, RoPE, grouped-query attention, SwiGLU' },
    moe: { title: 'tiny-moe', sub: 'Mixture of Experts: 4 experts, each token uses the top 2' },
};

/** Fetch the tiny modern models (made by tools/train_tiny_modern.py) and start them on the GPU context. */
export async function loadModernModels(gl: WebGL2RenderingContext): Promise<Partial<Record<IModelMode, IModernRuntime>>> {
    let out: Partial<Record<IModelMode, IModernRuntime>> = {};
    for (let name of ['llama', 'moe'] as const) {
        let resp = await fetch(`/tiny-${name}-model.json`);
        if (!resp.ok) {
            console.warn(`could not load tiny-${name}-model.json (${resp.status})`);
            continue;
        }
        out[name] = createModernRuntime(gl, name, await resp.json());
    }
    return out;
}
