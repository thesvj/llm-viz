// Real model shapes, shown without weights (shape only), like GPT-3 in the original visualizer.
// Numbers come from each model's published config.json on Hugging Face.

import { IModernConfig } from "./ModernForward";

export interface IArchPreset {
    id: string;
    title: string;
    sub: string;          // one line: what is special about it
    config: IModernConfig;
    published: { total: number, active?: number }; // published parameter counts
}

function cfg(c: Partial<IModernConfig> & Pick<IModernConfig, 'vocab_size' | 'n_embd' | 'n_layer' | 'n_head' | 'n_kv_head' | 'n_hidden'>): IModernConfig {
    return {
        block_size: 1024, // shown context length; the real models allow far more
        rope_theta: 10000, norm_eps: 1e-5, arch: c.n_expert ? 'moe' : 'llama',
        ...c,
    };
}

export const archPresets: IArchPreset[] = [{
    id: 'llama32-1b',
    title: 'Llama 3.2 1B',
    sub: '16 layers, 32 query heads share 8 K/V heads, input and output embeddings tied',
    config: cfg({ vocab_size: 128256, n_embd: 2048, n_layer: 16, n_head: 32, n_kv_head: 8, n_hidden: 8192, rope_theta: 500000, tied: true }),
    published: { total: 1.236e9 },
}, {
    id: 'qwen3-8b',
    title: 'Qwen3 8B',
    sub: '36 layers, 32 query heads share 8 K/V heads, extra RMS norm on Q and K',
    config: cfg({ vocab_size: 151936, n_embd: 4096, n_layer: 36, n_head: 32, n_kv_head: 8, n_hidden: 12288, rope_theta: 1000000, norm_eps: 1e-6, qk_norm: true }),
    published: { total: 8.19e9 },
}, {
    id: 'mistral-7b',
    title: 'Mistral 7B',
    sub: '32 layers, 32 query heads share 8 K/V heads, each token sees only the last 4096 tokens',
    config: cfg({ vocab_size: 32000, n_embd: 4096, n_layer: 32, n_head: 32, n_kv_head: 8, n_hidden: 14336, sliding_window: 4096 }),
    published: { total: 7.24e9 },
}, {
    id: 'mixtral-8x7b',
    title: 'Mixtral 8x7B',
    sub: '32 layers, 8 experts per layer, each token uses 2 of them',
    config: cfg({ vocab_size: 32000, n_embd: 4096, n_layer: 32, n_head: 32, n_kv_head: 8, n_hidden: 14336, n_expert: 8, top_k: 2, rope_theta: 1000000 }),
    published: { total: 46.7e9, active: 12.9e9 },
}];

/** Parameter count of a Llama-style model. `active` counts only the top-k experts each token uses. */
export function countParams(c: IModernConfig): { total: number, active: number } {
    let C = c.n_embd, A = c.head_dim ?? C / c.n_head, H = c.n_hidden, V = c.vocab_size;
    let attn = C * c.n_head * A + 2 * C * c.n_kv_head * A + c.n_head * A * C + (c.qk_norm ? 2 * A : 0);
    let norms = 2 * C;
    let expert = 3 * C * H; // gate, up, down
    let E = c.n_expert ?? 0, K = c.top_k ?? 0;
    let mlpTotal = E > 0 ? E * expert + C * E : expert;
    let mlpActive = E > 0 ? K * expert + C * E : expert;
    let embed = V * C * (c.tied ? 1 : 2) + C; // + final norm
    return {
        total: embed + c.n_layer * (attn + norms + mlpTotal),
        active: embed + c.n_layer * (attn + norms + mlpActive),
    };
}
