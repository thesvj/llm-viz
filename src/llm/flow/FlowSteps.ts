// The step-by-step guide for the modern (Llama-style and MoE) models.
// Each step names the blocks it is about; the guide flies the camera to them and dims the rest.
// Text uses the shown model's real sizes, so every number on screen matches a block.

import { IBlkDef } from "../GptModelLayout";
import { IModernLayout } from "../arch/ModernLayout";

export type IFlowPanel = 'rope' | 'tokens' | 'residual' | 'gqa' | 'mask' | 'attention' | 'moe' | 'output' | 'cache';

export interface IFlowStep {
    id: string;
    group: string;
    title: string;
    blocks: IBlkDef[];
    /** Plain-words explanation, one short paragraph per entry. */
    text: string[];
    /** Shape line, e.g. "11 tokens × 48 channels". */
    shape?: string;
    /** Extra live panel shown under the text. */
    panel?: IFlowPanel;
    /** How this differs from GPT-2, when it does. */
    vsGpt2?: string;
}

export function buildFlowSteps(layout: IModernLayout, layer: number): IFlowStep[] {
    let c = layout.config;
    let T = c.block_size, C = c.n_embd, nH = c.n_head, nKV = c.n_kv_head, A = C / nH, H = c.n_hidden;
    let V = c.vocab_size, L = c.n_layer, E = c.n_expert ?? 0, K = c.top_k ?? 0;
    let rep = nH / nKV;
    let b = layout.blocks[layer];
    let isMoe = E > 0;
    let lyr = `Layer ${layer + 1} of ${L}`;

    let steps: IFlowStep[] = [];

    // ---- input ----
    steps.push({
        id: 'tokens', group: 'Input', title: 'Tokens',
        blocks: [layout.idxObj],
        shape: `${T} positions`,
        text: [
            `The model reads a short list of letters and has to output them in sorted order. Each letter is a token, stored as a number: A = 0, B = 1, C = 2.`,
            `The model only ever sees these numbers. Real models split text into pieces from a list of 30,000 to 150,000 entries, but the idea is the same.`,
        ],
        panel: 'tokens',
    });
    steps.push({
        id: 'embed', group: 'Input', title: 'Token embedding',
        blocks: [layout.idxObj, layout.tokEmbedObj, layout.residual0],
        shape: `${T} tokens × ${C} numbers`,
        text: [
            `Each token id picks one column of the embedding table (${V} columns of ${C} numbers). That column becomes the token's starting vector.`,
            `Nothing here says where a token sits in the list. Position is added later, inside attention, by RoPE.`,
        ],
        vsGpt2: `GPT-2 adds a second, learned position table at this point. Llama-style models have no position table.`,
    });
    steps.push({
        id: 'residual', group: 'Input', title: 'The residual stream',
        blocks: layout.residuals,
        shape: `${layout.residuals.length} snapshots of ${T} × ${C}`,
        text: [
            `The ${T} × ${C} block of numbers in the centre column is the residual stream. Every part of every layer reads it and adds its result back onto it.`,
            `The panel below shows what the model would predict if it stopped after each snapshot ("logit lens"). The answer usually gets clearer layer by layer.`,
        ],
        panel: 'residual',
    });

    // ---- attention ----
    let ga = `${lyr} · Attention`;
    steps.push({
        id: 'norm1', group: ga, title: 'RMS norm',
        blocks: [b.norm1.rms, b.norm1.gamma, b.norm1.out],
        shape: `${T} × ${C}`,
        text: [
            `Before attention, each token's vector is rescaled to a standard size: divide by its root-mean-square (the "rms" row), then multiply each channel by a learned weight γ.`,
            `This keeps the numbers in a steady range as the layers stack up.`,
        ],
        vsGpt2: `GPT-2 uses LayerNorm, which also subtracts the mean and adds a bias. RMS norm skips both and is cheaper.`,
    });
    steps.push({
        id: 'qkv', group: ga, title: 'Queries, keys and values',
        blocks: [...b.heads.flatMap(h => [h.wq, h.q]), ...b.kv.flatMap(k => [k.wk, k.k, k.wv, k.v])],
        shape: `each ${T} × ${A}`,
        text: [
            `Three weight matrices turn each token vector into a query (what this token is looking for), a key (what it offers) and a value (what it passes on). Each is ${A} numbers per head.`,
            `This model has ${nH} query heads but only ${nKV} key/value heads. Each key/value head is shared by ${rep} query heads. This is grouped-query attention (GQA).`,
        ],
        panel: 'gqa',
        vsGpt2: `GPT-2 gives every head its own key and value. Sharing them makes the memory kept during generation (the KV cache) ${rep}× smaller.`,
    });
    steps.push({
        id: 'rope', group: ga, title: 'RoPE: adding position',
        blocks: [b.cos, b.sin, ...b.heads.flatMap(h => [h.q, h.qRope]), ...b.kv.flatMap(k => [k.k, k.kRope])],
        shape: `${T} × ${A}`,
        text: [
            `RoPE turns each pair of channels (j and j + ${A / 2}) of every query and key by an angle. The angle grows with the token's position, so the cos and sin tables change from column to column.`,
            `Because both query and key are turned, their dot product depends on how far apart the two tokens are. That is how the model knows order.`,
        ],
        panel: 'rope',
        vsGpt2: `GPT-2 adds position once, at the input. RoPE adds it in every layer, and only to queries and keys, not values.`,
    });
    steps.push({
        id: 'scores', group: ga, title: 'Scores and the causal mask',
        blocks: b.heads.flatMap(h => [h.qRope, h.scores]).concat(b.kv.map(k => k.kRope)),
        shape: `${T} × ${T} per head`,
        text: [
            `Each score is a query · key dot product, divided by √${A}. A high score means "this token is relevant to me".`,
            `A token may only look at itself and earlier tokens. Scores for later tokens (the empty upper triangle) are set to −∞, so they become exactly 0 after softmax.`,
        ],
        panel: 'mask',
    });
    steps.push({
        id: 'softmax', group: ga, title: 'Attention weights',
        blocks: b.heads.flatMap(h => [h.scores, h.probs]),
        shape: `${T} × ${T} per head, rows sum to 1`,
        text: [
            `Softmax turns each row of scores into weights that are positive and add up to 1. Row t says how much token t takes from each earlier token.`,
            `The attention maps below show the same numbers as heat maps, with the letters on both axes.`,
        ],
        panel: 'attention',
    });
    steps.push({
        id: 'mixv', group: ga, title: 'Mixing the values',
        blocks: [...b.heads.flatMap(h => [h.probs, h.out]), ...b.kv.map(k => k.v)],
        shape: `${T} × ${A} per head`,
        text: [
            `Each head's output for token t is the weighted sum of the value vectors, using row t of its attention weights.`,
            `Heads in the same group read the same shared value block, but each uses its own weights, so their outputs differ.`,
        ],
    });
    steps.push({
        id: 'attnout', group: ga, title: 'Output projection and residual add',
        blocks: [b.concat, b.wo, b.attnOut, b.resid1],
        shape: `${T} × ${C}`,
        text: [
            `The ${nH} head outputs are placed side by side (${nH} × ${A} = ${C} numbers) and mixed by one more weight matrix.`,
            `The result is added onto the residual stream. That sum is the "Attention Residual" block in the centre column.`,
        ],
    });

    // ---- feed-forward ----
    let gf = `${lyr} · ${isMoe ? 'Mixture of experts' : 'Feed-forward'}`;
    steps.push({
        id: 'norm2', group: gf, title: 'RMS norm',
        blocks: [b.norm2.rms, b.norm2.gamma, b.norm2.out],
        shape: `${T} × ${C}`,
        text: [`The same rescaling as before attention, with its own γ weights.`],
    });
    if (b.mlp) {
        let m = b.mlp;
        steps.push({
            id: 'gateup', group: gf, title: 'Gate and up projections',
            blocks: [m.wGate, m.gate, m.wUp, m.up],
            shape: `${T} × ${H}`,
            text: [
                `Two separate weight matrices widen each token from ${C} to ${H} numbers: the "gate" and the "up" projection.`,
                `Each token is handled on its own here. Tokens only exchange information in attention.`,
            ],
        });
        steps.push({
            id: 'swiglu', group: gf, title: 'SwiGLU: gate × up',
            blocks: [m.gate, m.up, m.act],
            shape: `${T} × ${H}`,
            text: [
                `The gate goes through SiLU (x · sigmoid(x)), which keeps positive values and squashes negative ones towards 0. It is then multiplied, number by number, with the up projection.`,
                `So the gate decides how much of each "up" channel gets through.`,
            ],
            vsGpt2: `GPT-2 uses one projection and GELU. SwiGLU uses two projections and a multiply, which trains better for the same cost.`,
        });
        steps.push({
            id: 'down', group: gf, title: 'Down projection and residual add',
            blocks: [m.act, m.wDown, m.down, b.resid2],
            shape: `${T} × ${C}`,
            text: [
                `A third matrix shrinks the ${H} numbers back to ${C}. The result is added onto the residual stream ("MLP Residual").`,
            ],
        });
    } else if (b.moe) {
        let m = b.moe;
        steps.push({
            id: 'router', group: gf, title: 'Router',
            blocks: [m.wRouter, m.logits, m.probs],
            shape: `${T} × ${E}`,
            text: [
                `This layer has ${E} separate feed-forward networks, called experts. A small router matrix gives each token one score per expert, and softmax turns the scores into probabilities.`,
            ],
            panel: 'moe',
        });
        steps.push({
            id: 'topk', group: gf, title: `Pick the top ${K} experts`,
            blocks: [m.probs, m.weights],
            shape: `${T} × ${E}, ${K} non-zero per token`,
            text: [
                `Each token keeps only its ${K} highest-scoring experts. Their probabilities are rescaled to add up to 1, and the other ${E - K} experts get weight 0.`,
                `So each token runs only ${K} of the ${E} experts. The model has many weights but uses only some of them per token.`,
            ],
            panel: 'moe',
        });
        steps.push({
            id: 'experts', group: gf, title: 'The experts',
            blocks: m.experts.flatMap(ex => [ex.wGate, ex.gate, ex.wUp, ex.up, ex.act, ex.wDown, ex.down]),
            shape: `${E} experts, each ${C} → ${H} → ${C}`,
            text: [
                `Each expert is a small SwiGLU network: gate and up projections, SiLU(gate) × up, then a down projection.`,
                `For clarity every expert is computed here for every token. A real MoE model runs each token only through its chosen experts.`,
            ],
            panel: 'moe',
        });
        steps.push({
            id: 'combine', group: gf, title: 'Combine and residual add',
            blocks: [m.weights, ...m.experts.map(ex => ex.down), m.out, b.resid2],
            shape: `${T} × ${C}`,
            text: [
                `For each token: output = w₁ · (first chosen expert) + w₂ · (second chosen expert). The other experts are multiplied by 0.`,
                `The result is added onto the residual stream ("MoE Residual").`,
            ],
            panel: 'moe',
        });
    }

    steps.push({
        id: 'layers', group: 'All layers', title: `The other ${L - 1} layers`,
        blocks: layout.blocks.filter((_, i) => i !== layer).map(bl => bl.resid2),
        text: [
            `Every layer repeats the same steps with its own weights. Pick a layer below to walk through it.`,
            `Real models stack many more: Llama 3.2 1B has 16 layers, Qwen3 8B has 36.`,
        ],
    });

    // ---- output ----
    steps.push({
        id: 'normf', group: 'Output', title: 'Final RMS norm',
        blocks: [layout.normF.rms, layout.normF.gamma, layout.normF.out],
        shape: `${T} × ${C}`,
        text: [`One last rescaling of the residual stream before the output.`],
    });
    steps.push({
        id: 'logits', group: 'Output', title: 'Logits',
        blocks: [layout.lmHead, layout.logits],
        shape: `${T} × ${V}`,
        text: [
            `The LM head matrix turns each token's ${C} numbers into ${V} scores, one per possible next token. These raw scores are called logits.`,
        ],
    });
    steps.push({
        id: 'probs', group: 'Output', title: 'Next-token probabilities',
        blocks: [layout.logits, layout.logitsSoftmax],
        shape: `${T} × ${V}`,
        text: [
            `Softmax turns the logits into probabilities. Column t is the model's guess for the token after position t.`,
            `Temperature divides the logits before softmax. Below 1 makes the top choice stronger, above 1 flattens the guesses. Try the slider.`,
        ],
        panel: 'output',
    });
    steps.push({
        id: 'generate', group: 'Output', title: 'Generating, and the KV cache',
        blocks: [layout.idxObj, layout.logitsSoftmax],
        text: [
            `To generate, the model appends its chosen token and runs again. Click "Next token" to watch the list grow.`,
            `Keys and values of earlier tokens never change, because tokens only look backwards. Real models store them (the KV cache) and compute only the newest column on each step.`,
        ],
        panel: 'cache',
    });

    return steps;
}
