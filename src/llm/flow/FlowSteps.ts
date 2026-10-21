// The step-by-step guide for the modern (Llama-style and MoE) models.
// Each step names the blocks it is about; the guide flies the camera to them and dims the rest.
// Text uses the shown model's real sizes, so every number on screen matches a block.

import { IBlkDef } from "../GptModelLayout";
import { IModernLayout } from "../arch/ModernLayout";

export type IFlowPanel = 'rope' | 'tokens' | 'residual' | 'gqa' | 'mask' | 'attention' | 'moe' | 'output' | 'cache';

/** A piece of step text: plain words, or words linked to blocks (hover lights them up, click flies there). */
export type FlowPart = string | { s: string, blks: IBlkDef[] };
/** One paragraph: a plain string, or a list of parts. */
export type FlowText = string | FlowPart[];

/** Link some words to one or more blocks. */
export const ref = (s: string, ...blks: (IBlkDef | IBlkDef[] | null | undefined)[]): FlowPart =>
    ({ s, blks: blks.flat().filter((b): b is IBlkDef => !!b) });

/** Tagged template: tx`text ${ref('words', blk)} more text ${n}` gives a paragraph with links. */
export function tx(strs: TemplateStringsArray, ...vals: (FlowPart | number)[]): FlowPart[] {
    let out: FlowPart[] = [];
    strs.forEach((str, i) => {
        if (str) out.push(str);
        if (i < vals.length) {
            let v = vals[i];
            out.push(typeof v === 'number' ? String(v) : v);
        }
    });
    return out;
}

export interface IFlowStep {
    id: string;
    group: string;
    title: string;
    blocks: IBlkDef[];
    /** Plain-words explanation, one short paragraph per entry. */
    text: FlowText[];
    /** Shape line, e.g. "11 tokens × 48 channels". */
    shape?: string;
    /** Extra live panel shown under the text. */
    panel?: IFlowPanel;
    /** How this differs from GPT-2, when it does. */
    vsGpt2?: string;
    /** What newer real models do differently from the tiny model shown. */
    newer?: string;
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
            tx`The model reads a short list of letters and has to output them in sorted order. Each letter is a ${ref('token', layout.idxObj)}, stored as a number: A = 0, B = 1, C = 2.`,
            `Change the letters in the panel below and every block in the 3D view updates.`,
            `The model only ever sees these numbers. Real models split text into pieces from a fixed list: 32,000 entries in Mistral 7B, about 262,000 in Gemma 3. The idea is the same.`,
        ],
        panel: 'tokens',
    });
    steps.push({
        id: 'embed', group: 'Input', title: 'Token embedding',
        blocks: [layout.idxObj, layout.tokEmbedObj, layout.residual0],
        shape: `${T} tokens × ${C} numbers`,
        text: [
            tx`Each ${ref('token id', layout.idxObj)} picks one column of the ${ref('embedding table', layout.tokEmbedObj)} (${V} columns of ${C} numbers). That column becomes the token's ${ref('starting vector', layout.residual0)}.`,
            `Nothing here says where a token sits in the list. Position is added later, inside attention, by RoPE.`,
        ],
        vsGpt2: `GPT-2 adds a second, learned position table at this point. Llama-style models have no position table.`,
    });
    steps.push({
        id: 'residual', group: 'Input', title: 'The residual stream',
        blocks: [...layout.residuals, ...layout.lens],
        shape: `${layout.residuals.length} snapshots of ${T} × ${C}`,
        text: [
            tx`The ${T} × ${C} block of numbers in the centre column is the ${ref('residual stream', layout.residuals)}. Every part of every layer reads it and adds its result back onto it.`,
            tx`The small ${T} × ${V} block to the right of each snapshot is a ${ref('"logit lens"', layout.lens)}: the guess the model would make if it stopped there. The panel below lists the same guesses; they usually get clearer layer by layer.`,
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
            tx`Before attention, each ${ref("token's vector", layer === 0 ? layout.residual0 : layout.blocks[layer - 1].resid2)} is rescaled to a standard size: divide by its ${ref('root-mean-square (the "rms" row)', b.norm1.rms)}, then multiply each channel by a ${ref('learned weight γ', b.norm1.gamma)}. The result is the ${ref('normalized input', b.norm1.out)}.`,
            `This keeps the numbers in a steady range as the layers stack up.`,
        ],
        vsGpt2: `GPT-2 uses LayerNorm, which also subtracts the mean and adds a bias. RMS norm skips both and is cheaper.`,
    });
    steps.push({
        id: 'qkv', group: ga, title: 'Queries, keys and values',
        blocks: [...b.heads.flatMap(h => [h.wq, h.q]), ...b.kv.flatMap(k => [k.wk, k.k, k.wv, k.v])],
        shape: `each ${T} × ${A}`,
        text: [
            tx`Three ${ref('weight matrices', b.heads.map(h => h.wq), b.kv.flatMap(k => [k.wk, k.wv]))} turn each token vector into a ${ref('query', b.heads.map(h => h.q))} (what this token is looking for), a ${ref('key', b.kv.map(k => k.k))} (what it offers) and a ${ref('value', b.kv.map(k => k.v))} (what it passes on). Each is ${A} numbers per head.`,
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
            tx`RoPE turns each pair of channels (j and j + ${A / 2}) of every ${ref('query', b.heads.map(h => h.qRope))} and ${ref('key', b.kv.map(k => k.kRope))} by an angle. The angle grows with the token's position, so the ${ref('cos', b.cos)} and ${ref('sin', b.sin)} tables change from column to column.`,
            `Because both query and key are turned, their dot product depends on how far apart the two tokens are. That is how the model knows order.`,
            `Here channel j is paired with channel j + ${A / 2}, as in the Hugging Face code. Meta's original Llama code pairs neighbouring channels (0 with 1, 2 with 3). Both give the same model; only the channel order differs.`,
        ],
        panel: 'rope',
        vsGpt2: `GPT-2 adds position once, at the input. RoPE adds it in every layer, and only to queries and keys, not values.`,
    });
    steps.push({
        id: 'scores', group: ga, title: 'Scores and the causal mask',
        blocks: b.heads.flatMap(h => [h.qRope, h.scores]).concat(b.kv.map(k => k.kRope)),
        shape: `${T} × ${T} per head`,
        text: [
            tx`Each ${ref('score', b.heads.map(h => h.scores))} is a ${ref('query', b.heads.map(h => h.qRope))} · ${ref('key', b.kv.map(k => k.kRope))} dot product, divided by √${A}. A high score means "this token is relevant to me".`,
            `A token may only look at itself and earlier tokens. Scores for later tokens (the empty upper triangle) are set to −∞, so they become exactly 0 after softmax.`,
        ],
        panel: 'mask',
        newer: `Many newer models mix two kinds of attention layer. A local layer lets each token see only recent tokens: 128 in gpt-oss, 1,024 in Gemma 3. A full layer sees every earlier token, as here. gpt-oss alternates them one to one, and Gemma 3 uses five local layers per full one. This keeps memory small for long texts.`,
    });
    steps.push({
        id: 'softmax', group: ga, title: 'Attention weights',
        blocks: b.heads.flatMap(h => [h.scores, h.probs]),
        shape: `${T} × ${T} per head, rows sum to 1`,
        text: [
            tx`Softmax turns each row of ${ref('scores', b.heads.map(h => h.scores))} into ${ref('weights', b.heads.map(h => h.probs))} that are positive and add up to 1. Row t says how much token t takes from each earlier token.`,
            `The attention maps below show the same numbers as heat maps, with the letters on both axes.`,
        ],
        panel: 'attention',
    });
    steps.push({
        id: 'mixv', group: ga, title: 'Mixing the values',
        blocks: [...b.heads.flatMap(h => [h.probs, h.out]), ...b.kv.map(k => k.v)],
        shape: `${T} × ${A} per head`,
        text: [
            tx`Each ${ref("head's output", b.heads.map(h => h.out))} for token t is the weighted sum of the ${ref('value vectors', b.kv.map(k => k.v))}, using row t of its ${ref('attention weights', b.heads.map(h => h.probs))}.`,
            `Heads in the same group read the same shared value block, but each uses its own weights, so their outputs differ.`,
        ],
    });
    steps.push({
        id: 'attnout', group: ga, title: 'Output projection and residual add',
        blocks: [b.concat, b.wo, b.attnOut, b.resid1],
        shape: `${T} × ${C}`,
        text: [
            tx`The ${nH} head outputs are ${ref('placed side by side', b.concat)} (${nH} × ${A} = ${C} numbers) and mixed by ${ref('one more weight matrix', b.wo)}.`,
            tx`The ${ref('result', b.attnOut)} is added onto the residual stream. That sum is the ${ref('"Attention Residual"', b.resid1)} block in the centre column.`,
        ],
    });

    // ---- feed-forward ----
    let gf = `${lyr} · ${isMoe ? 'Mixture of experts' : 'Feed-forward'}`;
    steps.push({
        id: 'norm2', group: gf, title: 'RMS norm',
        blocks: [b.norm2.rms, b.norm2.gamma, b.norm2.out],
        shape: `${T} × ${C}`,
        text: [tx`The same rescaling as before attention, with its own ${ref('γ weights', b.norm2.gamma)}.`],
    });
    if (b.mlp) {
        let m = b.mlp;
        steps.push({
            id: 'gateup', group: gf, title: 'Gate and up projections',
            blocks: [m.wGate, m.gate, m.wUp, m.up],
            shape: `${T} × ${H}`,
            text: [
                tx`Two separate weight matrices widen each token from ${C} to ${H} numbers: the ${ref('"gate"', m.gate)} and the ${ref('"up"', m.up)} projection.`,
                `Each token is handled on its own here. Tokens only exchange information in attention.`,
            ],
        });
        steps.push({
            id: 'swiglu', group: gf, title: 'SwiGLU: gate × up',
            blocks: [m.gate, m.up, m.act],
            shape: `${T} × ${H}`,
            text: [
                tx`The ${ref('gate', m.gate)} goes through SiLU (x · sigmoid(x)), which keeps positive values and squashes negative ones towards 0. It is then multiplied, number by number, with the ${ref('up projection', m.up)}, giving the ${ref('SwiGLU block', m.act)}.`,
                `So the gate decides how much of each "up" channel gets through.`,
            ],
            vsGpt2: `GPT-2 uses one projection and GELU. SwiGLU uses two projections and a multiply, which trains better for the same cost.`,
        });
        steps.push({
            id: 'down', group: gf, title: 'Down projection and residual add',
            blocks: [m.act, m.wDown, m.down, b.resid2],
            shape: `${T} × ${C}`,
            text: [
                tx`A ${ref('third matrix', m.wDown)} shrinks the ${H} numbers back to ${C}. The ${ref('result', m.down)} is added onto the residual stream (${ref('"MLP Residual"', b.resid2)}).`,
            ],
        });
    } else if (b.moe) {
        let m = b.moe;
        steps.push({
            id: 'router', group: gf, title: 'Router',
            blocks: [m.wRouter, m.logits, m.probs],
            shape: `${T} × ${E}`,
            text: [
                tx`This layer has ${E} separate feed-forward networks, called ${ref('experts', m.experts.flatMap(ex => [ex.gate, ex.up, ex.act, ex.down]))}. A small ${ref('router matrix', m.wRouter)} gives each token ${ref('one score per expert', m.logits)}, and softmax turns the scores into ${ref('probabilities', m.probs)}.`,
            ],
            panel: 'moe',
            newer: `Newer MoE models use many small experts. DeepSeek-V3 has 256 per layer and each token uses 8. It also has one shared expert that every token always uses. gpt-oss-20b has 32 experts and uses 4. The tiny model here follows the older Mixtral style: a few large experts and no shared expert.`,
        });
        steps.push({
            id: 'topk', group: gf, title: `Pick the top ${K} experts`,
            blocks: [m.probs, m.weights],
            shape: `${T} × ${E}, ${K} non-zero per token`,
            text: [
                tx`Each token keeps only its ${K} highest-scoring experts. Their ${ref('probabilities', m.probs)} are ${ref('rescaled to add up to 1', m.weights)}, and the other ${E - K} experts get weight 0.`,
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
                tx`For each token: ${ref('output', m.out)} = ${ref('w₁', m.weights)} · (first chosen expert) + ${ref('w₂', m.weights)} · (second chosen expert). The other experts are multiplied by 0.`,
                tx`The result is added onto the residual stream (${ref('"MoE Residual"', b.resid2)}).`,
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
        text: [tx`One last ${ref('rescaling', layout.normF.out)} of the ${ref('residual stream', layout.residuals[layout.residuals.length - 1])} before the output.`],
    });
    steps.push({
        id: 'logits', group: 'Output', title: 'Logits',
        blocks: [layout.lmHead, layout.logits],
        shape: `${T} × ${V}`,
        text: [
            tx`The ${ref('LM head matrix', layout.lmHead)} turns each token's ${ref(`${C} numbers`, layout.normF.out)} into ${V} scores, one per possible next token. These raw scores are called ${ref('logits', layout.logits)}.`,
        ],
    });
    steps.push({
        id: 'probs', group: 'Output', title: 'Next-token probabilities',
        blocks: [layout.logits, layout.logitsSoftmax],
        shape: `${T} × ${V}`,
        text: [
            tx`Softmax turns the ${ref('logits', layout.logits)} into ${ref('probabilities', layout.logitsSoftmax)}. Column t is the model's guess for the token after position t.`,
            `Temperature divides the logits before softmax. Below 1 makes the top choice stronger, above 1 flattens the guesses. Try the slider.`,
        ],
        panel: 'output',
    });
    steps.push({
        id: 'generate', group: 'Output', title: 'Generating, and the KV cache',
        blocks: b.kv.flatMap(k => [k.kRope, k.v]),
        text: [
            `To generate, the model appends its chosen token and runs again. Click "Next token" to watch the list grow.`,
            `Keys and values of earlier tokens never change, because tokens only look backwards. Real models store them (the KV cache) and compute only the newest column on each step.`,
            `The 3D view shows this layer's keys and values. After a step, the reused columns are outlined in blue ("cached") and the one new column in orange ("new").`,
        ],
        panel: 'cache',
        newer: `DeepSeek-V3 shrinks this cache further with multi-head latent attention. It stores one short compressed vector per token, plus a small part for RoPE, and expands it back into keys and values when needed.`,
    });

    return steps;
}
