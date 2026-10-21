# LLM visualization

A 3D view of small language models running in the browser. You can follow every number from the input tokens to the next-token guess.

**Live site:** https://thesvj.github.io/llm-viz/llm

## Models

- **nano-gpt**: a GPT-2 style model with 85,584 parameters that sorts the letters A, B and C. It comes with the original step-by-step walkthrough.
- **tiny Llama**: the same task in Llama style, with RMS norm, RoPE, grouped-query attention and SwiGLU. It has 76,656 parameters.
- **tiny MoE**: tiny Llama with a mixture of experts. Each layer has 4 experts and each token uses 2. It has 132,528 parameters.
- **Real model shapes**: GPT-2 Small and XL, GPT-3, Llama 3.2 1B, Qwen3 8B, Mistral 7B, Mixtral 8x7B and gpt-oss-20b. They are drawn at full size without weights. Their parameter counts match the published totals within 1%.

## What you can do

- Step through the tiny Llama and MoE with a guide on the left. Use N to go forward and P to go back.
- Press Play on a step to watch its blocks fill in one number at a time, while the inspector shows how the current number is made. You can drag the timeline, change the speed, follow a single position, and let the guide move on by itself.
- Hover a coloured word in the guide to light up its block in 3D, or click it to fly there.
- Change the prompt letters and watch every block and guess update.
- Hover any block to see how one of its numbers is computed.
- See the model's guess at every layer (logit lens), the RoPE angles, the attention maps and the router choices.
- Generate one token at a time and watch which keys and values are reused from the cache.
- Read short notes on where newer models differ: local attention layers, shared experts and compressed key/value caches.

## Running locally

1. Install dependencies: `yarn`
2. Start the dev server: `yarn dev`
3. Open http://localhost:3002/llm

## Checks

- `node tools/check_modern_forward.ts`: the browser maths matches PyTorch for both tiny models.
- `npx tsx tools/check_presets.ts`: the real-model parameter counts match the published ones.
- `tools/train_tiny_modern.py` trains the tiny Llama and MoE and exports their weights.

## Credits

This project builds on [llm-viz](https://github.com/bbycroft/llm-viz) by [Brendan Bycroft](https://github.com/bbycroft). He made the 3D renderer, the GPT-2 walkthrough and the CPU simulator in `src/cpu`.

The nano-gpt model and its sorting task come from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT).

The real-model shapes come from each model's published `config.json` on Hugging Face.

## License

MIT, see [`LICENSE`](LICENSE). Copyright (c) 2023-2026 Brendan Bycroft. Third-party assets such as fonts and RISC-V tests keep their own licenses.
