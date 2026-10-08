# 🤖 Models registry

Micro-models **born for the constraint** — not quantized-down giants.
Every model here has: training script, dataset, license, and at least one
benchmark result on a $5-class board.

## Registry

| Model | Task | Arch | Params | Quant | Size | Boards | Dataset | License |
|---|---|---|---|---|---|---|---|---|
| *(first model incoming: Spanish smart-home intent parser)* | | | | | | | | |

## What "born for the constraint" means

1. **Architecture sized to the board** — 10–30M params for 8 MB PSRAM chips; 1–5M for 264 KB SRAM
2. **Tokenizer budgeted** — vocab size counts against flash; a 32k-vocab BPE can cost more than the model
3. **Task-first data** — narrow domain, synthetic + human hybrid, ES/EN from day 1
4. **Quantization-aware** — QAT or easy-to-quantize layers; ternary/BitNet-style weights welcome
5. **Deployed & measured** — no model is "done" until it's on the wall of [`benchmarks/`](../benchmarks/)

## Publishing a model

- Weights → GitHub Releases or Hugging Face (link here)
- Training code → `recipes/NNN-*/train/` or linked repo
- Dataset → `datasets/` entry
- One row in the registry table above
