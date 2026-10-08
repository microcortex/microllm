# Recipe 000: A tiny Llama on the ESP32-S3, from scratch

> **Status:** 🚧 draft — this is the first recipe of the project, being built now
> **Task:** text generation (a "stories"-class tiny model) running 100% offline on a $5 chip
> **Target boards:** ESP32-S3-DevKitC-1-N8R8 (8 MB PSRAM)
> **Final size:** target ≤ 16 MB model (XIP from flash) + ~50–100 KB runtime RAM

## Why this recipe exists

Everyone asks "can I run ChatGPT on my ESP32?" — no. A 0.5B model quantized to bits
is still 200–400 MB; your chip has 8 MB of PSRAM. But a **15M-parameter model trained
from scratch** (à la Karpathy's `llama2.c` "stories" models) at 8-bit is ~15 MB and
**actually fits**, generating a few tokens/s. That's enough to prove the whole pipeline:
dataset → train → quantize → flash → benchmark. Every later recipe reuses it.

## Hardware

| Item | Detail |
|---|---|
| Board | ESP32-S3-DevKitC-1-N8R8 (8 MB quad PSRAM, 8 MB flash) |
| Extras | SSD1306 OLED (I2C) for the "wow" demo output |
| Programmer | USB-JTAG/serial built into the S3 |

## Bill of materials

| Part | Qty | ~Price | Link |
|---|---|---|---|
| ESP32-S3-DevKitC-1-N8R8 | 1 | $8–12 | (any vendor) |
| SSD1306 0.96" I2C OLED | 1 | $2–3 | (any vendor) |

## Pipeline overview

```
tiny stories/bilingual text ──▶ train llama2.c-style (15M) ──▶ Q8_0 ──▶ XIP image ──▶ ESP32-S3 ──▶ tok/s on screen
```

## Stage 1 — Dataset

- Start: TinyStories-style short English text + a small Spanish corpus (the tokenizer budget matters: target 8k–16k vocab)
- `make dataset` → downloads + pre-processes into `data/train.bin`, `data/val.bin` *(to be written)*

## Stage 2 — Train

- Base: `llama2.c` (MIT) — pure C training + inference, no framework tax
- Config sketch: 6 layers, 6 heads, d_model 288, ctx 256, ~15M params
- Wall time: a few hours on a single consumer GPU (RTX 3060-class)

```bash
make train   # *(to be written)* produces out/model.bin
```

## Stage 3 — Quantize

- Q8_0 via llama2.c export; stretch goal: ternary (BitNet-style) variant

```bash
make quantize  # *(to be written)*
```

## Stage 4 — Convert & Flash

- ESP-IDF v5.x project; model as embedded blob + **XIP** (execute-in-place / memory-mapped reads from flash)
- Runtime: llama2.c `run.c` adapted to read from PSRAM/flash with 8-bit matmul kernels via `esp-nn`

```bash
make flash     # *(to be written)*
```

## Quickstart

```bash
# prereqs: esp-idf v5.3+ on PATH
idf.py set-target esp32s3
idf.py build flash monitor
```

## Results

📌 *(pending first measurement — will link `benchmarks/results/000-…` when done)*

Expected ballpark (honest): **1–5 tok/s decode**, ~15–18 MB flash, <200 KB RAM peak.

## Reproduce

*(this section is completed when the recipe reaches ✅)*

## Credits & sources

- Andrej Karpathy — [llama2.c](https://github.com/karpathy/llama2.c) (MIT)
- Ronen Eldan & co. — TinyStories paper (the "small models can be coherent" proof)
- Espressif — [esp-nn](https://github.com/espressif/esp-nn) optimized kernels
