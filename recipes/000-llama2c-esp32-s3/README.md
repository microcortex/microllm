# Recipe 000: A tiny Llama on the ESP32-S3, from scratch

> **Status:** 🚧 code complete, awaiting first hardware run
> **Task:** text generation (Shakespeare-style char-level model) running 100% offline on a $5 chip
> **Target boards:** ESP32-S3-DevKitC-1-N8R8 (8 MB flash, 8 MB octal PSRAM)
> **Final size:** ~6.1 MB Q8 blob + ~100 KB runtime RAM

## Why this recipe exists

Everyone asks "can I run ChatGPT on my ESP32?" — no. A 0.5B model quantized to bits
is still 200–400 MB; your chip has 8 MB of PSRAM. But a **~6M-parameter model trained
from scratch** (à la Karpathy's `llama2.c` "stories" models) at Q8 is ~6.1 MB and
**actually fits**, generating several tokens/s. That's enough to prove the whole pipeline:
dataset → train → quantize → flash → benchmark. Every later recipe reuses it
(recipe 001 will shrink the *same* architecture to <1 MB with LittleBit-style 0.3 BPW).

## Architecture (micro-6M)

| Param | Value |
|---|---|
| dim | 288 |
| hidden_dim | 768 (SwiGLU) |
| n_layers | 6 |
| n_heads / n_kv_heads | 6 / 6 (GQA-ready) |
| vocab | ≤256 (char-level) |
| seq_len | 96 |
| params | ~6.1M → **~6.1 MB Q8** |

**RAM budget on the S3 (8 MB PSRAM):** blob 6.1 MB, más KV cache fp32 (6×2×96×288×4B ≈ 1.33 MB), más scratch ≈ **7.6 MB total ✅** (that's why seq=96, not 128).

## File tree

```text
000-llama2c-esp32-s3/
├── train/
│   ├── johnny_train.py     # self-contained trainer + exporter (johnny v1 format, Q8)
│   └── requirements.txt    # torch, numpy
└── firmware/               # ESP-IDF v5.3+ project ("johnny" runtime)
    ├── partitions.csv      # factory 1.5MB + "model" data partition 6.7MB @ 0x190000
    ├── sdkconfig.defaults  # 8MB flash, octal PSRAM, 240MHz
    └── main/
        ├── johnny_format.h # .jny file format spec (single source of truth)
        ├── johnny.c/.h     # loader + KV-cached forward + Q8 kernels + sampling
        ├── oled_ssd1306.c  # 128x64 OLED text output (the "wow" screen)
        └── main.c          # app_main: load → generate → stream tokens
```

## Hardware

| Item | Detail |
|---|---|
| Board | ESP32-S3-DevKitC-1-N8R8 (8 MB quad-flash? **no**: 8 MB flash + 8 MB octal PSRAM) |
| OLED | SSD1306 128x64 I2C, addr 0x3C, default SDA=GPIO8 SCL=GPIO9 (see `menuconfig`) |
| Programmer | USB-JTAG/serial built into the S3 |

## Bill of materials

| Part | Qty | ~Price | Link |
|---|---|---|---|
| ESP32-S3-DevKitC-1-N8R8 | 1 | $8–12 | (any vendor) |
| SSD1306 0.96" I2C OLED | 1 | $2–3 | (any vendor) |

## Pipeline overview

```text
corpus .txt ──▶ char tokenizer ──▶ train micro-6M ──▶ export Q8 (.jny) ──▶ flash partition ──▶ tokens/s on OLED
```

Each stage is a **single command** (see Stages below).

## Stage 1 — Dataset

v1 uses **Tiny Shakespeare** (1.1 MB, char-level, ~65 unique chars) — the classic
pipeline-proof corpus. Swapping to Spanish text (D001/D003) is just `--data mi_corpus.txt`
(the tokenizer auto-builds from whatever you pass, capped at 256 chars).

```bash
# the trainer auto-downloads input.txt on first run; or place your own:
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -P data/
```

## Stage 2 — Train

```bash
cd train
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python johnny_train.py --data data/input.txt --steps 5000 --out out/
```

- Wall time: ~20–40 min on a consumer GPU (RTX 3060-class); a few hours CPU-only (steps↓ 2000 works)
- Every 500 steps it prints a sample generation so you can watch it learn
- Checkpoint saved to `out/model_ckpt.pt`, then exported to `out/model_Q8.jny`

## Stage 3 — Quantize (included in export)

Export writes the per-tensor Q8 blob (int8 + fp32 scale per matrix; RMSNorm vectors stay fp32).
The format spec lives in `firmware/main/johnny_format.h` — trainer and runtime must agree.
`--quant fp32` produces a debug variant (12.3 MB, won't fit 8 MB flash with the app).

## Stage 4 — Convert & Flash

```bash
cd ../firmware
idf.py set-target esp32s3
idf.py menuconfig    # optional: OLED pins under "Johnny runtime"
idf.py build flash monitor
# in a SECOND terminal (or after monitor: Ctrl+] then):
esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 921600 write_flash 0x190000 ../train/out/model_Q8.jny
# reset the board (EN button) and re-run idf.py monitor
```

## Quickstart

```bash
# prereqs: ESP-IDF v5.3+ (with . $IDF_PATH/export.sh) and Python 3.10+ with torch
cd train  && pip install -r requirements.txt && python johnny_train.py --data data/input.txt --steps 5000 --out out/
cd ../firmware && idf.py set-target esp32s3 && idf.py build
esptool.py --chip esp32s3 --port /dev/ttyUSB0 write_flash 0x190000 ../train/out/model_Q8.jny
idf.py -p /dev/ttyUSB0 flash monitor
```

## Results

📌 *(pending first hardware run — will link `benchmarks/results/000-…` when done)*

**Honest expectations:** plain-C Q8 kernels on a 240 MHz LX7 ≈ **1–10 tok/s**
(memory-bound from PSRAM); the Arena run will give exact numbers, then esp-nn/SIMD
kernels and LittleBit 0.3-BPW (recipe 001) multiply that.

## Reproduce

*(this section is completed when the recipe reaches ✅ — the steps above are the contract;
any deviation found on hardware gets fixed here first.)*

## Credits & sources

- Andrej Karpathy — [llama2.c](https://github.com/karpathy/llama2.c) (MIT) — architecture & format inspiration
- Ronen Eldan & co. — TinyStories (proof that small models can be coherent)
- Espressif — [esp-nn](https://github.com/espressif/esp-nn) (optimized kernels; integration planned)
- Samsung Research — LittleBit (NeurIPS 2025) — the reason recipe 001 exists (see `docs/` analysis)
- Built by **Emmanuel Vázquez** ([@Vazquezelectronics](https://github.com/Vazquezelectronics)) and the MicroCortex community
