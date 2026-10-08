# Recipe: [TITLE]

> **Status:** 🚧 draft / ✅ reproducible / ⚠️ needs updater
> **Task:** [what this model does — one sentence]
> **Target boards:** [e.g. ESP32-S3 (8MB PSRAM), RP2350]
> **Final size:** [model size + runtime, e.g. 10.4 MB + 60 KB]

## Why this recipe exists

[The pain it solves. Why "just quantize X" doesn't work here. What is born-tiny about it.]

## Hardware

| Item | Detail |
|---|---|
| Board | [exact devkit + revision] |
| PSRAM / Flash | [sizes + speeds] |
| Extras | [e.g. I2S mic INMP441, SSD1306 OLED 128x64] |
| Programmer | [USB-JTAG / UART bridge] |

## Bill of materials

| Part | Qty | ~Price | Link |
|---|---|---|---|

## Pipeline overview

```
dataset ──▶ train ──▶ quantize ──▶ convert ──▶ flash ──▶ measure
```

Each stage has its own section below. Every stage is a **single command**.

## Stage 1 — Dataset

[Where it comes from, schema, size, generation script if synthetic.]

```bash
make dataset   # downloads/generates into data/
```

## Stage 2 — Train

[Framework: llama2.c / nanoGPT / custom C. Hyperparams table. Expected wall time on GPU X.]

```bash
make train     # produces out/model.bin
```

## Stage 3 — Quantize

[Method: Q8_0 / ternary / QAT. Size before → after.]

```bash
make quantize  # out/model.bin -> out/model.q8.bin
```

## Stage 4 — Convert & Flash

[Conversion to device format; ESP-IDF / arduino-cli / PlatformIO build.]

```bash
make flash     # builds + flashes + boots
```

## Quickstart

```bash
# from a fresh checkout, with [prereqs: esp-idf v5.x]
make setup flash monitor
```

## Results

[Link to `benchmarks/results/NNN-....md` — the numbers live there, not here.]

## Reproduce

[Exact commands from zero, including toolchain install. If it can't be done in one
sitting by a stranger, it's not done.]

## Credits & sources

[Papers, forks, prior art. Be generous.]
