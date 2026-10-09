<div align="center">

# MicroCortex

**Tiny LLMs born for $5 chips.**
Recipes, benchmarks and open models that fit where the cloud can't.

**Author:** [Emmanuel Vazquez](https://github.com/Vazquezelectronics) ([@Vazquezelectronics](https://github.com/Vazquezelectronics)) — founder & author of MicroCortex

> 🧠 *Cortex* = the layer of the brain that does the thinking. *Micro* = where it lives.

[![CI](https://github.com/microcortex/microllm/actions/workflows/ci.yml/badge.svg)](https://github.com/microcortex/microllm/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

English · [Español](README.es.md)

</div>

---

## What is MicroCortex?

A fully open-source lab dedicated to running **useful language models on microcontrollers**:
ESP32-S3, RP2040/RP2350, $5 MCUs and even small FPGAs.

We don't just *quantize* existing LLMs (a 0.5B model at 4-bit is ~350 MB — it will never fit a
512 KB SRAM chip). We **build models born tiny**: dataset → train → quantize → deploy → measure,
and we publish the whole recipe.

> Every chip we flash is a von Neumann machine — his architecture, our playground.

## The three pillars

| Pillar | What it is | Where |
|---|---|---|
| 🍳 **Recipes** | Reproducible, step-by-step guides (dataset → train → quantize → flash → benchmark) | [`recipes/`](recipes/) |
| 📊 **TinyLLM Arena** | Honest benchmarks: tokens/s, peak RAM, mJ/token — measured on real boards | [`benchmarks/`](benchmarks/) |
| 🤖 **Models & Datasets** | Micro-models born for specific tasks (ES + EN) and the datasets behind them | [`models/`](models/) · [`datasets/`](datasets/) |

## Hardware we target

| Board | SRAM | PSRAM | Price | Notes |
|---|---|---|---|---|
| ESP32-S3 | 512 KB | up to 8 MB | ~$5 | The main workhorse. Vector instructions via `esp-nn` |
| RP2040 | 264 KB | – | ~$1 | The $1 challenge. PIO + overclock tricks |
| RP2350 | 520 KB | up to 16 MB | ~$5 | New kid, double the muscle |
| ESP32-C6 | 512 KB | – | ~$3 | Wi-Fi 6 + BLE, tiny footprint |
| iCE40 / ECP5 FPGA | LUTs 🤘 | – | $10–30 | Open toolchain (Yosys/nextpnr) experiments |
| Raspberry Pi Zero 2 W | 512 MB | – | ~$15 | The "big leagues" reference point |

## Quickstart

```bash
# Recipe 000: run a tiny LLM on an ESP32-S3 from scratch
git clone https://github.com/microcortex/microllm
cd microllm/recipes/000-llama2c-esp32-s3
make setup flash monitor   # ⏳ in progress — see the recipe for status
```

🌐 **No toolchain? Flash from your browser** (ESP Web Tools) — coming soon in [`tools/flash-web/`](tools/flash-web/).

## Benchmark wall

| Model | Params | Quant | Board | RAM peak | Speed | Energy | Recipe |
|---|---|---|---|---|---|---|---|
| *(first results coming — see [`benchmarks/`](benchmarks/))* | | | | | | | |

## Roadmap

- [ ] **R0** — Recipe 000: `llama2.c`-style model compiled for ESP32-S3, XIP from flash
- [ ] **R1** — TinyLLM Arena v1: benchmark harness + first 5 boards measured
- [ ] **R2** — First *original* micro-model: Spanish smart-home intent parser, 100% offline
- [ ] **R3** — Flash-from-browser for all recipes
- [ ] **R4** — `bottleneck`: tool that fits a model to your board's RAM budget
- [ ] **R5** — FPGA recipe: a Llama block on LUTs (iCE40)

Full plan: [`docs/roadmap.md`](docs/roadmap.md)

## Community

- 🐦 X/Twitter: [@microcortex](https://x.com/microcortex)
- 💬 Discussions: use the tab up top for show & tell
- 🇪🇸 Todo el proyecto tiene documentación en español: [README.es.md](README.es.md)

## Support

If a recipe saved you hours of pain, consider [sponsoring](https://github.com/sponsors/microcortex) ☕
— it pays for boards, sensors and the coffee behind late-night token/s measurements.

## License

MIT — see [LICENSE](LICENSE). Models and datasets have their own licenses, listed per artifact.

---

<div align="center">
<sub>Built in the open. Same prompt, real boards, published numbers. 🦾</sub>
<sub><em>In memory of Johnny.</em></sub>
</div>
