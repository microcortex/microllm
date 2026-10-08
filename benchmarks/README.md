# 📊 TinyLLM Arena — benchmarks

The rule: **same prompt, real boards, published numbers.** No cherry-picking.

This is the "wall of truth" of the project. Every model that appears in a recipe
gets measured here with the same methodology, so results are comparable across
boards, models and time.

## Metrics we measure

| Metric | Definition | Tool |
|---|---|---|
| **Decode tok/s** | Generated tokens per second, single stream, after warm-up | firmware timer |
| **TTFT** (time to first token) | Cold and warm, from reset | firmware timer |
| **Peak RAM** | Max heap used during inference (incl. PSRAM if mapped) | `heap_caps_get_minimum_free()` / IDF heap trace |
| **Flash footprint** | Model + runtime .bin size | `esptool.py image_info` + `size` |
| **Energy per token** | mJ/token at 3.3 V, measured at the board power rail | INA219 / Nordic PPK2 / Joulescope |
| **Current draw** | mA idle vs. inference avg | same as above |

## Standard conditions

- CPU at max stable clock (240 MHz ESP32-S3), flash/PSRAM speed documented
- Compiler optimizations `-O2` minimum, flag exact toolchain version
- Temperature: room temp; sustained-load results noted if throttling occurs
- 3 runs minimum, report median; outlier runs published, not hidden

## Boards

Board revisions and configs live here as results are added. See
[`TEMPLATE.md`](TEMPLATE.md) for the exact per-result format.

## Submitting

1. Copy `TEMPLATE.md` into `results/` named `NNN-model-board.md`
2. Fill every field — "n/a" is fine, "?" is not
3. Open a PR; a reproducer will try it on their board before merge
