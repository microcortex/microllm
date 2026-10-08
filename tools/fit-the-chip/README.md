# 🔧 fit-the-chip — fit a model to a board's memory budget

> **Status:** 📋 spec — implementation wanted (see "Milestones")

The inverse of the usual workflow: don't ask "how small can I quantize this model?",
ask "I have THIS board — what's the biggest/best model that fits?"

## Input

A board profile (JSON), e.g.:

```json
{
  "board": "ESP32-S3-DevKitC-1-N8R8",
  "sram_kb": 512,
  "psram_mb": 8,
  "flash_mb": 8,
  "cpu_mhz": 240,
  "features": ["esp-nn", "fp32-simd", "xip"],
  "runtime_reserve_kb": 120
}
```

## Output

- Max params achievable at each quant level (Q8/Q4/ternary) — with tokenizer budget accounted
- Recommended vocab size + hidden dim/layer count for a from-scratch model
- A "fits / doesn't fit" verdict for named models (llama2.c 15M, SmolLM-135M, BitNet-125M…)
- Estimated tok/s range using the TinyLLM Arena calibration curve

## CLI sketch

```bash
fit-the-chip --board esp32s3-n8r8.json
# ┌─────────────────────────────┐
# │ Budget: 8 MB PSRAM, 512 KB SRAM │
# │ Q8_0  → max ~7.4M params (in-PSRAM)  │
# │ Q4_0  → max ~14.8M params            │
# │ ternary → max ~29M params            │
# │ Verdict: SmolLM2-135M ❌ (won't fit) │
# └─────────────────────────────┘
```

## Milestones

- [ ] M1: board profiles in `profiles/*.json` (5 boards)
- [ ] M2: arithmetic core (params ↔ bytes per quant scheme, vocab cost)
- [ ] M3: known-model database + verdicts
- [ ] M4: tok/s estimator calibrated from benchmark results

## License

MIT.
