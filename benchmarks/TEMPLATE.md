# [NNN] model-name @ board-name

- **Date (YYYY-MM-DD):**
- **Submitter / reproducer:**
- **Recipe:** link to `recipes/.../README.md`
- **Model:** name + params + quantization (e.g. `micro-intent-es-10M, Q8_0, 10.4 MB`)
- **Board:** exact revision (e.g. `ESP32-S3-DevKitC-1-N8R8, rev 1.1`)
- **Firmware:** commit hash + toolchain version (e.g. `esp-idf v5.3, gcc 13.2`)

## Configuration

- CPU clock: (e.g. 240 MHz)
- Flash / PSRAM: (e.g. 8 MB PSRAM @ 80 MHz, XIP enabled)
- Power mode: (e.g. default PM disabled)
- Model location: (e.g. XIP from flash / fully in PSRAM)

## Results (median of 3 runs; raw runs listed)

| Metric | Value |
|---|---|
| Decode tok/s (1 stream) | |
| TTFT cold / warm | |
| Peak internal RAM | |
| Peak PSRAM used | |
| Flash footprint (model + runtime) | |
| Energy per token | |
| Current idle / inference | |

## Notes

- Anomalies, throttling, thermal, anything the reader should know.

## Reproduce

```bash
# exact commands, from a fresh checkout, that produce these numbers
```
