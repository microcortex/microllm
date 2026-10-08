# 🌐 flash-web — flash any recipe from the browser, zero installs

> **Status:** 📋 spec — implementation wanted

The lowest-friction way for a maker to try a micro-LLM: plug the board over USB,
open a webpage, click **Connect**, done. No ESP-IDF, no Arduino, no 4 GB toolchain.

## How

[ESP Web Tools](https://esphome.github.io/esp-web-tools/) (Web Serial API):

```html
<esp-web-tools-button
  manifest="https://microcortex.dev/manifest.json">
  <button slot="activate">⚡ Flash recipe 000 to my ESP32-S3</button>
</esp-web-tools-button>
```

`manifest.json` example:

```json
{
  "name": "MicroCortex Recipe 000 — tiny llama",
  "version": "0.1.0",
  "new_install_prompt_erase": true,
  "builds": [
    {
      "chipFamily": "esp32s3",
      "parts": [
        { "path": "bootloader.bin", "offset": 0 },
        { "path": "partition-table.bin", "offset": 32768 },
        { "path": "firmware.bin", "offset": 65536 },
        { "path": "model.q8.bin", "offset": 1048576 }
      ]
    }
  ]
}
```

## Requirements & notes

- Chrome/Edge (Web Serial API); Firefox fallback page with instructions
- Serve over HTTPS; builds per chipFamily (esp32s3, esp32c6, rp2040→ uf2 drag&drop page)
- Model partition layout: custom partition table with a `model` partition (XIP-mapped)

## Milestones

- [ ] M1: static page + manifest for recipe 000 (esp32s3)
- [ ] M2: auto chip-detect + version display
- [ ] M3: RP2040/RP2350 uf2 flow
- [ ] M4: microcortex.dev/flash — one URL to rule them all
