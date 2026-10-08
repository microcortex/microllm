# Roadmap — first 90 days

> KPIs at day 90: **500+ GitHub stars · 1,000+ X followers · 1 demo video >50k views · Recipe 000 reproducible ✅**

## Phase 0 — Foundations (week 1-2)

- [ ] Create GitHub org `microcortex` + profile README + org avatar
- [ ] Push this skeleton as `microllm` repo; enable Discussions + Sponsors
- [ ] Create X account `@microcortex`; pinned intro thread (EN+ES)
- [ ] Register `microcortex.dev` (~$13/yr) — redirect to GitHub until the site exists
- [ ] Record a 30s "why" video: RAM price supercycle → the answer is $5 chips

## Phase 1 — Recipe 000 (week 2-6)

- [ ] ESP-IDF v5.x project skeleton + OLED driver
- [ ] Train llama2.c-style 15M model (stories + small ES corpus)
- [ ] Q8_0 export + XIP flash mapping
- [ ] First `tok/s` on-screen demo → **record it** (this is the GIF in every README)
- [ ] Recipe 000 → ✅ reproducible; CI green

## Phase 2 — TinyLLM Arena v1 (week 6-10)

- [ ] Benchmark harness: tok/s + peak RAM + flash footprint (energy via INA219 later)
- [ ] Measure: ESP32-S3, ESP32-C6, RP2040, RP2350, RPi Zero 2 W
- [ ] Publish wall of truth + `TEMPLATE.md` workflow for community submissions
- [ ] Thread: "I benchmarked tiny LLMs on 5 boards. The $1 chip surprised me." (EN+ES)

## Phase 3 — First original micro-model (week 10-14)

- [ ] Dataset D001 `intents-domotica-es` v0 (synthetic + human-verified 5%)
- [ ] Train ~10-30M intent-parser model; quantize; deploy on ESP32-S3
- [ ] Demo: voice-ish text in → JSON intent out, 100% offline
- [ ] **Launch day:** Show HN + r/LocalLLaMA + r/esp32 + Hackaday tip line + Espressif blog ping

## Phase 4 — Monetization rails (week 12+)

- [ ] GitHub Sponsors live from day 1 (tier: "buy me a devkit $5" — on brand)
- [ ] Ko-fi page; affiliate links (boards/mics/OLED) in every recipe BOM
- [ ] Pitch first sponsorship at ~5k followers (Espressif, Seeed, PCBWay, JLCPCB)
- [ ] Kit BOM finalized → Tindie listing draft (week 16+)

## Beyond 90 days

- R3 flash-web live · R4 fit-the-chip CLI · R5 FPGA recipe (iCE40) · NLnet grant application · course/playbook
