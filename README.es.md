<div align="center">

# MicroCortex

**Micro-LLMs nacidos para chips de $5.**
Recetas, benchmarks y modelos abiertos que caben donde la nube no llega.

> 🧠 *Cortex* = la capa del cerebro que piensa. *Micro* = donde vive.

[English](README.md) · Español

</div>

---

## ¿Qué es MicroCortex?

Un laboratorio 100% open-source dedicado a correr **modelos de lenguaje útiles en microcontroladores**:
ESP32-S3, RP2040/RP2350, micros de $5 y hasta FPGAs pequeñas.

No solo *quantizamos* LLMs existentes (un modelo de 0.5B a 4-bit pesa ~350 MB — jamás cabrá en un
chip de 512 KB de SRAM). **Creamos modelos nacidos pequeños**: dataset → entrenar → quantizar → flashear → medir,
y publicamos la receta completa.

> Cada chip que flasheamos es una máquina de von Neumann — su arquitectura, nuestro patio de juegos.

## Los tres pilares

| Pilar | Qué es | Dónde |
|---|---|---|
| 🍳 **Recetas** | Guías reproducibles paso a paso (dataset → entrenar → quantizar → flashear → medir) | [`recipes/`](recipes/) |
| 📊 **TinyLLM Arena** | Benchmarks honestos: tokens/s, RAM pico, mJ/token — medidos en placas reales | [`benchmarks/`](benchmarks/) |
| 🤖 **Modelos y Datasets** | Micro-modelos nacidos para tareas específicas (ES + EN) y sus datasets | [`models/`](models/) · [`datasets/`](datasets/) |

## Hardware objetivo

| Placa | SRAM | PSRAM | Precio | Notas |
|---|---|---|---|---|
| ESP32-S3 | 512 KB | hasta 8 MB | ~$5 | El caballo de batalla. Instrucciones vectoriales vía `esp-nn` |
| RP2040 | 264 KB | – | ~$1 | El reto de $1 |
| RP2350 | 520 KB | hasta 16 MB | ~$5 | El nuevo, con el doble de músculo |
| ESP32-C6 | 512 KB | – | ~$3 | Wi-Fi 6 + BLE |
| FPGA iCE40 / ECP5 | LUTs 🤘 | – | $10–30 | Toolchain abierto (Yosys/nextpnr) |
| Raspberry Pi Zero 2 W | 512 MB | – | ~$15 | El punto de referencia de las "grandes ligas" |

## Inicio rápido

```bash
# Receta 000: correr un micro-LLM en un ESP32-S3 desde cero
git clone https://github.com/microcortex/microllm
cd microllm/recipes/000-llama2c-esp32-s3
make setup flash monitor   # ⏳ en progreso — ver la receta para el estado
```

🌐 **¿Sin toolchain? Flashea desde tu navegador** (ESP Web Tools) — próximamente en [`tools/flash-web/`](tools/flash-web/).

## Muro de benchmarks

| Modelo | Params | Quant | Placa | RAM pico | Velocidad | Energía | Receta |
|---|---|---|---|---|---|---|---|
| *(primeros resultados próximamente — ver [`benchmarks/`](benchmarks/))* | | | | | | | |

## Roadmap

- [ ] **R0** — Receta 000: modelo estilo `llama2.c` compilado para ESP32-S3, XIP desde flash
- [ ] **R1** — TinyLLM Arena v1: harness de benchmarks + primeras 5 placas medidas
- [ ] **R2** — Primer micro-modelo *original*: parser de intents de domótica en español, 100% offline
- [ ] **R3** — Flasheo desde el navegador para todas las recetas
- [ ] **R4** — `bottleneck`: herramienta que hace caber un modelo en el presupuesto de RAM de tu placa
- [ ] **R5** — Receta FPGA: un bloque Llama en LUTs (iCE40)

Plan completo: [`docs/roadmap.md`](docs/roadmap.md)

## Comunidad

- 🐦 X/Twitter: [@microcortex](https://x.com/microcortex)
- 💬 Discussions: usa la pestaña de arriba para mostrar tus experimentos

## Apoya el proyecto

Si una receta te ahorró horas de dolor, considera [patrocinarme](https://github.com/sponsors/microcortex) ☕
— paga las placas, los sensores y el café de las mediciones nocturnas de tokens/s.

## Licencia

MIT — ver [LICENSE](LICENSE). Los modelos y datasets tienen su propia licencia, listada en cada artefacto.

---

<div align="center">
<sub>Construido en abierto. Mismo prompt, placas reales, números publicados. 🦾</sub>
<sub><em>En memoria de Johnny.</em></sub>
</div>
