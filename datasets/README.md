# 🗂️ Datasets

Micro-models are only as good as their (small!) datasets. This folder hosts and
indexes datasets designed for **tiny task-specific models**, Spanish + English.

## Design rules

1. **Tiny by design** — a 10M-param model doesn't need 1T tokens. 10–500 MB of
   task-specific data usually beats 1T tokens of generic web text for narrow tasks.
2. **JSONL everywhere** — one example per line, schema declared per dataset.
3. **Split discipline** — `train` / `val` / `test` with no leakage; test set never
   used for decisions.
4. **License declared** — CC0/MIT/Apache-2.0 preferred; sourced data must document origin.
5. **Bilingual from day 1** — ES and EN versions of every intent template.

## Planned datasets

| ID | Name | Task | Size (target) | Status |
|---|---|---|---|---|
| D001 | `intents-domotica-es` | Smart-home command → JSON intent (ES) | 50k examples | 🚧 spec only |
| D002 | `intents-smarthome-en` | Same, English | 50k examples | 🚧 spec only |
| D003 | `sensor-logs-es` | Log-line → anomaly summary/label (ES) | 100k lines | 📋 idea |
| D004 | `micro-instructions-es` | Short instructions → step lists (ES) | TBD | 📋 idea |

## Format example (D001)

```json
{"text": "enciende la luz de la cocina", "intent": "light_on", "slots": {"room": "cocina"}}
{"text": "apaga el ventilador del cuarto", "intent": "device_off", "slots": {"device": "ventilador", "room": "cuarto"}}
```

## Contributing data

Synthetic generation is welcome — but 5% must be human-verified real-world phrasings
(regionalisms: *mándale luz a la cocina*, *dale gas al boiler*...). PRs with a
`datasets/D00X/README.md` (schema, provenance, license, counts) are auto-tracked here.
