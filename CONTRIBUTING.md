# Contributing to MicroCortex

Everything here is open. Same prompt, real boards, published numbers.
If your numbers can't be reproduced, they don't go on the wall.

## Ways to contribute

1. **🍳 Add a recipe** — copy [`recipes/TEMPLATE.md`](recipes/TEMPLATE.md), fill it, open a PR.
   A recipe is only merged if someone else can reproduce it end-to-end.
2. **📊 Add benchmark results** — follow [`benchmarks/TEMPLATE.md`](benchmarks/TEMPLATE.md).
   Include your board revision, power configuration and measurement method.
3. **🤖 Train a micro-model** — models live in `models/` with their training script and dataset.
4. **🛠️ Improve the tools** — `tools/fit-the-chip/` and `tools/flash-web/` have open specs.

## Rules of the house

- **Reproducible or it didn't happen.** Every number needs a method description.
- **One recipe, one task.** No "general chatbot on 256KB" fantasy — task-specific models are the point.
- **Bilingual docs welcome.** English + Spanish is the default pairing; use one, we'll mirror it.
- **Cite your sources.** Papers, blog posts, prior forks.

## Recipe naming

```text
recipes/NNN-short-name/<files>
```

- `NNN` — zero-padded number (next one: check existing folders)
- `short-name` — `kebab-case`, e.g. `001-intent-parser-es-esp32s3`

## Commit style

Conventional commits: `feat:`, `fix:`, `docs:`, `bench:`, `recipe:`.

## Code of conduct

Be the kind of person who answers a beginner's `idf.py build` error at 2am. That's it.
