# Competitive analysis — 2026-10-08

## Direct reference case: Mia's AI Lab (mia-ai.net)

> @MiaAI_lab · github.com/MiaAI-Lab · HF: Mia-AiLab · ~6,146 stars across 23 recipes · 36k+ X followers

**What she does:** personal brand ("Mia") running frontier open-weight models (GLM, DeepSeek,
Qwen) locally on **NVIDIA DGX Sparks** ($3,499 desktop AI boxes) and RTX GPUs. Publishes
"recipes": benchmarks, patches, launch scripts — every number in the open, same-prompt
comparisons, no cherry-picking.

**Her funnel (the interesting part):**

1. Free recipes on GitHub → stars → credibility
2. Own website (mia-ai.net) as the hub: recipes index, tier list, experiments, downloads
3. **Services:** "Personal Install by Mia" via Whop — 1×, 2×, 4× DGX Spark installs
4. **Book:** "From Zero to Seen — Get monetized on X" (her own growth story: 200 → 36k in 4 months)
5. **Donations:** GitHub Sponsors + Ko-fi + crypto (NOWPayments widget)

**Lessons we copy (structure, not niche):**

- ✅ "Recipes" as the product unit — every recipe is a repo, a page and a video
- ✅ Same prompt / no cherry-picking / raw outputs published — trust is the moat
- ✅ Own domain as hub, GitHub as backend, X as megaphone
- ✅ A named benchmark tool (she has `sparkDash`; we'll have **TinyLLM Arena**)
- ✅ Monetization rails (Sponsors/Ko-fi/services/book) visible from day 1

**How we're different (this is the point):** she is the **high end** — $3.5k desktop
clusters, frontier models, million-token contexts. We are the **floor** — $1–5 chips,
models born tiny, offline tasks. She quantizes what exists; we train what can't be
bought. No overlap in audience hardware → **complementary, not competitive.** Her X
growth playbook is even a template we can learn from without competing.

## Other players (from prior research)

| Actor | Focus | Our gap |
|---|---|---|
| llama.cpp / ggml | Desktop/server inference | MCUs barely supported |
| llama2.c (Karpathy) | Education, pure C | Frozen since 2023; no recipes/boards |
| SmolLM2/3 (HF) | 135M–1.7B edge models | Still too big for $5 MCUs; EN-centric |
| BitNet b1.58 (MSFT) | 1.58-bit ternary models | No maker-facing tooling/content |
| Espressif esp-dl / esp-nn | Official SDK kernels | Cold docs, no community demos |
| Edge Impulse | Commercial TinyML SaaS | Closed, expensive, not LLM-focused |
| tinyML Foundation | Academic community | KWS/vision, almost no LLMs |

## Verdict

- **Tech:** scattered across papers and forks; nobody owns the "recipes for $5 chips" category.
- **Content:** Spanish maker/local-AI niche ≈ empty. English content exists but fragmented and
  rarely reproducible end-to-end.
- **Our moat:** TinyLLM Arena (measured, honest, comparable) + bilingual recipes + flash-from-browser
  friction removal. The demo IS the marketing.
