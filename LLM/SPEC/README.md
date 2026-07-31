# LLM / SPEC — Speculative Decoding

General-purpose speculative-decoding papers for text LLMs. `VLA/SPEC` (SPECVLA, CEEDVLA) ports
these techniques onto robot VLA models; this folder holds the underlying methods.
Naming follows the repo convention ([`../../README.md`](../../README.md)): short uppercase
abbreviations, `<NAME>.pdf`.

> **30 PDFs across eight sub-themes plus a survey.** Files are flat in this folder; the headings
> below are the conceptual grouping.
> **◆ = systems-priority** (training-free or serving-system, fast to ship). Venues distinguish
> peer-reviewed proceedings from arXiv-only preprints.

---

## FOUNDATION · draft-and-verify origins
- ✅◆ **SPECDECODE** — Fast Inference from Transformers via Speculative Decoding (ICML 2023 Oral) — `arXiv:2211.17192`
- ✅◆ **SPECSAMPLE** — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv, DeepMind) — `arXiv:2302.01318`

## MULTITOKEN · multi-token / multi-head self-drafting
- ✅ **MEDUSA** — Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (ICML 2024) — `arXiv:2401.10774`
- ✅ **MTP** — Better & Faster Large Language Models via Multi-token Prediction (ICML 2024) — `arXiv:2404.19737`
- ✅◆ **DEEPSEEKV3** — DeepSeek-V3 Technical Report (cascaded single-layer MTP draft heads) — `arXiv:2412.19437`
- ✅ **HYDRA** — Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding (COLM 2024) — `arXiv:2402.05109`
- ✅ **REDRAFTER** — Recurrent Drafter for Fast Speculative Decoding (arXiv, Apple) — `arXiv:2403.09919`
- ✅ **PARALLELSPEC** — ParallelSpec: Parallel Drafter for Efficient Speculative Decoding (arXiv) — `arXiv:2410.05589`
- ✅ **CLOVER2** — Clover-2: Accurate Inference for Regressive Lightweight Speculative Decoding (arXiv) — `arXiv:2408.00264`

## EAGLE · feature-level prediction family (SafeAILab)
- ✅ **EAGLE** — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (ICML 2024) — `arXiv:2401.15077`
- ✅ **EAGLE2** — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (EMNLP 2024) — `arXiv:2406.16858`
- ✅ **EAGLE3** — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (NeurIPS 2025) — `arXiv:2503.01840`
- ✅ **HASS** — Learning Harmonized Representations for Speculative Sampling (ICLR 2025) — `arXiv:2408.15766`

## TREE · tree-structured drafting / verification
- ✅◆ **SPECINFER** — SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference and Verification (ASPLOS 2024) — `arXiv:2305.09781`
- ✅◆ **SEQUOIA** — Sequoia: Scalable and Robust Speculative Decoding (NeurIPS 2024) — `arXiv:2402.12374`
- ✅ **DDTREE** — Accelerating Speculative Decoding with Block Diffusion Draft Trees (arXiv) — `arXiv:2604.12989`
- ✅◆ **BLOCKVERIFY** — Block Verification Accelerates Speculative Decoding (ICLR 2025) — `arXiv:2403.10444`

## SELFSPEC · self-speculative / early-exit / layer-skip (no separate draft model)
- ✅◆ **DRAFTVERIFY** — Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding (ACL 2024) — `arXiv:2309.08168`
- ✅ **LAYERSKIP** — LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding (ACL 2024, Meta) — `arXiv:2404.16710`
- ✅ **KANGAROO** — Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting (NeurIPS 2024) — `arXiv:2404.18911`

## SYSTEM · draft-model-free / deployment systems
- ✅◆ **LOOKAHEAD** — Break the Sequential Dependency of LLM Inference Using Lookahead Decoding (ICML 2024) — `arXiv:2402.02057`
- ✅◆ **REST** — REST: Retrieval-Based Speculative Decoding (NAACL 2024) — `arXiv:2311.08252`
- ✅◆ **SPECEXEC** — SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices (NeurIPS 2024) — `arXiv:2406.02532`

## LONGCONTEXT · long-context / KV-cache-bottleneck speculative decoding
- ✅◆ **TRIFORCE** — TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding (COLM 2024) — `arXiv:2404.11912`
- ✅◆ **MAGICDEC** — MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation (ICLR 2025) — `arXiv:2408.11049`
- ✅ **LONGSPEC** — LongSpec: Long-Context Lossless Speculative Decoding with Efficient Drafting and Verification (ICML 2025) — `arXiv:2502.17421`

## DIFFUSION · diffusion-based drafters
- ✅ **BLOCKDIFF** — Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (ICLR 2025 Oral) — `arXiv:2503.09573`
- ✅◆ **DFLASH** — DFlash: Block Diffusion for Flash Speculative Decoding (ICML 2026) — `arXiv:2602.06036`
- ✅ **CADDT** — Cost-Aware Diffusion Draft Trees for Speculative Decoding (arXiv) — `arXiv:2606.01813`

## SURVEY
- ✅ **SPECSURVEY** — Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding (ACL 2024 Findings; defines Spec-Bench) — `arXiv:2401.07851`

---

## Lineage
FOUNDATION (lossless draft-and-verify) → MULTITOKEN (multi-head self-drafting; Medusa/Hydra/ReDrafter/
ParallelSpec/Clover) → EAGLE (feature-level 1→2→3, plus HASS harmonization) → TREE (optimal trees,
block verification) → SELFSPEC (early-exit/layer-skip, no separate draft model) → SYSTEM (draft-model-
free and offloaded serving) → LONGCONTEXT (sparse-KV drafting for the long-context bottleneck) →
DIFFUSION (block diffusion → one-pass block drafting → draft trees over diffusion distributions).

## Sources
All PDFs are from arXiv. Every ID was verified against its page-1 title before download.
Download convention: `curl arxiv.org/pdf/<ID>` → `<NAME>.pdf`.
