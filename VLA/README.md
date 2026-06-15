# VLA — Vision-Language-Action Inference Papers

A papers collection for entering **VLA inference research** from a VDB/ANNS systems background.
Organization and naming follow this repo's [`../README.md`](../README.md) convention:
PDFs live under theme folders, named with short uppercase abbreviations.

> **43 PDFs (~330 MB)** across 14 themes. Study path: [`ROADMAP/00-roadmap.md`](ROADMAP/00-roadmap.md).
> **◆ = systems-priority** (serving / cache / quantization / scheduling — mostly training-free, fast to ship). Years are taken from arXiv IDs.

---

## SURVEY · `SURVEY/`
- ✅◆ **EFFICIENTVLA** — A Survey on Efficient VLA Models (closest to inference systems) — `arXiv:2510.24795`
- ✅◆ **EFFICIENTVLASYS** — Efficient VLA for Embodied Manipulation: A Systematic Survey — `arXiv:2510.17111`
- ✅ **PUREVLA** — Pure VLA Models: A Comprehensive Survey (paradigm taxonomy) — `arXiv:2509.19012`
- ✅ **VLAEMBODIED** — A Survey on VLA Models for Embodied AI (foundational survey) — `arXiv:2405.14093`
- ✅ **ANATOMYVLA** — An Anatomy of VLA Models: From Modules to Milestones — `arXiv:2512.11362`

## MODEL · foundational models · `MODEL/`
- ✅ **OPENVLA** — OpenVLA: An Open-Source VLA Model (CoRL 2024, the entry baseline) — `arXiv:2406.09246`
- ✅ **PI0** — π₀: A VLA Flow Model (flow-matching line) — `arXiv:2410.24164`
- ✅ **PI05** — π₀.₅: VLA with Open-World Generalization — `arXiv:2504.16054`
- ✅ **OCTO** — Octo: Open-Source Generalist Robot Policy (RSS 2024) — `arXiv:2405.12213`
- ✅ **RDT** — RDT-1B: Diffusion Foundation Model for Bimanual (ICLR 2025) — `arXiv:2410.07864`
- ✅ **RT2** — RT-2: VLA Transfer Web Knowledge (CoRL 2023) — `arXiv:2307.15818`
- ✅ **OPENX** — Open X-Embodiment / RT-X (ICRA 2024; data + cross-embodiment) — `arXiv:2310.08864`

## DECODE · parallel decoding / non-autoregressive / action tokenization · `DECODE/`
- ✅◆ **OPENVLAOFT** — Fine-Tuning VLA: Optimizing Speed & Success (parallel decode + chunking, 25–50×) — `arXiv:2502.19645`
- ✅◆ **FAST** — FAST: Efficient Action Tokenization (DCT+BPE, 10× fewer action tokens) — `arXiv:2501.09747`
- ✅◆ **VOTE** — VLA Optimization with Trajectory Ensemble Voting — `arXiv:2507.05116`

## CACHE · KV cache / temporal reuse (caching system = your home turf) · `CACHE/`
- ✅◆ **VLACACHE** — VLA-Cache: Adaptive Token Caching (reuse KV of static tokens, training-free) — `arXiv:2502.02175`
- ✅◆ **TEMPOFIT** — TempoFit: Layer-Wise Temporal KV Memory (plug-and-play) — `arXiv:2603.07647`
- ✅◆ **KVEFFICIENT** — KV-Efficient VLA: RNN-Gated Chunked KV Cache — `arXiv:2509.21354`
- ✅◆ **STATICDYNAMIC** — Static-Dynamic Disentanglement for Long-Horizon VLA — `arXiv:2602.03983`

## PRUNE · visual token pruning · `PRUNE/`
- ✅◆ **VLAPRUNER** — Bridging the Semantic-Action Gap in Visual Token Pruning — `arXiv:2511.16449`

## SPEC · speculative / early-exit decoding · `SPEC/`
- ✅◆ **SPECVLA** — Spec-VLA: Speculative Decoding with Relaxed Acceptance (EMNLP 2025) — `arXiv:2507.22424`
- ✅◆ **CEEDVLA** — CEED-VLA: Consistency VLA with Early-Exit Decoding — `arXiv:2506.13725`

## QUANT · quantization / edge deployment (bandwidth & resource = your home turf) · `QUANT/`
- ✅◆ **LITEVLAEDGE** — LiteVLA-Edge: Quantized On-Device Control (Jetson, 4-bit, ~6.6 Hz) — `arXiv:2603.03380`
- ✅◆ **QAIL** — Quantization-Aware Imitation Learning — `arXiv:2412.01034`
- ✅◆ **SALIENCYQUANT** — Saliency-Aware Quantized Imitation Learning — `arXiv:2505.15304`

## SERVING · asynchronous / real-time inference (scheduling & pipelining = your home turf) · `SERVING/`
- ✅◆ **VLARAIL** — VLA-RAIL: Real-Time Async Inference Linker (model-agnostic middleware) — `arXiv:2512.24673`
- ✅◆ **VLASH** — VLASH: Real-Time VLAs via Future-State-Aware Async Inference — `arXiv:2512.01031`
- ✅◆ **MASKEDCHUNK** — Real-Time Robot Execution with Masked Action Chunking — `arXiv:2601.20130`
- ✅◆ **LEAVENOOBS** — Leave No Observation Behind: Real-time Correction for Chunks — `arXiv:2509.23224`

## DUALSYS · fast-slow dual system / hierarchy (cut latency via architecture) · `DUALSYS/`
- ✅ **FASTINSLOW** — Fast-in-Slow: Dual-System Unifying Fast within Slow — `arXiv:2506.01953`
- ✅ **ASYNCFASTSLOW** — Asynchronous Fast-Slow VLA for Whole-Body Manipulation — `arXiv:2512.20188`
- 🔗 Helix (Figure AI) and GR00T N1 (NVIDIA) have no arXiv PDF; links in the ROADMAP.

## MOE · mixture-of-experts / sparse activation · `MOE/`
- ✅ **ADAMOE** — Action-Specialized MoE for VLA — `arXiv:2510.14300`
- ✅ **HIMOEVLA** — HiMoE-VLA: Hierarchical MoE Generalist Policy — `arXiv:2512.05693`
- ✅ **FEDVLA** — FedVLA: Dual Gating Mixture-of-Experts — `arXiv:2508.02190`

## DIFFUSION · diffusion/flow action-head acceleration (denoising-step bottleneck) · `DIFFUSION/`
- ✅◆ **ONEDP** — One-Step Diffusion Policy (distillation, 1.5 → 62 Hz) — `arXiv:2410.21257`
- ✅ **CONSISTENCYPOLICY** — Consistency Policy: Consistency Distillation (RSS 2024) — `arXiv:2405.07503`
- ✅ **ONESTEPFLOW** — One-Step Flow Policy: Self-Distillation — `arXiv:2603.12480`

## BENCH · profiling / benchmark (best first entry for a systems person) · `BENCH/`
- ✅◆ **VLAPERF** — How Fast Can I Run My VLA? (demystifies inference perf; a profiling template) — `arXiv:2602.18397`
- ✅ **BENCHMARKING** — What Are We Actually Benchmarking in Robot Manipulation? (a critique) — `arXiv:2606.04233`

## RETRIEVAL · retrieval-augmented robotics (leverages your ANNS expertise) · `RETRIEVAL/`
- ✅◆ **RAEA** — Retrieval-Augmented Embodied Agents (CVPR 2024) — `arXiv:2404.11699`
- ✅◆ **RETRIEVEREASONACT** — Retrieval-Augmented Robots via Retrieve-Reason-Act — `arXiv:2603.02688`
- ✅ **MABR** — Multi-Agent Behavior Retrieval — `arXiv:2312.02008`

## DATASET · `DATASET/`
- ✅ **DROID** — DROID: Large-Scale In-The-Wild Manipulation Dataset (RSS 2024) — `arXiv:2403.12945`
- 🔁 Cross-embodiment large-scale data **OPENX** is under `MODEL/`.

---

## Themes a systems person should hit first
`CACHE/` · `PRUNE/` · `QUANT/` · `SERVING/` · `BENCH/` · `RETRIEVAL/`
— all serving / caching / scheduling work, mostly training-free and fast to ship. The topic-selection formula is in
[`ROADMAP/00-roadmap.md`](ROADMAP/00-roadmap.md) §"Transfer map".

## Sources
All PDFs are from arXiv. Every ID was verified against its title via the arXiv API before download.
Download convention: `curl https://arxiv.org/pdf/<ID>` → `<THEME>/<NAME>.pdf`.
