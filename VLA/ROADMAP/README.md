# VLA Inference Systems Roadmap — Companion Index

This folder is the companion index for the study path in [`00-roadmap.md`](00-roadmap.md).
Unlike the VDB roadmap, VLA papers are the archive themselves and live by **theme** under [`../`](../) (`SURVEY/`, `CACHE/`, …);
this README maps each roadmap section to its theme folder. Math / transformer foundations are reused from [`../../VDB/ROADMAP/`](../../VDB/ROADMAP/).

> The papers behind this path are **43 PDFs (~330 MB)**, listed in [`../README.md`](../README.md).
> ✅ downloaded (under `../<THEME>/`) · 🔁 cross-repo reference · 🔗 link only (no arXiv). **◆ = systems-priority**.

---

## Roadmap section → theme folder

| Section | Theme | Folder | Key PDFs |
|---------|-------|--------|----------|
| §1 Build a cost model | survey + profiling | `../SURVEY/` `../BENCH/` | ◆ EFFICIENTVLA, VLAPERF |
| §2 Understand a baseline | foundational models | `../MODEL/` | OPENVLA, PI0, OPENX |
| §3 Decoding acceleration | parallel / non-AR | `../DECODE/` | ◆ OPENVLAOFT, FAST |
| §4 Temporal KV cache ◆ | caching | `../CACHE/` | ◆ VLACACHE, TEMPOFIT |
| §5 Token pruning ◆ | pruning | `../PRUNE/` | ◆ VLAPRUNER |
| §6 Quantization / edge ◆ | quantization | `../QUANT/` | ◆ LITEVLAEDGE, QAIL |
| §7 Speculative / early-exit | speculative decode | `../SPEC/` | ◆ SPECVLA, CEEDVLA |
| §8 Async / real-time serving ◆ | serving | `../SERVING/` | ◆ VLARAIL, VLASH |
| §9 Dual-system / MoE | architecture | `../DUALSYS/` `../MOE/` | FASTINSLOW, ADAMOE |
| §10 Diffusion acceleration | action head | `../DIFFUSION/` | ◆ ONEDP |
| §11 Profiling / benchmark | evaluation | `../BENCH/` | ◆ VLAPERF, BENCHMARKING |
| §12 Retrieval augmentation ◆ | retrieval | `../RETRIEVAL/` | ◆ RAEA |
| §13 Transfer map | LLM-serving base | 🔁 `../../VDB/ROADMAP/09-transformer-attention/` | vLLM/PagedAttention, FlashAttention |

---

## Suggested order (from `00-roadmap.md` §14)

- **Phase 0 — build the map**: skim `../SURVEY/` + ◆ `../BENCH/VLAPERF.pdf`
- **Phase 1 — baseline**: ◆ `../MODEL/OPENVLA.pdf` → run LIBERO → latency breakdown
- **Phase 2 — the standard answer**: ◆ `../DECODE/OPENVLAOFT.pdf`
- **Phase 3 — go deep on your turf**: ◆ `../CACHE/` → `../PRUNE/` → `../SERVING/` (caching + scheduling, training-free, easiest to publish)
- **Phase 4 — fill in**: `../QUANT/` → `../SPEC/` → `../DIFFUSION/` → `../DUALSYS/` `../MOE/`
- **Phase 5 — pick a topic**: choose one from the §13 transfer map + `../RETRIEVAL/` (differentiation) → write a proposal

---

## Works with no arXiv PDF (mentioned in roadmap §9)
- 🔗 Helix (Figure AI, dual-system S2-VLM + S1-visuomotor): https://www.figure.ai/news/helix
- 🔗 GR00T N1 (NVIDIA, S1 diffusion 10 ms + S2 planner): https://developer.nvidia.com/isaac/gr00t

## Sources
All PDFs are from arXiv; every ID was verified against its title via the arXiv API before download.
File names follow this repo's convention (short uppercase abbreviations).
