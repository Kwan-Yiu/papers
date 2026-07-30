# LLM — Inference & AI Systems Research Library

This library targets text-LLM inference and serving-systems research. It excludes VLA/VLM and is
designed for database and systems researchers moving into LLM inference.

## Start here

1. [`ROADMAP/00-roadmap.md`](ROADMAP/00-roadmap.md) — complete research roadmap and review of the original draft;
2. [`ROADMAP/01-competency-plan.md`](ROADMAP/01-competency-plan.md) — prerequisite-driven competency plan;
3. [`ROADMAP/02-research-projects.md`](ROADMAP/02-research-projects.md) — research project ladder;
4. [`ROADMAP/03-architecture-taxonomy.md`](ROADMAP/03-architecture-taxonomy.md) — fine-grained architecture taxonomy;
5. [`ROADMAP/04-moe-deep-dive.md`](ROADMAP/04-moe-deep-dive.md) — end-to-end MoE deep dive;
6. [`ROADMAP/05-inference-systems-taxonomy.md`](ROADMAP/05-inference-systems-taxonomy.md) — full inference-systems taxonomy;
7. [`ROADMAP/06-bottleneck-research-map.md`](ROADMAP/06-bottleneck-research-map.md) — bottleneck and research map;
8. [`RESOURCES/GITHUB-REPO-ATLAS.md`](RESOURCES/GITHUB-REPO-ATLAS.md) — categorized GitHub repository atlas;
9. [`RESOURCES/README.md`](RESOURCES/README.md) — local course, paper, and source-code entry points.

## Library

The curated library contains **121 PDFs**:

- **116 research papers** across eleven themes;
- **5 course/reference PDFs**;
- [`SPEC/`](SPEC/README.md) remains a focused collection of 30 speculative-decoding papers.

| Folder | Count | Scope |
|---|---:|---|
| [`FOUNDATION/`](FOUNDATION/README.md) | 3 | Transformer, Llama, reasoning workloads |
| [`PERF/`](PERF/README.md) | 4 | Roofline, inference cost, offload |
| [`ARCHITECTURE/`](ARCHITECTURE/README.md) | 22 | MQA/GQA, RoPE, model deltas, SSM/hybrid, MTP, dynamic depth, diffusion LM |
| [`ATTENTION/`](ATTENTION/README.md) | 6 | FlashAttention/FlashInfer, long-context sparse attention |
| [`CACHE/`](CACHE/README.md) | 6 | KV eviction, quantization, virtual memory, disaggregation |
| [`QUANT/`](QUANT/README.md) | 6 | INT8/INT4, PTQ, outlier/rotation |
| [`SERVING/`](SERVING/README.md) | 10 | batching, PagedAttention, SLO scheduling, PD disaggregation |
| [`PARALLEL/`](PARALLEL/README.md) | 1 | tensor/model parallel foundation |
| [`MOE/`](MOE/README.md) | 18 | routing, shared/fine-grained/extreme-scale experts, kernels, EP, offload |
| [`SPEC/`](SPEC/README.md) | 30 | speculative decoding full lineage |
| [`FRONTIER/`](FRONTIER/README.md) | 10 | 2025–2026 KV/state/agentic/batch frontier |
| [`COURSE/`](COURSE/README.md) | 5 | CS229, Boyd, MLSys, CUDA manuals |

## Local source repositories

`RESOURCES/repos/` contains **33 shallow clones** pinned in
[`RESOURCES/SOURCES.md`](RESOURCES/SOURCES.md). The first wave covers serving engines, kernels,
KV systems, simulation and courses. The second wave adds model semantics, SSM/hybrid architectures,
MoE frameworks/kernels/communication, cluster orchestration, quantization and speculative decoding.

The broader categorized list—including linked repositories that were intentionally not cloned and
archived/moved warnings—is in
[`RESOURCES/GITHUB-REPO-ATLAS.md`](RESOURCES/GITHUB-REPO-ATLAS.md).

The clone directory is intentionally git-ignored by this paper repository; source URLs and exact
commits are tracked instead.

## Relation to sibling libraries

- [`../VDB`](../VDB) — vector database / ANNS;
- [`../VLA`](../VLA) — Vision-Language-Action inference;
- `LLM/SPEC` contains general methods that also feed into `VLA/SPEC`.

## Provenance

Research PDFs come from arXiv or official conference sites. Course PDFs come from official
university/book/vendor sites. Every newly downloaded PDF was checked as a valid PDF and its first-page
title was matched to the source; full URLs are recorded in
[`RESOURCES/SOURCES.md`](RESOURCES/SOURCES.md).
