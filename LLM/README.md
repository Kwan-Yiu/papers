# LLM — Inference & AI Systems Research Library

This library targets text-LLM inference and serving-systems research. It excludes VLA/VLM and is
designed for database and systems researchers moving into LLM inference.

## Start here

| Entry | Purpose |
|---|---|
| [`ROADMAP/README.md`](ROADMAP/README.md) | roadmap document map and navigation |
| [`ROADMAP/00-roadmap.md`](ROADMAP/00-roadmap.md) | complete field map, external cost-model sources, and entry points |
| [`ROADMAP/01-ai-ml-foundations.md`](ROADMAP/01-ai-ml-foundations.md) | curated AI/ML, mathematics, neural-network, and PyTorch resources |
| [`ROADMAP/02-transformer-foundations.md`](ROADMAP/02-transformer-foundations.md) | curated Transformer and Hugging Face tutorials plus exact source paths |
| [`ROADMAP/03-modern-llm-architecture.md`](ROADMAP/03-modern-llm-architecture.md) | modern architecture → memory, compute, kernel, and communication costs |
| [`ROADMAP/04-gpu-compiler-kernels.md`](ROADMAP/04-gpu-compiler-kernels.md) | curated GPU, profiling, compiler, and kernel courses/source paths |
| [`ROADMAP/05-single-node-inference-engine.md`](ROADMAP/05-single-node-inference-engine.md) | readable-engine courses and production-engine request/source paths |
| [`ROADMAP/06-kv-scheduling-serving.md`](ROADMAP/06-kv-scheduling-serving.md) | KV/state, scheduling, decoding, and serving |
| [`ROADMAP/07-distributed-inference-moe.md`](ROADMAP/07-distributed-inference-moe.md) | curated distributed foundations plus MoE research taxonomy |
| [`ROADMAP/08-production-reliability.md`](ROADMAP/08-production-reliability.md) | curated SRE/Kubernetes/LLM-platform operational reading map |
| [`ROADMAP/09-bottleneck-research.md`](ROADMAP/09-bottleneck-research.md) | bottleneck classes, research questions, and evidence standards |
| [`ROADMAP/10-research-projects.md`](ROADMAP/10-research-projects.md) | bounded systems-research project catalog |
| [`ROADMAP/COMPETENCY-GATES.md`](ROADMAP/COMPETENCY-GATES.md) | cross-cutting required evidence and exit gates |
| [`RESOURCES/GITHUB-REPO-ATLAS.md`](RESOURCES/GITHUB-REPO-ATLAS.md) | categorized English GitHub repository atlas |
| [`RESOURCES/README.md`](RESOURCES/README.md) | local course, paper, and source-code entry points |

The roadmap files are navigators rather than self-contained tutorials. They link to English courses,
blogs, official documentation, papers, and repositories, then identify the exact chapters or source
paths to read and the evidence required to move forward.

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
