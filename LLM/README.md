# LLM — Inference & AI Systems Research Library

This library targets LLM inference and AI-systems research for database and systems researchers.
Text generation is the main path; multimodal encode/prefill/decode, embeddings/rerankers,
diffusion-language models, reasoning and agent workloads are explicit system branches. Robot VLA
papers remain in the sibling `VLA/` library.

## Start here

| Entry | Purpose |
|---|---|
| [`ROADMAP/README.md`](ROADMAP/README.md) | roadmap document map and navigation |
| [`ROADMAP/00-roadmap.md`](ROADMAP/00-roadmap.md) | complete field map, external cost-model sources, and entry points |
| [`ROADMAP/01-ai-ml-foundations.md`](ROADMAP/01-ai-ml-foundations.md) | curated AI/ML, mathematics, neural-network, and PyTorch resources |
| [`ROADMAP/02-transformer-foundations.md`](ROADMAP/02-transformer-foundations.md) | curated Transformer and Hugging Face tutorials plus exact source paths |
| [`ROADMAP/03-modern-llm-architecture.md`](ROADMAP/03-modern-llm-architecture.md) | modern architecture → memory, compute, kernel, and communication costs |
| [`ROADMAP/04-training-post-training-systems.md`](ROADMAP/04-training-post-training-systems.md) | data, scaling, distributed training, SFT/PEFT, preference learning, RL and rollout systems |
| [`ROADMAP/05-single-node-inference-optimization.md`](ROADMAP/05-single-node-inference-optimization.md) | quantization, KV compression, efficient attention, profiling, GPU, compiler, and kernel paths |
| [`ROADMAP/06-decoding-test-time-compute.md`](ROADMAP/06-decoding-test-time-compute.md) | sampling, structured output, speculative decoding, parallel/diffusion decoding, reasoning and agent inference |
| [`ROADMAP/07-single-node-inference-engine.md`](ROADMAP/07-single-node-inference-engine.md) | readable-engine courses and production-engine request/source paths |
| [`ROADMAP/08-kv-scheduling-serving.md`](ROADMAP/08-kv-scheduling-serving.md) | KV/state, scheduling, batching, routing and serving |
| [`ROADMAP/09-distributed-inference-moe.md`](ROADMAP/09-distributed-inference-moe.md) | distributed foundations plus MoE research taxonomy |
| [`ROADMAP/10-production-reliability.md`](ROADMAP/10-production-reliability.md) | model lifecycle, SRE, Kubernetes and LLM-platform operational reading map |
| [`ROADMAP/11-bottleneck-research.md`](ROADMAP/11-bottleneck-research.md) | bottleneck classes, research questions, and evidence standards |
| [`ROADMAP/COMPETENCY-GATES.md`](ROADMAP/COMPETENCY-GATES.md) | cross-cutting required evidence and exit gates |
| [`RESOURCES/GITHUB-REPO-ATLAS.md`](RESOURCES/GITHUB-REPO-ATLAS.md) | categorized English GitHub repository atlas |
| [`RESOURCES/README.md`](RESOURCES/README.md) | local course, paper, and source-code entry points |

The roadmap files are navigators rather than self-contained tutorials. They link to English courses,
blogs, official documentation, papers, and repositories, then identify the exact chapters or source
paths to read and the evidence required to move forward.

## Library

The curated library contains **164 PDFs**:

- **159 research papers** across fourteen themes;
- **5 course/reference PDFs**;
- [`SPEC/`](SPEC/README.md) remains a focused collection of 30 speculative-decoding papers.

| Folder | Count | Scope |
|---|---:|---|
| [`FOUNDATION/`](FOUNDATION/README.md) | 3 | Transformer, Llama, reasoning workloads |
| [`TRAINING/`](TRAINING/README.md) | 25 | data/scaling, distributed training, SFT/PEFT, preference learning and reasoning RL |
| [`PERF/`](PERF/README.md) | 4 | Roofline, inference cost, offload |
| [`ARCHITECTURE/`](ARCHITECTURE/README.md) | 26 | MQA/GQA/MLA, cross-layer KV/attention sharing, model deltas, SSM/hybrid |
| [`ATTENTION/`](ATTENTION/README.md) | 12 | FlashAttention/FlashInfer, sparse, linear/recurrent and hybrid attention |
| [`CACHE/`](CACHE/README.md) | 8 | KV eviction, selection, quantization, virtual memory, disaggregation |
| [`QUANT/`](QUANT/README.md) | 10 | weight/activation/KV quantization, PTQ/QAT, low-bit adaptation, outliers and rotations |
| [`COMPRESSION/`](COMPRESSION/README.md) | 2 | weight sparsity, pruning and model compression |
| [`SERVING/`](SERVING/README.md) | 10 | batching, PagedAttention, SLO scheduling, PD disaggregation |
| [`PARALLEL/`](PARALLEL/README.md) | 1 | tensor/model parallel foundation |
| [`MOE/`](MOE/README.md) | 18 | routing, shared/fine-grained/extreme-scale experts, kernels, EP, offload |
| [`SPEC/`](SPEC/README.md) | 30 | speculative decoding full lineage |
| [`FRONTIER/`](FRONTIER/README.md) | 10 | 2025–2026 KV/state/agentic/batch frontier |
| [`COURSE/`](COURSE/README.md) | 5 | CS229, Boyd, MLSys, CUDA manuals |

## Local source repositories

`RESOURCES/repos/` contains **50 shallow clones** pinned in
[`RESOURCES/SOURCES.md`](RESOURCES/SOURCES.md). They cover courses, model semantics, data and
training stacks, post-training/RL systems, serving engines, speculative/structured decoding,
attention and GEMM kernels, KV systems, MoE, distributed communication, orchestration,
quantization, profiling and simulation.

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
