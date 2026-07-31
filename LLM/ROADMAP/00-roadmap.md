# LLM Inference & AI Systems Research Roadmap

> **Target:** Database / systems researcher → LLM inference and AI systems researcher
>
> **Primary question:** Where do computation, memory, communication, state, and queueing go?
>
> **Boundary:** text-LLM inference and serving systems, with training knowledge included only where required

[Roadmap index](README.md) ·
[AI/ML foundations](01-ai-ml-foundations.md) ·
[Transformer foundations](02-transformer-foundations.md) ·
[Competency gates](COMPETENCY-GATES.md)

---

## 1. End Goal

Build enough AI and model knowledge to reason correctly about execution, then go deep on the systems
layers that determine inference efficiency and reliability:

```text
AI/ML and PyTorch
→ Transformer mechanics
→ modern LLM architecture
→ single-node inference optimization
→ single-node inference engine
→ KV cache, scheduling, and serving
→ distributed inference and MoE
→ production reliability
→ bottleneck-driven systems research
```

The goal is not to memorize model names, clone many repositories, or report isolated speedups. The
goal is to formulate and validate quantitative claims about end-to-end inference systems.

---

## 2. Complete Layer Map

### Layer 01 — AI, ML, and PyTorch

Learn:

- tensors, shapes, dtypes, devices, and layouts;
- matrix multiplication, probability, softmax, and cross entropy;
- neural networks, activations, normalization, and residuals;
- forward, backward, gradients, and optimizers;
- training/evaluation/generalization;
- PyTorch modules, autograd, data loading, and model state.

Output: small trainable models and the ability to inspect loss, gradients, parameters, and runtime
state.

Read: [`01-ai-ml-foundations.md`](01-ai-ml-foundations.md).

### Layer 02 — Transformer and Hugging Face

Learn:

- tokenization, embeddings, and next-token likelihood;
- Q/K/V, causal masking, multi-head attention;
- RoPE, RMSNorm, SwiGLU, residual connections;
- training, prefill, decode, autoregressive generation, and KV cache;
- sampling and stopping;
- Hugging Face tokenizer/config/model/generation/cache interfaces.

Output: a minimal decoder and KV cache, plus an end-to-end inspection of a real
`AutoModelForCausalLM`.

Read: [`02-transformer-foundations.md`](02-transformer-foundations.md).

### Layer 03 — Modern LLM Architecture

Learn architecture deltas:

- GPT/Llama/Qwen/DeepSeek families;
- MHA/GQA/MQA/MLA;
- full, sliding-window, sparse, and hybrid attention;
- dense FFN and sparse MoE;
- SSM/recurrent/hybrid state;
- MTP and speculative-friendly heads;
- dynamic depth and iterative/diffusion generation;
- precision as an execution design.

Output: an architecture delta matrix that predicts memory, FLOPs, kernels, state, and communication.

Read: [`03-modern-llm-architecture.md`](03-modern-llm-architecture.md).

### Layer 04 — Single-Node Inference Optimization

Learn:

- source-grounded memory/FLOP/bandwidth models and profiler reconciliation;
- weight, activation, KV, attention, and MoE quantization;
- pruning, hardware-supported sparsity, structural reduction, and distillation boundaries;
- KV selection, eviction, compression, tiering, and quality contracts;
- MQA/GQA/MLA and cross-layer KV/attention sharing;
- sliding-window, sparse, retrieval, linear, recurrent, SSM, and hybrid attention;
- FlashAttention/FlashInfer and the boundary between algorithms and kernels;
- GPU execution, CUDA, Triton, CUTLASS/CuTe, compilation, and CUDA Graphs;
- production-engine configuration, backend dispatch, and fallback paths.

Output: a source-grounded optimization comparison that connects predicted resource reduction,
measured memory/runtime evidence, quality, backend support, and end-to-end serving behavior.

Read: [`04-single-node-inference-optimization.md`](04-single-node-inference-optimization.md).

### Layer 05 — Single-Node Inference Engine

Learn:

- request and sequence state;
- continuous batching;
- prefill/decode mixing;
- token budgets;
- paged/block KV allocation;
- preemption and resumption;
- model runner and attention backend;
- logits, sampling, detokenization, and streaming;
- CPU/GPU coordination.

Output: minimal engine components and a source-level request path through vLLM or SGLang.

Read: [`05-single-node-inference-engine.md`](05-single-node-inference-engine.md).

### Layer 06 — KV Cache, Scheduling, and Serving

Learn:

- workload and arrival taxonomy;
- TTFT, ITL, E2E, throughput, and goodput;
- static/continuous batching and chunked prefill;
- priority, fairness, deadlines, and preemption;
- prefix caching, paging, tiering, compression, and state transfer;
- quantization and speculative decoding;
- serving-framework boundaries and evaluation matrices.

Output: workload-aware state/scheduling decisions defended using tail latency, memory, and SLO
goodput.

Read: [`06-kv-scheduling-serving.md`](06-kv-scheduling-serving.md).

### Layer 07 — Distributed Inference and MoE

Learn:

- TP, PP, DP, CP/SP, and EP;
- collectives and point-to-point transfer;
- PCIe, NVLink/NVSwitch, InfiniBand/RoCE, and accelerator fabrics;
- placement, topology, disaggregation, and overlap;
- MoE routing, dispatch, grouped GEMM, all-to-all, skew, and expert placement;
- cross-accelerator boundaries.

Output: topology-aware parallelism/placement model and end-to-end MoE critical-path analysis.

Read: [`07-distributed-inference-moe.md`](07-distributed-inference-moe.md).

### Layer 08 — Production Reliability and Operations

Learn:

- service contracts and SLOs;
- control plane/data plane;
- metrics, tracing, and structured logs;
- routing, admission, backpressure, and autoscaling;
- cold start;
- worker/GPU/node/network/state failures;
- replay, recovery, cancellation, and idempotency;
- tenant isolation, artifact safety, cost, energy, and capacity.

Output: overload, recovery, isolation, and cost evidence for a production-style service.

Read: [`08-production-reliability.md`](08-production-reliability.md).

### Layer 09 — Bottleneck-Driven Research

Learn:

- workload characterization;
- resource and queue diagnosis;
- falsifiable hypotheses;
- strong baselines;
- cost models;
- mechanism design;
- ablations, stress cases, negative cases, and break-even boundaries;
- reproducible claim packaging.

Output: a research claim whose evidence and limitations can be independently checked.

Read: [`09-bottleneck-research.md`](09-bottleneck-research.md) and
[`10-research-projects.md`](10-research-projects.md).

---

## 3. Source-Grounded Inference Cost Model

The roadmap does not define a new cost-model tutorial. Use the following external sources, then
apply their models consistently across architecture, kernel, scheduler, and placement work.

| Topic | Core external source | Exact reading target | Roadmap use |
|---|---|---|---|
| compute, bandwidth, capacity | [How To Scale Your Model — Rooflines](https://jax-ml.github.io/scaling-book/roofline/) | full chapter | FLOPs, bytes, arithmetic intensity, lower bounds |
| GPU and network topology | [How To Scale Your Model — GPUs](https://jax-ml.github.io/scaling-book/gpus/) | GPU hierarchy, collectives, LLM rooflines | chip/node/cluster cost |
| Transformer FLOPs and shapes | [How To Scale Your Model — Transformer Math](https://jax-ml.github.io/scaling-book/transformers/) | forward-pass and sharding calculations | parameter and operator ledger |
| inference prefill/decode | [Efficiently Scaling Transformer Inference](../PERF/TRANSFORMERINFER.pdf) | analytical model and evaluation | phase-specific compute/memory behavior |
| framework overhead and fusion | [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | overhead, fusion, memory, compilation | eager/compiler/kernel boundaries |
| memory measurement | [PyTorch — Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/main/torch_cuda_memory) | snapshots, allocator state and visibility boundary | formula-versus-measurement reconciliation |
| quantization concepts | [Hugging Face — Quantization concepts](https://huggingface.co/docs/transformers/quantization/concept_guide) | scheme, granularity, PTQ/QAT and formats | weight/activation/KV representation |
| quantized deployment | [vLLM — Quantization](https://docs.vllm.ai/en/stable/features/quantization/index.html) | method and hardware support matrix | backend and hardware feasibility |
| attention/KV architecture | [LLMs-from-scratch — Chapter 4 bonus material](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04) | GQA, MLA, window, DeltaNet, sparse and cross-layer sharing | readable mechanism and memory estimators |
| KV memory and paging | [vLLM](../SERVING/VLLM.pdf) | PagedAttention memory model | persistent inference state |
| communication primitives | [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | collective semantics | payload, rounds, and synchronization |
| queueing and iteration scheduling | [Orca](../SERVING/ORCA.pdf) | iteration-level scheduling | online request interference |
| serving objectives | [DistServe](../SERVING/DISTSERVE.pdf) and [Sarathi-Serve](../SERVING/SARATHI.pdf) | metrics and SLO evaluation | TTFT, TPOT/ITL, E2E, throughput, goodput |
| production SLOs | [Google SRE — Service Level Objectives](https://sre.google/sre-book/service-level-objectives/) | full chapter | SLI/SLO/error-budget contract |

Required cost record for any later claim:

- exact tensor/operator/request shape;
- dtype and precision policy;
- FLOPs and bytes moved;
- persistent weights, KV/state, workspaces, and metadata;
- serial steps, launches, barriers, collectives, and transfers;
- request distribution, arrival process, queue state, and saturation;
- predicted bottleneck before measurement;
- measured profiler/serving evidence;
- model, hardware, topology, software commit, and quality/SLO contract.

---

## 4. Research Entry Points for Database Researchers

| Entry point | Database analogy | LLM-systems problem |
|---|---|---|
| KV/state | buffer pool, paging, tiering | allocation, reuse, eviction, migration, prefetch |
| scheduling | queues, admission, fairness | dynamic batches, interference, tail latency |
| disaggregation | compute/storage separation | state transfer, placement, weak-interconnect boundaries |
| MoE | partitioning and skew | expert placement, replication, all-to-all, hot experts |
| reliability | logging/recovery/isolation | request resumption, stale state, tenant boundaries |

The project catalog is [`10-research-projects.md`](10-research-projects.md).

---

## 5. Reading and Code Strategy

### English learning spine

| Source | Primary use |
|---|---|
| [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en) | mathematics, ML, neural-network, and PyTorch prerequisites |
| [PyTorch Tutorials](https://github.com/pytorch/tutorials) | official framework tutorials and recipes |
| [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) | decoder and language-model basics |
| [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer) | executable original Transformer |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | model semantics and configuration |
| [Stanford CS336](https://github.com/stanford-cs336/lectures) | model, systems, scaling, and evaluation |
| [LLMs from Scratch — Chapter 4 bonus material](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04) | memory analysis, KV, GQA, MLA, sliding window, DeltaNet, sparse attention, and cross-layer KV sharing |
| [GPU MODE Lectures](https://github.com/gpu-mode/lectures) | GPU, kernels, and communication |
| [Efficient Deep Learning Systems](https://github.com/mryab/efficient-dl-systems) | profiling, compilation, deployment, and inference |
| [Google DeepMind Scaling Book](https://github.com/jax-ml/scaling-book) | topology, sharding, and cross-accelerator scaling |
| [MLSysBook](https://github.com/harvard-edge/cs249r_book) | deployment, reliability, security, and systems breadth |

The full categorized map is
[`../RESOURCES/GITHUB-REPO-ATLAS.md`](../RESOURCES/GITHUB-REPO-ATLAS.md).

### Source-reading order

```text
concept / paper claim
→ model configuration and semantics
→ minimal implementation
→ production engine path
→ compiler/kernel path
→ profiler trace
→ workload-level evaluation
```

---

## 6. Competency Model

The roadmap documents curate external explanations and point to paper/code taxonomies.
[`COMPETENCY-GATES.md`](COMPETENCY-GATES.md) defines the required evidence and exit gates.

Use it to answer:

- can the concept be derived rather than repeated;
- can the mechanism be implemented or traced;
- can cost be predicted before measurement;
- can profiler evidence confirm or falsify the prediction;
- can correctness/quality constraints be separated from performance;
- can negative cases and claim boundaries be stated?

---

## 7. Final Standard

The roadmap is complete when you can:

- explain AI/ML training mechanics and use PyTorch without treating it as opaque;
- implement and inspect a decoder Transformer and Hugging Face causal LM;
- reconstruct modern LLM architecture from configuration;
- calculate parameter, activation, KV/state, FLOP, and communication costs;
- classify and trace quantization, KV compression/eviction, efficient attention and cross-layer
  sharing without conflating their semantic and systems layers;
- trace model code through compiler IR, kernels, and runtime dispatch;
- profile prefill/decode and predict bottlenecks;
- trace requests through scheduler, model runner, attention backend, state manager, and streaming;
- reason about cache, batching, admission, preemption, prefix reuse, and tail latency;
- choose parallelism and placement from model, workload, topology, and SLO;
- analyze MoE routing, grouped GEMM, communication, skew, and expert placement;
- reason about overload, faults, recovery, isolation, cost, and energy;
- convert a measured bottleneck into a falsifiable, reproducible systems research claim.

---

**Next:** [`01-ai-ml-foundations.md`](01-ai-ml-foundations.md) ·
**Evaluation:** [`COMPETENCY-GATES.md`](COMPETENCY-GATES.md)
