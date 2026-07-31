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
→ GPU, compiler, and kernels
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

### Layer 04 — GPU, Compiler, and Kernels

Learn:

- GPU execution and memory hierarchy;
- Roofline and arithmetic intensity;
- GEMM/attention/kernel shapes;
- profiling and benchmark hygiene;
- CUDA, Triton, CUTLASS/CuTe, and vendor libraries;
- graph capture, TorchDynamo, Inductor, compiler IR, and code generation;
- CUDA Graphs and dynamic-shape constraints;
- portable versus backend-specific reasoning.

Output: operator cost model, profiler trace, modified kernel, and compiler graph/IR trace.

Read: [`04-gpu-compiler-kernels.md`](04-gpu-compiler-kernels.md).

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

## 3. Unified Inference Cost Model

Translate every architecture, kernel, scheduler, and placement proposal into the same resource
language.

### 3.1 Quantities

| Quantity | Controls |
|---|---|
| FLOPs | arithmetic demand |
| bytes moved | memory and interconnect traffic |
| persistent bytes | weights, KV/state, adapters, buffers, metadata |
| serial steps | autoregressive iterations, refinement, verification |
| synchronization | launches, barriers, collectives, dispatch/combine |
| queueing | waiting, interference, preemption, and tail latency |

### 3.2 Weight capacity

```text
weight_bytes ≈ parameter_count × bytes_per_weight
```

Separate total, resident, active, replicated, quantized, and temporarily staged weights.

### 3.3 KV capacity

For conventional attention:

```text
kv_bytes
≈ 2 × layers × KV_heads × head_dimension
  × retained_tokens × bytes_per_element
```

GQA, MQA, MLA, sliding windows, recurrent state, compression, and offload change this model.

### 3.4 Roofline

```text
arithmetic_intensity = FLOPs / bytes_moved

attainable_compute
= min(peak_compute,
      arithmetic_intensity × memory_bandwidth)
```

Prefill and decode occupy different shape and intensity regimes. Neither is universally compute- or
memory-bound.

### 3.5 Communication

Record:

```text
payload bytes
message count
peers and hops
link bandwidth/latency
serialization
synchronization
contention
load imbalance
```

Fewer bytes can still regress if a mechanism adds rounds, small messages, or critical-path skew.

### 3.6 Serving objectives

```text
TTFT       = arrival → first token
ITL / TPOT = time between output tokens
E2E        = arrival → final token
throughput = tokens or requests / second
goodput    = requests / second satisfying all declared SLOs
```

Report workload distributions, saturation, percentiles, warmup, repetitions, model, precision,
hardware, topology, software commit, and quality/SLO contract.

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
| [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) | decoder and language-model basics |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | model semantics and configuration |
| [Stanford CS336](https://github.com/stanford-cs336/lectures) | model, systems, scaling, and evaluation |
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

The learning documents teach concepts and mechanisms. [`COMPETENCY-GATES.md`](COMPETENCY-GATES.md)
defines the required evidence and exit gates.

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
