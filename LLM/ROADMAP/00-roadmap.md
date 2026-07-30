# LLM Inference & AI Systems Research Roadmap

> Target: Database Systems Researcher → LLM Inference / AI Systems Researcher
>
> Focus: cost model, GPU execution, KV state, serving, scheduling, distributed inference, MoE
> Frontier snapshot: 2026-07-30

This roadmap does not postpone systems work until every LLM topic has been learned. The correct
progression is:

1. run a measurable inference system as early as possible;
2. build cost models for compute, bandwidth, memory capacity, communication, and queueing;
3. validate each concept with profiling and experiments;
4. select a measured bottleneck for reproduction, modification, and research.

The prerequisite-driven execution path is in
[`01-competency-plan.md`](01-competency-plan.md), and research projects are in
[`02-research-projects.md`](02-research-projects.md). Detailed topic maps include:

- [`03-architecture-taxonomy.md`](03-architecture-taxonomy.md) — architecture deltas;
- [`04-moe-deep-dive.md`](04-moe-deep-dive.md) — MoE;
- [`05-inference-systems-taxonomy.md`](05-inference-systems-taxonomy.md) — serving stack;
- [`06-bottleneck-research-map.md`](06-bottleneck-research-map.md) — bottleneck and topic map.

All local resources are indexed in [`../RESOURCES/README.md`](../RESOURCES/README.md). The GitHub
repository map is in
[`../RESOURCES/GITHUB-REPO-ATLAS.md`](../RESOURCES/GITHUB-REPO-ATLAS.md).

---

## 1. Review of the original draft

The original direction was sound: Transformer → modern LLM → quantization → inference → systems →
infrastructure, with mostly primary sources. An inference-systems research path, however, needs the
following corrections.

| Original design | Problem | Revision |
|---|---|---|
| Complete linear algebra, probability, and optimization first | Systems experiments are delayed by a broad prerequisite wall | Learn mathematics just in time, beginning with the parts needed for attention, loss, and quantization |
| Put systems and infrastructure last | Every inference layer depends on hardware and systems costs | Introduce latency, bandwidth, memory, queueing, and profiling at the baseline gate |
| Make training a full core phase | SFT and RLHF are not prerequisites for inference research | Keep the required background in pretraining, tokenization, loss, and scaling; treat alignment as optional |
| Treat quantization as an isolated phase | Its value depends on kernels, batch shape, hardware, and quality constraints | Study it after the cost model, alongside KV compression and speculation |
| List GPT, Llama, Qwen, and DeepSeek | A list of model names does not create systems understanding | Learn architecture deltas: how RoPE, GQA/MLA, MoE, and MTP change costs |
| Omit serving metrics and workloads | There is no way to judge online-system value | Require TTFT, ITL/TPOT, E2E, throughput, goodput, P99, and cost/token |
| Name Triton, CUTLASS, and vLLM as projects | This encourages linear reading of large repositories | Require a verifiable exit gate for every module and complete baseline → profile → change → ablation |
| Make the path completely linear | Modern bottlenecks cross kernels, runtimes, clusters, and workloads | Keep a prerequisite core, then select research branches from a bottleneck matrix |

The original draft was a useful LLM knowledge map, but not yet a systems-research path. This
revision promotes **performance modeling, experimental method, and system-state management** to the
core and moves nonessential training details into background material.

---

## 2. Build a unified inference cost model first

Ask five questions before evaluating any paper or optimization:

1. What is the target workload: short prompts, long context, long outputs, batch, interactive,
   agentic, multi-LoRA, or MoE?
2. What is limiting it: compute, HBM bandwidth, capacity, kernel-launch/CPU overhead, network,
   queueing, or imbalance?
3. Which bytes, FLOPs, serial steps, or queueing delays does the optimization change?
4. Which semantics are preserved: exact output, distribution, bounded quality loss, or best effort?
5. Under which load, hardware, and SLO does the benefit disappear?

### 2.1 Four quantities you must calculate

**Weight memory**

```text
weight_bytes ≈ parameter_count × bytes_per_weight
```

**Per-request KV cache**

```text
kv_bytes ≈ 2 × n_layers × n_kv_heads × head_dim × sequence_length × bytes_per_element
```

GQA, MQA, and MLA are systems designs as much as model designs: they directly change
`n_kv_heads` or the cached representation.

**Roofline upper bound**

```text
arithmetic_intensity = FLOPs / bytes_moved
attainable_FLOPs = min(peak_FLOPs, arithmetic_intensity × memory_bandwidth)
```

Small-batch decode is often limited by weight reads and HBM bandwidth, while long-prompt prefill is
more likely to enter a compute-bound region. Never treat "decode is always memory-bound" as an
unconditional rule.

**Serving objectives**

```text
TTFT      = arrival → first token
ITL/TPOT  = time between output tokens
E2E       = arrival → final token
throughput = tokens or requests / second
goodput   = requests / second that satisfy all SLOs
```

Average tokens/s is insufficient for a research paper. At minimum, report workload distributions,
P50/P95/P99, warmup, repetitions, model, precision, hardware, software commit, and SLO.

---

## 3. Inference-systems bottleneck map

| Layer | Main bottleneck | Typical techniques | Critical counterexample |
|---|---|---|---|
| Kernel | attention/GEMM, launch overhead, fusion | FlashAttention, FlashInfer, Triton, CUDA Graph | A single-shape microbenchmark does not represent dynamic serving |
| Device memory | weights, KV capacity, fragmentation, bandwidth | paged/virtual KV, quantization, eviction, sparse attention | Compression ratio does not imply end-to-end speedup |
| Runtime | dynamic batching, prefill/decode interference, CPU scheduler | continuous batching, chunked prefill, asynchronous scheduling | Higher throughput may damage tail latency |
| Node/cluster | KV migration, communication, topology, heterogeneous resources | PD disaggregation, TP/PP/EP, RDMA, cache-aware routing | Full disaggregation may be slower on a weak interconnect |
| MoE | all-to-all, expert imbalance, small expert batches | expert parallelism, token dispatch/combine, load balancing | Parameter sparsity does not imply low communication or latency |
| Long context | KV working set, attention, CPU/SSD I/O | hierarchical cache, offload/prefetch, native sparse attention | A cache hit can still block on data loading |
| Reasoning/batch | long and unknown output lengths, stragglers | output-aware scheduling, speculation, coroutines | Optimizing only TTFT hides completion time |
| Agentic/multi-turn | tool gaps, cross-turn state, prefix reuse | TTL/pinning, program-aware routing, prefetch | A request-level scheduler cannot see the workflow |
| Operations | bursts, multi-tenancy, autoscaling, faults | admission, routing, flow control, simulation | Single-node throughput cannot answer production-cost questions |

Three entry points align especially well with a database-systems background:

1. **KV cache as a storage engine**: paging, tiering, indexing, admission, eviction, prefetch, and I/O;
2. **SLO-aware scheduling**: dynamic batches, cache locality, mixed request lengths, and
   online/offline coexistence;
3. **Disaggregated stateful serving**: separation of compute and KV state, routing, migration, and
   weak-interconnect costs.

---

# Core Path

## Core Module 0 — Baseline Before Theory

**Goal**

Obtain a reproducible inference baseline and observe real bottlenecks.

### Do

- Serve a 1B–8B decoder-only model that fits the available GPU with vLLM or SGLang.
- Test short-prompt/short-output, long-prompt/short-output, and short-prompt/long-output workloads.
- Sweep request rate, batch/token budget, and context length.
- Record TTFT, ITL, E2E, throughput, goodput, and GPU memory.
- Use PyTorch Profiler or Nsight Systems to inspect CPU activity, kernels, idle gaps, and synchronization.

### Deliverable

A baseline report containing the environment, commands, workloads, three performance curves, and
an initial bottleneck diagnosis.

### Exit gate

Explain why prefill and decode have different resource profiles, and show that the conclusion comes
from measurements rather than received wisdom.

---

## Core Module 1 — Just-in-Time Foundations

**Goal**

Understand the numerical semantics of a decoder-only Transformer without completing an exhaustive
mathematics curriculum first.

### Essential math

- tensor shapes, matrix multiplication, and intuition for bases and vector spaces;
- dot products, orthogonality, and projections;
- softmax, log-sum-exp, and probability distributions;
- entropy, cross entropy, and KL divergence;
- gradients, the chain rule, and the role of SGD/Adam;
- intuition for SVD, low rank, and reconstruction error.

### Defer until needed

- complete eigenvalue theory;
- every proof in convex optimization;
- a full Bayesian inference course;
- diffusion mathematics;
- RLHF/PPO details.

### Model concepts

- tokenization, embeddings, and causal masks;
- Q/K/V and multi-head attention;
- residual connections, normalization, and FFNs;
- next-token likelihood and cross entropy;
- the distinct execution forms of training, prefill, and decode;
- why a KV cache is correct, what it stores, and what it cannot store.

### Read / build

- [`FOUNDATION/ATTENTION.pdf`](../FOUNDATION/ATTENTION.pdf)
- `RESOURCES/repos/cs336-lectures`
- Reuse `/home/junyao/code/study/tutorials/llm/LLMs-from-scratch`.
- Reuse `/home/junyao/code/study/tutorials/transformer/annotated-transformer`.

### Exit gate

- Calculate a two-token, one-head attention example by hand.
- Write every Q/K/V tensor shape.
- Derive parameter count, weight memory, and KV bytes per token from a model configuration.
- Implement a minimal decoder block and check its outputs against PyTorch.

---

## Core Module 2 — GPU and Performance Foundations

**Goal**

Distinguish compute, memory, overhead, and communication instead of guessing from aggregate GPU
utilization.

### Topics

- GPU memory hierarchy: HBM, L2, shared memory, and registers;
- warps, blocks, occupancy, and coalescing;
- Tensor Cores, GEMM shapes, and low precision;
- arithmetic intensity and Roofline;
- operator fusion, kernel launches, and CUDA Graphs;
- the Triton programming model;
- Nsight Systems, Nsight Compute, and PyTorch Profiler;
- benchmark hygiene: warmup, synchronization, repetitions, and statistics.

### Read / build

- [`PERF/ROOFLINE.pdf`](../PERF/ROOFLINE.pdf)
- [`PERF/TRANSFORMERINFER.pdf`](../PERF/TRANSFORMERINFER.pdf)
- [`COURSE/CUDA-PROGRAMMING-GUIDE.pdf`](../COURSE/CUDA-PROGRAMMING-GUIDE.pdf)
- [`COURSE/CUDA-BEST-PRACTICES.pdf`](../COURSE/CUDA-BEST-PRACTICES.pdf)
- `RESOURCES/repos/gpu-mode-lectures`
- Reuse `/home/junyao/code/study/tutorials/gpu-systems/triton`.
- Reuse `/home/junyao/code/study/tutorials/llm/cs336/assignment2-systems`.

### Build

1. Benchmark PyTorch matmul, softmax, and LayerNorm.
2. Plot arithmetic intensity and achieved bandwidth.
3. Implement fused softmax in Triton.
4. Profile an attention kernel and explain why each optimization succeeds or fails.

### Exit gate

Given a slow kernel, state a falsifiable bottleneck hypothesis and test it with a counter-sweep.

---

## Core Module 3 — Modern Decoder Architecture as a System

**Goal**

Do not memorize model names. Study how each architecture delta changes execution and state.

### Topics

- RoPE and position handling;
- RMSNorm and SwiGLU;
- MHA → GQA → MQA;
- MLA's compressed representation and decode path;
- dense → MoE: active parameters, expert routing, and all-to-all;
- multi-token prediction / MTP;
- dynamic depth / conditional layer execution;
- diffusion / masked iterative generation;
- long context and native sparse attention;
- long-output workloads from reasoning models.

### Read

- [`FOUNDATION/LLAMA2.pdf`](../FOUNDATION/LLAMA2.pdf)
- [`ARCHITECTURE/README.md`](../ARCHITECTURE/README.md)
- [`ARCHITECTURE/DEEPSEEK-V2.pdf`](../ARCHITECTURE/DEEPSEEK-V2.pdf)
- [`ARCHITECTURE/DEEPSEEK-V3.pdf`](../ARCHITECTURE/DEEPSEEK-V3.pdf)
- [`ARCHITECTURE/QWEN3.pdf`](../ARCHITECTURE/QWEN3.pdf)
- [`03-architecture-taxonomy.md`](03-architecture-taxonomy.md)
- [`FOUNDATION/DEEPSEEKR1.pdf`](../FOUNDATION/DEEPSEEKR1.pdf)
- [`ATTENTION/NSA.pdf`](../ATTENTION/NSA.pdf)

### Required output

Build an architecture cost table comparing Llama-like GQA and DeepSeek-like MLA+MoE across prefill
and decode, including FLOPs, weight bytes, KV bytes, communication, and viable parallel strategies.

### Exit gate

Given a new model configuration, predict memory consumption and primary serving risks before
running it.

---

## Core Module 4 — Single-Node Inference Engine

**Goal**

Understand vLLM and SGLang as collections of execution, memory, and scheduling decisions rather
than black-box servers.

### 4.1 Execution

- prefill, decode, and append;
- continuous / iteration-level batching;
- ragged requests and token budgets;
- chunked prefill;
- CUDA Graphs and dynamic shapes.

Read: [`SERVING/ORCA.pdf`](../SERVING/ORCA.pdf),
[`SERVING/VLLM.pdf`](../SERVING/VLLM.pdf),
[`SERVING/SARATHI.pdf`](../SERVING/SARATHI.pdf).

### 4.2 KV memory

- paged KV and virtual memory;
- fragmentation and copy-on-write;
- prefix/radix caches;
- eviction, offload, and prefetch;
- KV quantization / selection.

Read: [`CACHE/KIVI.pdf`](../CACHE/KIVI.pdf),
[`CACHE/VATTENTION.pdf`](../CACHE/VATTENTION.pdf),
[`SERVING/SGLANG.pdf`](../SERVING/SGLANG.pdf).

### 4.3 Attention kernels

- tiled exact attention;
- prefill/decode kernel shape differences;
- paged/ragged KV layouts;
- indexing and I/O costs of sparse attention.

Read: [`ATTENTION/FLASHATTN.pdf`](../ATTENTION/FLASHATTN.pdf) →
[`ATTENTION/FLASHATTN2.pdf`](../ATTENTION/FLASHATTN2.pdf) →
[`ATTENTION/FLASHINFER.pdf`](../ATTENTION/FLASHINFER.pdf).

### 4.4 Compression and fewer serial steps

- weight-only, weight-activation, and KV quantization;
- calibration, outliers, and quality metrics;
- draft, verification, and acceptance in speculative decoding;
- why optimization outcomes change with batch and context.

Read: [`QUANT/SMOOTHQUANT.pdf`](../QUANT/SMOOTHQUANT.pdf),
[`QUANT/AWQ.pdf`](../QUANT/AWQ.pdf),
[`SPEC/SPECDECODE.pdf`](../SPEC/SPECDECODE.pdf),
[`SPEC/EAGLE3.pdf`](../SPEC/EAGLE3.pdf).

The complete speculative-decoding collection remains in
[`SPEC/README.md`](../SPEC/README.md); it does not need to be read linearly as part of the core path.

### Build

Implement a simplified simulator or mini-engine containing at least:

- request arrivals;
- token-level continuous batching;
- a block-based KV allocator;
- FCFS and one SLO-aware or cache-aware policy;
- TTFT, ITL, E2E, and goodput outputs.

### Exit gate

Trace a request through the main vLLM or SGLang path from API → scheduler → model runner → KV
manager → attention backend, then modify one policy and run an A/B test.

---

## Core Module 5 — Serving Systems

**Goal**

Move from "one fast GPU" to a system that satisfies SLOs under dynamic workloads.

### Topics

- arrival processes and prompt/output length distributions;
- queueing, head-of-line blocking, and admission;
- tail latency and goodput;
- cache-aware routing;
- prefill/decode interference;
- PD disaggregation and KV transfer;
- migration, elasticity, and multi-tenancy;
- simulator validity and trace-driven evaluation.

Use [`05-inference-systems-taxonomy.md`](05-inference-systems-taxonomy.md) as the layer-by-layer checklist.

### Read in order

1. [`SERVING/ORCA.pdf`](../SERVING/ORCA.pdf) — iteration-level scheduling;
2. [`SERVING/VLLM.pdf`](../SERVING/VLLM.pdf) — KV memory;
3. [`SERVING/SARATHI.pdf`](../SERVING/SARATHI.pdf) — chunked prefill;
4. [`SERVING/DISTSERVE.pdf`](../SERVING/DISTSERVE.pdf) — PD disaggregation;
5. [`CACHE/MOONCAKE.pdf`](../CACHE/MOONCAKE.pdf) — KV-centric disaggregation;
6. [`SERVING/PREBLE.pdf`](../SERVING/PREBLE.pdf) — distributed prefix-aware scheduling;
7. [`SERVING/VIDUR.pdf`](../SERVING/VIDUR.pdf) — simulation.

### Build

- Compare vLLM and SGLang under the same workload.
- Sweep arrival rate and plot latency-throughput and goodput curves.
- Locate the saturation knee.
- Predict one configuration with Vidur or a custom simulator, then calibrate its error against a
  real run.
- Model the PD-disaggregation break-even point across network bandwidths.

### Exit gate

For every claimed speedup, identify its workload, SLO, baseline, hardware, and failure region.

---

## Core Module 6 — Distributed Inference and MoE

**Goal**

Make quantitative model-sharding and communication decisions.

### Topics

- data, tensor, pipeline, context, and expert parallelism;
- all-reduce, all-gather, reduce-scatter, and all-to-all;
- NVLink/NVSwitch, PCIe, and InfiniBand/RDMA;
- latency and bandwidth terms;
- topology-aware placement;
- communication-compute overlap;
- MoE token dispatch/combine, expert batches, and load imbalance;
- wide expert parallelism and low-latency kernels.

For MoE, follow [`04-moe-deep-dive.md`](04-moe-deep-dive.md) from router to grouped GEMM and
expert-parallel communication; do not skip directly to an end-to-end throughput number.

### Read / inspect

- [`PARALLEL/MEGATRON.pdf`](../PARALLEL/MEGATRON.pdf)
- [`MOE/SWITCH.pdf`](../MOE/SWITCH.pdf)
- [`MOE/DEEPSPEEDMOE.pdf`](../MOE/DEEPSPEEDMOE.pdf)
- [`MOE/MEGABLOCKS.pdf`](../MOE/MEGABLOCKS.pdf)
- `RESOURCES/repos/deepep`
- `RESOURCES/repos/tensorrt-llm`

### Exit gate

Given a model, GPU count, and topology, derive the memory and communication costs of TP/PP/EP and
explain why the best configuration changes with prefill/decode, batch, and SLO.

---

## Core Module 7 — Frontier and Research Loop

**Goal**

Complete a compact research cycle with the rigor of a systems-paper project.

### 2026 frontier snapshot

These papers are problem samples, not a required linear reading list:

- [`FRONTIER/SERVEGEN.pdf`](../FRONTIER/SERVEGEN.pdf): realistic language, multimodal, and
  reasoning workloads;
- [`FRONTIER/STRATA.pdf`](../FRONTIER/STRATA.pdf): tiered cache and I/O for long context;
- [`FRONTIER/ECHO.pdf`](../FRONTIER/ECHO.pdf): native sparse-attention KV offload/prefetch;
- [`FRONTIER/SYMPHONY.pdf`](../FRONTIER/SYMPHONY.pdf): compute-memory disaggregation;
- [`FRONTIER/ECOSERVE.pdf`](../FRONTIER/ECOSERVE.pdf): commodity clusters with weak interconnects;
- [`FRONTIER/LIBRA.pdf`](../FRONTIER/LIBRA.pdf): dynamic and imbalanced workloads;
- [`FRONTIER/BATCHGEN.pdf`](../FRONTIER/BATCHGEN.pdf): large-scale batch inference;
- [`FRONTIER/CONTINUUM.pdf`](../FRONTIER/CONTINUUM.pdf): multi-turn agents and tool gaps;
- [`FRONTIER/KVFLOW.pdf`](../FRONTIER/KVFLOW.pdf): workflow-aware prefix caching;
- [`FRONTIER/DROIDSPEAK.pdf`](../FRONTIER/DROIDSPEAK.pdf): cross-model KV reuse.

### Research loop

1. **Characterize**: prove that the bottleneck exists and provide a breakdown.
2. **Model**: derive a cost model and break-even condition.
3. **Baseline**: select a strong baseline rather than a weak reimplementation.
4. **Mechanism**: change one primary mechanism.
5. **Ablation**: explain where the benefit comes from.
6. **Stress**: sweep workloads, hardware/network, SLO, model, and quality.
7. **Negative result**: identify the failure region.
8. **Reproduce**: preserve commits, configurations, traces, logs, and statistical scripts.

Final deliverables:

- a reproducible artifact;
- a workshop-style report;
- end-to-end, breakdown, trade-off, and ablation figures;
- a clear explanation of why existing systems do not directly solve the gap.

---

## 4. How much training knowledge is required?

An inference-systems researcher needs to understand:

- how the tokenizer determines sequence length;
- pretraining loss and next-token distributions;
- why optimizer state makes training memory much larger than inference memory;
- how scaling laws change models and workloads;
- how SFT, preference optimization, and RLVR create new output-length and serving workloads;
- which quantization, speculative-drafter, and MoE-routing methods require additional training.

The following are not core prerequisites:

- the full RLHF engineering stack;
- every PPO/DPO derivation;
- large-scale data pipelines;
- distributed-training fault recovery.

Use the relevant CS336 lecture or assignment when one of these topics becomes necessary. Do not
pause the inference path to complete all of them preemptively.

---

## 5. Reading priority

### Tier A — Read closely and reproduce the concept

- Attention Is All You Need
- Roofline
- Efficiently Scaling Transformer Inference
- FlashAttention
- Orca
- vLLM / PagedAttention
- Sarathi-Serve
- DistServe
- SGLang
- Vidur
- Megatron-LM
- DeepSeek-V3 architecture and systems sections

### Tier B — Read closely when selecting the corresponding branch

- KV: KIVI, vAttention, Mooncake, Strata, and ECHO;
- quantization: SmoothQuant, AWQ, and QuaRot;
- speculation: SPECDECODE, EAGLE3, and TRIFORCE/MAGICDEC;
- MoE: DeepSpeed-MoE, MegaBlocks, and DeepEP;
- agentic: Continuum and KVFlow;
- sparse/long context: MInference and NSA.

### Tier C — Consult as references

Use the remaining papers, the complete CUDA manual, and the full TensorRT-LLM/llm-d codebases as
references. Do not read large repositories linearly.

---

## 6. Final competency standard

After completing the path, you should be able to:

- estimate weights, KV, FLOPs, communication, and cost/token from a model configuration;
- use a profiler to distinguish compute, bandwidth, overhead, communication, and queueing;
- explain and modify a scheduling or caching path in a mainstream inference engine;
- evaluate quality/performance trade-offs in quantization, speculation, and sparse attention;
- design SLO-aware, trace-driven serving experiments;
- build break-even models for TP/PP/EP and PD disaggregation;
- derive a systems gap from a real workload instead of inventing a problem from a technique name;
- complete a reproducible systems-research artifact.

The endpoint is not reading a fixed number of PDFs. It is the ability to complete this cycle for a
new bottleneck:

```text
measure → model → design → implement → evaluate → explain limits
```
