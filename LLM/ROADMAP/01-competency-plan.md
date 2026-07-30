# Prerequisite-Driven Competency Plan

This is the execution layer of [`00-roadmap.md`](00-roadmap.md). Progress is controlled exclusively
by prerequisites, artifacts, and exit gates.

Use the first module whose prerequisites you satisfy but whose exit gate you do not. Several
independent branches may be pursued in parallel after the core inference-engine gate.

## Progression graph

```text
Baseline measurement
        |
        v
Tensor and decoder mechanics
        |
        +--------------------+
        |                    |
        v                    v
GPU cost model        Architecture deltas
        |                    |
        +----------+---------+
                   |
                   v
        Single-node inference engine
                   |
        +----------+-----------+----------------+
        |                      |                |
        v                      v                v
KV/state systems       Serving/scheduling   Compression/decoding
        |                      |                |
        +----------+-----------+----------------+
                   |
                   v
       Distributed inference fundamentals
                   |
          +--------+---------+
          |                  |
          v                  v
  Dense parallelism    MoE expert parallelism
          |                  |
          +--------+---------+
                   |
                   v
       Bottleneck-driven research loop
```

## Gate 0 — Reproducible baseline

### Prerequisites

- Python and shell fluency;
- basic PyTorch model execution;
- access to a local GPU, remote GPU, or simulator.

### Required work

- Run one current serving engine such as vLLM or SGLang.
- Record the exact model, tokenizer, precision, framework commit, GPU, driver, and CUDA version.
- Measure at least three workload shapes:
  - chat-like: short prompt and moderate output;
  - retrieval-like: long prompt and short output;
  - reasoning-like: moderate prompt and long output.
- Sweep arrival rate from low load through saturation.
- Record TTFT, ITL/TPOT, E2E latency, request throughput, input/output token throughput, peak
  memory, and SLO goodput.

### Required artifact

A baseline report containing:

```text
environment
exact reproduction command
workload distributions
latency percentiles
throughput and goodput curves
memory usage
saturation point
one profiler trace
```

### Exit gate

You can explain why the same engine alternates among compute-bound, bandwidth-bound, overhead-bound,
and queueing-bound regions as the workload changes.

## Gate 1 — Tensor and decoder mechanics

### Prerequisites

- Gate 0;
- matrix multiplication, vector spaces, projections, and softmax;
- basic probability, cross-entropy, and maximum likelihood.

### Required work

- Derive all tensor shapes in one decoder block.
- Implement causal self-attention, RMSNorm, RoPE, SwiGLU, and residual connections.
- Implement autoregressive generation both with and without a KV cache.
- Verify cached and uncached logits numerically.
- Train a tiny language model to separate training, prefill, and decode semantics.

### Required artifact

- a minimal decoder implementation;
- shape and FLOP annotations;
- cached/uncached correctness tests;
- a note explaining one-token generation from input IDs to sampled output.

### Exit gate

You can reconstruct a decoder block from a model configuration and derive:

```text
parameter bytes
activation bytes
KV bytes per token
attention FLOPs
FFN FLOPs
logits/sampling path
```

## Gate 2 — GPU cost model and profiling

### Prerequisites

- Gate 1;
- basic CPU/GPU memory hierarchy;
- comfort reading profiler timelines.

### Required work

- Build a Roofline-style model for GEMM, elementwise operations, reductions, attention, and decode.
- Distinguish HBM bandwidth, L2 reuse, shared memory, registers, occupancy, launch overhead, and
  synchronization.
- Profile prefill and decode separately.
- Implement or modify one Triton/CUDA kernel.
- Compare predicted and observed bottlenecks.

### Required artifact

- operator cost sheet;
- annotated Nsight Systems, Nsight Compute, or PyTorch trace;
- baseline and optimized kernel benchmark;
- at least one negative result showing where the optimization stops helping.

### Exit gate

Before changing code, you can predict whether the target is limited by arithmetic throughput,
memory traffic, launch overhead, synchronization, or insufficient parallelism.

## Gate 3 — Architecture as a systems cost

### Prerequisites

- Gates 1 and 2;
- [`03-architecture-taxonomy.md`](03-architecture-taxonomy.md).

### Required work

Build a delta sheet for:

- MHA → GQA → MQA;
- standard attention → MLA;
- full attention → sliding-window, sparse, or hybrid attention;
- dense FFN → sparse MoE;
- attention → recurrent/SSM/hybrid state;
- single-token head → MTP/speculative-friendly heads;
- fixed depth → dynamic depth;
- autoregressive → diffusion/masked iterative generation;
- BF16/FP16 → FP8/INT8/INT4-style execution.

For every delta, record:

```text
semantic change
persistent state
weight capacity
bytes per token
FLOPs per token
kernel shape
parallelism impact
prefill impact
decode impact
quality constraint
```

### Required artifact

One architecture comparison matrix backed by model configurations and at least two measured
controlled comparisons.

### Exit gate

Given an unfamiliar model configuration, you can predict the dominant serving costs before running
it and identify the code paths needed to verify the prediction.

## Gate 4 — Single-node inference engine

### Prerequisites

- Gates 2 and 3.

### Required work

Trace one request through:

```text
API
→ tokenizer
→ request queue
→ scheduler
→ batch assembly
→ model runner
→ attention backend
→ KV manager
→ logits/sampling
→ streaming response
```

Implement small versions of:

- token-level continuous batching;
- block-based KV allocation;
- request preemption and resumption;
- chunked prefill;
- prefix-cache lookup;
- one sampling path.

### Required artifact

- request execution-path note with exact source locations;
- scheduler simulator;
- KV block allocator and fragmentation experiment;
- prefill/decode profiler comparison.

### Exit gate

You can explain where time, memory, and synchronization are spent for every request lifecycle stage,
including CPU control-plane overhead and GPU idle gaps.

## Branch A — KV and inference-state systems

### Prerequisite

- Gate 4.

### Required work

- derive KV capacity under MHA, GQA, MQA, and MLA;
- compare contiguous and paged allocation;
- evaluate fragmentation and block-size sensitivity;
- implement or simulate prefix reuse and eviction;
- model host, peer-GPU, local-storage, and remote KV tiers;
- derive transfer/recompute break-even points;
- evaluate migration, preemption, and failure recovery.

### Exit gate

You can select a KV policy from workload locality, context distribution, bandwidth, capacity, and
SLO constraints rather than from cache hit rate alone.

## Branch B — Online serving and scheduling

### Prerequisite

- Gate 4.

### Required work

- model open-loop and closed-loop arrivals;
- compare FCFS, priority, SRPT-like, fairness-aware, and deadline-aware policies;
- quantify head-of-line blocking;
- study chunked prefill and decode interference;
- evaluate admission control and load shedding;
- report P50/P95/P99 TTFT, ITL, E2E, goodput, fairness, and starvation;
- compare simulation predictions with real measurements.

### Exit gate

You can identify a saturation knee, explain tail-latency collapse, and show whether a policy improves
goodput without silently shifting cost to another request class.

## Branch C — Compression and decoding

### Prerequisite

- Gates 3 and 4.

### Required work

- compare weight-only, weight-activation, and KV quantization;
- inspect calibration, outliers, scales, dequantization, and kernel support;
- model draft, verification, acceptance, and rollback in speculative decoding;
- compare model-based, head-based, feature-based, n-gram, and prompt-lookup proposals;
- report quality, memory, latency, throughput, and batch interaction.

### Exit gate

You can state the exact quality constraint and break-even region of a compression or decoding
mechanism, including workloads where it regresses.

## Gate 5 — Distributed inference fundamentals

### Prerequisites

- Gate 4;
- at least one of Branch A, B, or C;
- basic topology and collective-communication knowledge.

### Required work

- derive memory and communication for TP, PP, DP, CP/SP, and sequence parallelism;
- benchmark or model all-reduce, all-gather, reduce-scatter, all-to-all, and point-to-point transfer;
- distinguish PCIe, NVLink/NVSwitch, InfiniBand/RoCE, and storage paths;
- model prefill/decode disaggregation;
- measure serialization, synchronization, imbalance, and cross-node tails.

### Required artifact

- parallelism calculator;
- topology-aware collective model;
- measured-vs-predicted comparison;
- disaggregation break-even analysis.

### Exit gate

Given a model, workload, GPU topology, and SLO, you can choose a parallelism and placement strategy
and defend it quantitatively.

## Branch D — MoE architecture and expert parallelism

### Prerequisites

- Gates 3 and 5;
- [`04-moe-deep-dive.md`](04-moe-deep-dive.md).

### Required work

- implement or inspect top-1, top-2/top-k, expert-choice, shared-expert, and fine-grained routing;
- log tokens per expert, hotness, imbalance, dropped tokens, and max-rank tail;
- trace router → dispatch → grouped GEMM → combine;
- compare padded, grouped, block-sparse, fused, and quantized expert kernels;
- model EP dispatch/combine bytes and topology;
- evaluate placement, replication, offload, and expert caching;
- separate prefill and decode MoE behavior;
- evaluate expert-aware scheduling under real routing skew.

### Exit gate

You can explain why sparse active FLOPs do not imply sparse storage, memory traffic, communication,
or latency, and can identify which layer dominates under a given batch and topology.

## Gate 6 — Bottleneck-driven research loop

### Prerequisites

- Gate 5 or Branch D;
- one reproducible real-system baseline;
- [`06-bottleneck-research-map.md`](06-bottleneck-research-map.md).

### Required sequence

1. Characterize a real workload and identify the dominant resource or queue.
2. Reproduce the strongest relevant baseline.
3. State a falsifiable hypothesis.
4. Build one mechanism that attacks the measured bottleneck.
5. Run baseline, mechanism, ablation, stress, and negative-case experiments.
6. Explain quality and semantic constraints.
7. Identify the break-even boundary.
8. Package commands, configs, raw results, and analysis for reproduction.

### Required artifact

```text
problem statement
workload contract
bottleneck evidence
cost model
mechanism
strong baselines
ablation
stress tests
negative cases
limitations
reproduction instructions
```

### Exit gate

An independent reader can reproduce the main result and distinguish:

- observed evidence from assumptions;
- system improvement from model-quality change;
- average speedup from SLO goodput;
- mechanism benefit from workload selection;
- measured boundary from claimed generality.

## Hardware-independent progress

Lack of a multi-GPU machine does not block the early or middle gates.

- With one GPU: complete kernels, KV allocation, batching, profiling, and single-node serving.
- Without a GPU: complete cost models, schedulers, trace analysis, and simulator validation against
  published or previously captured measurements.
- With temporary remote GPUs: run only experiments whose code paths and measurement contracts are
  already validated.
- Before multi-node access: build analytical and simulation models, then select a small set of
  high-information validation points.

Do not treat downloading a large checkpoint as progress. Prefer small models that expose the target
mechanism and fit the required experiment.
