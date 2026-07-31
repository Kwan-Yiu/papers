# LLM Inference Bottleneck and Research Map

> Target: identify a paper-worthy systems problem, not merely reproduce a speedup.
>
> Snapshot: 2026-07-31

[Roadmap index](README.md) ·
[Overview](00-roadmap.md) ·
[Production reliability](08-production-reliability.md) ·
[Project catalog](10-research-projects.md) ·
[Competency gates](COMPETENCY-GATES.md)

---

Modern LLM inference has no permanent bottleneck. The limiting component is jointly determined by:

```text
architecture
× model size / precision
× prompt/output distribution
× batch and arrival rate
× cache reuse
× hardware/topology
× SLO
× runtime policy
```

A research question should take this form:

> Under workload `W`, hardware/topology `H`, and SLO `S`, component `C` becomes the critical path
> because of mechanism `M`. Design `D` changes bytes/FLOPs/serial steps/queueing, and no benefit is
> claimed beyond boundary `B`.

---

## Document Guide

| Sections | Focus |
|---|---|
| 1–2 | workload matrix and bottleneck classes |
| 3 | database-systems analogy map |
| 4–5 | research idea cards and topic selection |
| 6–7 | evidence progression and paper-quality evaluation |

---

## 1. Bottleneck matrix by workload

| Workload | Likely primary bottlenecks | Often-hidden secondary bottlenecks |
|---|---|---|
| short prompt + short output | launch/CPU/scheduler, weight bandwidth | tokenizer, sampling, network stack |
| long prompt + short output | prefill attention/GEMM, admission | KV allocation, chunking, TTFT queue |
| short prompt + long output | repeated weight reads, KV reads | output-length stragglers, ITL tail |
| long prompt + long output | compute + KV capacity/bandwidth | preemption, multi-tier state |
| high prefix reuse | cache index/load/routing | cache-hit transfer blocking, skew |
| RAG | reusable document composition | non-contiguous KV, retrieval/preprocessing |
| reasoning | long unpredictable output | fairness, speculation variance, cost |
| agentic | state lifecycle across tool gaps | TTL, pinning, resumption, burst fan-out |
| MoE | expert weight capacity + dispatch/GEMM | per-step skew, topology, small expert batches |
| multi-LoRA | adapter locality and batching | eviction, kernel diversity |
| multimodal | encode + long tokenized inputs | encode/prefill/decode coordination |
| offline batch | throughput and energy | stragglers, job scheduling, fairness |

---

## 2. Bottleneck classes

### B1 — Model-weight capacity

Symptoms:

- model cannot fit;
- high parallelism chosen only for capacity;
- low concurrency after weight load;
- MoE total expert weights dominate.

Techniques:

- weight quantization;
- offload;
- tensor/pipeline/expert parallelism;
- expert caching/replication;
- pruning/distillation.

Research gaps:

- topology-aware mixed-precision placement;
- dynamic hot-expert residency;
- quantized MoE grouped kernels;
- shared weight store across replicas/processes;
- fast scale-out without full cold load.

### B2 — Model-weight bandwidth

Symptoms:

- decode speed scales with aggregate HBM bandwidth;
- small batch;
- GEMMs behave matvec-like;
- compute units underutilized.

Techniques:

- batching;
- quantization;
- fusion;
- speculative decoding;
- parallelism across HBM domains.

Research gaps:

- SLO-aware batching near low load;
- mixed interactive/offline fill work;
- dynamic kernel/precision selection;
- weight-cache hierarchy for sparse experts/adapters.

### B3 — KV/state capacity

Symptoms:

- concurrency limited by context;
- frequent eviction/preemption;
- OOM despite weights fitting;
- prefix cache occupies useful request capacity.

Techniques:

- GQA/MQA/MLA and cross-layer KV sharing;
- paged/virtual KV;
- KV quantization/compression;
- sliding/sparse attention;
- linear/recurrent/hybrid attention state;
- tiered offload;
- admission control.

Research gaps:

- state admission as buffer-pool management;
- value-aware eviction;
- multi-tenant fairness;
- mixed KV + recurrent-state manager;
- precise global state directory.

### B4 — KV/state bandwidth and locality

Symptoms:

- long-context decode ITL grows with context;
- HBM read dominates;
- cache hit still has high load time;
- remote state transfer appears on critical path.

Research gaps:

- learned/predictive prefetch;
- query-dependent KV layout;
- remote-state batching;
- topology-aware cache routing;
- compressed transfer with fused consumption;
- separate hot metadata and cold values.

### B5 — Prefill compute

Symptoms:

- long prompt dominates TTFT;
- attention/GEMM tensor cores saturated;
- long prefill blocks decode.

Techniques:

- FlashAttention;
- chunked prefill;
- sparse attention;
- prompt compression;
- prefix reuse;
- context parallelism;
- PD disaggregation.

Research gaps:

- SLO-optimal chunk size control;
- prefix composition;
- sparse prefill with quality contracts;
- multimodal encode/prefill overlap;
- burst-aware prefill admission.

### B6 — Compiler, kernel launch, and CPU runtime

Symptoms:

- graph breaks or repeated compilation;
- excessive shape specialization;
- visible GPU gaps;
- small model/small batch;
- many short kernels;
- scheduler core saturated;
- CUDA Graph gives large gain.

Research gaps:

- dynamic-shape compilation for serving;
- graph/compiled-artifact caching across workload changes;
- compiler decisions aware of paged KV, MoE skew, and online batches;
- event-driven GPU scheduler;
- persistent execution;
- dynamic-shape graph capture;
- fused scheduling/sampling;
- Rust/C++ control plane;
- GPU-resident block/expert bookkeeping.

### B7 — Dynamic batching and queueing

Symptoms:

- throughput/latency cliff near saturation;
- long request blocks short;
- high variance despite stable kernels;
- mean improves but P99 regresses.

Research gaps:

- predicted remaining-work scheduling;
- deadline/slack scheduling;
- cache/expert/adapter-aware batching;
- online policy with robustness to prediction error;
- fairness under unknown output lengths;
- joint admission and autoscaling.

### B8 — Communication

Symptoms:

- multi-GPU scale efficiency collapses;
- frequent collectives;
- network/PCIe visible in ITL;
- one rank stalls all peers.

Sources:

- TP collectives;
- EP dispatch/combine;
- PP activations;
- KV transfer;
- context parallelism.

Research gaps:

- topology-aware parallel decomposition;
- overlap without SM starvation;
- low-latency small-message collectives;
- compressed activation/KV/expert traffic;
- portable GPU-initiated communication;
- congestion-aware inference placement.

### B9 — MoE imbalance and irregularity

See [`07-distributed-inference-moe.md`](07-distributed-inference-moe.md).

Research gaps:

- decode-first MoE kernels;
- expert-aware batching;
- dynamic replication;
- prefill-based routing prediction;
- joint KV/expert locality;
- workload-drift-aware placement.

### B10 — Speculative decoding critical path

Symptoms:

- low or variable acceptance;
- draft overhead;
- target verification shapes inefficient;
- benefit disappears under load.

Research gaps:

- workload-aware draft choice;
- scheduling across heterogeneous acceptance;
- draft/target disaggregation;
- memory-bounded speculation;
- adaptive tree width/depth;
- systems support for MTP and rollback.

### B11 — Prefix/cache routing

Symptoms:

- replicas independently hold redundant prefixes;
- load balancing causes cache misses;
- hot prefixes overload one replica;
- global directory is stale/expensive.

Research gaps:

- popularity-aware prefix replication;
- load/locality joint objective;
- distributed radix/hash index;
- partial/composable prefix reuse;
- privacy-aware cross-tenant dedup;
- cache-aware autoscaling.

### B12 — Operations and reliability

Symptoms:

- performance good in microbench but unstable in cluster;
- cold start dominates;
- one failed rank removes a wide parallel group;
- cancellation leaks compute/state;
- autoscaler oscillates.

Research gaps:

- low-cost request recovery;
- state-aware live migration;
- SLO-aware autoscaling with long startup;
- fault-domain-aware TP/EP placement;
- performance regression detection;
- cluster-wide causal observability.

---

## 3. Database-systems analogy map

| Database concept | LLM inference analogue | Research opportunity |
|---|---|---|
| buffer pool | HBM KV/state pool | admission, eviction, tiering |
| page table | KV block table | allocation, indirection, compaction |
| index | prefix/KV global directory | lookup, consistency, placement |
| cache hierarchy | HBM/DRAM/SSD/remote KV | promotion, prefetch, pinning |
| skew/hot keys | hot prefixes/experts/adapters | replication and load routing |
| query optimizer | parallelism/kernel configuration | cost-based plan selection |
| query scheduler | token/request scheduler | SLO, fairness, deadlines |
| materialized view | reusable prefix KV | maintenance and composition |
| disaggregated storage | remote KV/state service | bandwidth/latency/consistency |
| transaction cancellation | request cancellation across workers | resource reclamation |
| workload management | online/offline/multi-tenant mix | admission and quotas |
| cardinality prediction | output length / expert load prediction | robust scheduling |

This mapping is useful for intuition, but do not mechanically rename a DB technique. LLM state is append-heavy,
GPU kernels need regular shapes, and each decode token creates a strict sequential dependency.

---

## 4. Research idea cards

### R1 — Tiered KV cache as a storage engine

Hypothesis: separating metadata/hot blocks/cold blocks and using workload-aware admission can improve SLO goodput over
plain LRU offload.

Baselines:

- GPU-only paged KV;
- LRU CPU offload;
- LMCache/Mooncake-supported tier;
- recomputation.

Key ablations:

- admission;
- eviction;
- prefetch;
- tier;
- compression;
- remote index.

### R2 — Joint cache-aware and load-aware routing

Hypothesis: a cost model using reusable tokens, queue slack and transfer latency beats cache-only or load-only routing.

Failure boundary:

- low reuse;
- stale directory;
- hot-prefix concentration;
- weak transfer bandwidth.

### R3 — Agent-state lifecycle manager

Treat tool gaps as inactive transactions:

- pin, demote, evict, prefetch, expire;
- predict tool-return time;
- resume with SLO.

Compare keeping KV, offloading, recomputing and summarizing.

### R4 — Output-aware scheduler robust to prediction error

Use prediction intervals rather than a single output-length estimate. Optimize goodput and starvation under
heavy-tailed reasoning workloads.

### R5 — Decode-first MoE batcher

Group active sequences by predicted expert overlap while bounding extra queue/ITL. Evaluate against pure FCFS/token
budget scheduling.

### R6 — Joint KV/expert locality router

Replica choice considers both prefix state and likely expert load. Study when two locality objectives conflict.

### R7 — Prefill/decode asymmetric configuration search

Choose different:

- parallelism;
- precision;
- chunk size;
- kernel;
- cache tier;
- EP degree

for prefill and decode. Optimize under actual transfer/network constraints.

### R8 — Commodity-network disaggregation

Most disaggregation results assume strong interconnect. Study adaptive co-location vs separation over PCIe/Ethernet or
mixed fabrics, with an explicit break-even model.

### R9 — Portable kernel/backend selection

Learn or cost-model selection across attention/MoE/quant backends by shape, dtype and GPU. Include compilation/warmup
and performance drift across versions.

### R10 — Speculation-aware global scheduling

Scheduler allocates target verification capacity using acceptance prediction and SLO slack, rather than treating each
request as one-token decode.

### R11 — Prefix composition for RAG

Reuse independently cached document chunks and combine them without recomputing the entire prompt. Address positional
encoding, cross-segment attention and quality.

### R12 — State-aware fault recovery

Recover partially decoded requests using distributed KV/state replicas or recomputation checkpoints; quantify
availability vs steady-state overhead.

### R13 — Multi-tenant state isolation

Prevent cache timing/data leakage while preserving safe dedup. Study tenant-scoped hashing, encrypted remote tiers and
fair eviction.

### R14 — Inference trace and simulator calibration

Build a trace schema spanning queue, scheduler, blocks, kernels, collectives and SLO. Calibrate Vidur/LLMServingSim-like
models against hardware and quantify uncertainty.

### R15 — Energy-aware serving

Use workload phase and SLO slack to choose GPU frequency, precision, batching and placement. Report energy/token and
goodput, not only instantaneous power.

---

## 5. How to select one topic

Score each direction from 0–3:

| Criterion | Question |
|---|---|
| repeatable pain | can you reliably trigger the bottleneck? |
| accessible hardware | can you evaluate the relevant topology? |
| open baseline | is there code you can modify? |
| measurable mechanism | can you count bytes/FLOPs/queue/communication? |
| novelty | do current systems leave a clear gap? |
| generality | does it cover more than one model/trace? |
| falsifiability | can an evaluation prove the idea wrong? |
| quality semantics | can you state exact/approximate behavior? |
| implementation scope | can one researcher finish a strong prototype? |
| paper story | is there one coherent thesis? |

Recommended initial ranking for a database researcher:

1. tiered KV/state management;
2. cache-aware SLO scheduling;
3. agent-state lifecycle;
4. joint KV/expert locality;
5. workload/simulator calibration;
6. commodity-network disaggregation;
7. MoE placement/replication.

Kernel-only projects are valuable but require a different comparative advantage and deeper CUDA work.

---

## 6. Evidence Progression

Do not jump from idea to end-to-end benchmark.

```text
E0: analytic cost model
E1: trace shows bottleneck exists
E2: microbenchmark isolates mechanism
E3: component prototype changes predicted cost
E4: end-to-end single-node result
E5: online load + SLO goodput
E6: multi-model / multi-workload generality
E7: multi-node / failure / drift robustness
```

At every level record a falsification condition.

---

## 7. Paper-Quality Evaluation Checklist

- precise problem and non-goals;
- open-source baseline at pinned commit;
- exact model/config/precision;
- workload distributions and arrival process;
- hardware topology;
- TTFT/ITL/E2E/goodput/P99;
- memory and network bytes;
- profiling evidence;
- mechanism ablation;
- sensitivity and break-even analysis;
- quality/correctness;
- overhead when the opportunity is absent;
- robustness to burst, skew, drift, failure;
- reproducible scripts and raw results.

The research deliverable is not "X% faster". It is:

1. a validated bottleneck model;
2. a design with explicit invariants;
3. an implementation;
4. an evaluation that explains why/when it works;
5. a boundary where it stops working.

---

**Previous:** [`08-production-reliability.md`](08-production-reliability.md) ·
**Project catalog:** [`10-research-projects.md`](10-research-projects.md) ·
**Back to overview:** [`00-roadmap.md`](00-roadmap.md)
