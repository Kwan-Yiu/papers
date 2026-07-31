# Research Project Catalog

> **Role:** bounded project formulations derived from measured inference bottlenecks
>
> **Selection rule:** choose by problem fit and available evidence, not by document order

[Roadmap index](README.md) ·
[Overview](00-roadmap.md) ·
[Bottleneck research](09-bottleneck-research.md) ·
[Competency gates](COMPETENCY-GATES.md)

---

Projects are grouped by fit with a database-systems background. Every project begins with workload
characterization before mechanism design. For 15 more detailed idea cards, a bottleneck matrix, and
an evidence progression, see
[`09-bottleneck-research.md`](09-bottleneck-research.md).

## P1 — Tiered KV Cache as a Storage Engine

**Why it fits**

The KV cache has evolved from a temporary tensor into a stateful working set spanning requests,
turns, and nodes. It exposes familiar database problems: pages, admission, eviction, prefetch,
tiering, indexing, and locality.

**Question**

How should placement, eviction, and prefetch be coordinated across HBM, CPU DRAM, SSD, and remote
memory while satisfying TTFT and ITL SLOs?

**Baseline**

vLLM/SGLang with LRU; Mooncake/LMCache-style remote caches.

**Minimum evaluation**

- real or synthetic prompt-prefix traces;
- HBM-budget and tier-bandwidth sweeps;
- hit rate, bytes moved, stall time, TTFT/P99, and goodput;
- LRU vs size-aware vs reuse-distance/prediction-aware policies.

**Read**

Mooncake → Preble → Strata → Symphony → Continuum/KVFlow.

---

## P2 — Program-Aware Scheduling for Agentic Workloads

**Question**

After a tool call creates an unknown-length gap, should the system retain, offload, or evict KV?
How can a request-level scheduler use a workflow DAG, turn identity, and predicted future reuse?

**Core tension**

Retaining cache improves next-turn TTFT but occupies HBM and reduces current batch capacity.
Offloading recovers capacity but adds I/O.

**Minimum evaluation**

- multi-turn and tool-gap traces;
- request-level FCFS/LRU;
- session affinity;
- TTL-aware or prediction-aware policies;
- job completion time, per-turn TTFT, HBM occupancy, and eviction waste.

**Read**

ServeGen → Continuum → KVFlow.

---

## P3 — Partial Disaggregation on Commodity Clusters

**Question**

On clusters with PCIe/Ethernet rather than NVLink/InfiniBand, where are the boundaries among
colocation, full PD disaggregation, and time-selective or traffic-selective partial disaggregation?

**Model**

```text
benefit = removed prefill/decode interference
cost    = KV serialization + transfer + queueing + placement imbalance
```

**Minimum evaluation**

- network-bandwidth and latency sweeps;
- prompt/output-ratio sweeps;
- colocated, chunked-prefill, full-disaggregation, and selective policies;
- TTFT/ITL goodput and bytes transferred.

**Read**

Sarathi → DistServe → Mooncake → EcoServe → Libra.

---

## P4 — Workload-Aware Speculative Decoding in Serving

**Question**

Should the optimal speculative draft length or tree adapt to batch, queue, context, KV pressure,
and SLO?

**Why it matters**

Single-request acceptance and speedup do not automatically translate into serving goodput. The
additional compute, KV state, and batch interactions introduced by verification can reverse the
benefit.

**Minimum evaluation**

- acceptance and draft/verification cost;
- batch, context, and request-rate sweeps;
- fixed policy vs queue-aware or KV-aware adaptive policy;
- exactness tests, ITL, goodput, and energy/token.

**Read**

Follow foundation → EAGLE → tree → long-context in
[`../SPEC/README.md`](../SPEC/README.md).

---

## P5 — MoE Expert Placement and Load Balancing

**Question**

How should expert placement, replication, token routing, and batch formation be coordinated to
reduce all-to-all traffic and hot-expert imbalance?

**Minimum evaluation**

- real or model-generated routing distributions;
- uniform, skewed, and drifting workloads;
- static placement vs frequency-aware or online policies;
- bytes, link utilization, expert batch size, stragglers, and ITL.

**Read / code**

[`07-distributed-inference-moe.md`](07-distributed-inference-moe.md) → DeepSpeed-MoE → MegaBlocks → DeepSeek-V3 →
`RESOURCES/repos/deepep`.

**Difficulty**

High. A complete end-to-end result requires multiple GPUs or RDMA, but the work can begin with a
trace-driven simulator.

---

## P6 — Shape- and Workload-Portable Kernel Selection

**Question**

How can a system cheaply select or generate correct, fast attention/GEMM kernels across prefill,
decode, paged/ragged KV, GQA/MLA, and different GPUs?

**Minimum evaluation**

- shape distributions from real serving traces;
- Triton, FlashInfer, and vendor-kernel baselines;
- latency, compile/tuning cost, workspace, and correctness;
- offline tuning vs online adaptation.

**Read / code**

FlashAttention 1–3, FlashInfer, GPU Mode, and CS336 systems.

**Difficulty**

High. First prove that fixed selection fails across the observed shape distribution, then automate
the decision.

---

## Project filter

Before implementation begins, every answer below must be yes:

1. Can a real or credible trace prove that the bottleneck exists?
2. Can the failure conditions of existing methods be stated?
3. Is at least one strong open-source baseline available?
4. Can the core evidence be produced on available hardware?
5. Are there end-to-end metrics rather than only microbenchmarks?
6. Can correctness or output quality be checked?
7. Is the mechanism more general than tuning a few parameters?
8. Can the evaluation include ablations and negative cases?

P1 and P2 are the strongest starting points for a database researcher. They make direct use of
caching, storage, scheduling, and workload-modeling experience and can produce solid evidence on a
single machine plus a simulator.

---

**Previous:** [`09-bottleneck-research.md`](09-bottleneck-research.md) ·
**Competency gates:** [`COMPETENCY-GATES.md`](COMPETENCY-GATES.md) ·
**Back to overview:** [`00-roadmap.md`](00-roadmap.md)
