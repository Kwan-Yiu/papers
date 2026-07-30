# LLM Inference Systems Taxonomy

> Scope: single request → GPU kernel → multi-node production serving.
>
> Snapshot: 2026-07-30
> Companion: [`06-bottleneck-research-map.md`](06-bottleneck-research-map.md)

An LLM serving stack is not merely a model behind an HTTP server. The complete path is:

```text
client
→ gateway / auth / quota
→ tokenize
→ model & replica routing
→ admission / queue
→ prefill scheduling
→ model execution
→ KV/state placement
→ iterative decode scheduling
→ sampling / structured decoding
→ detokenize / stream
→ metrics / autoscaling / fault handling
```

The first rule of systems research is to identify the layer being optimized and determine whether
the cost was merely shifted to another layer.

---

## 1. Workload taxonomy

### 1.1 Request shape

| Workload | Input | Output | Typical pressure |
|---|---:|---:|---|
| chat | short–medium | short–medium | interactive ITL |
| long-document QA | very long | short | TTFT, KV capacity |
| code generation | medium | long | decode time, stragglers |
| reasoning | medium | very long/variable | output-length uncertainty |
| summarization | long | medium | prefill compute |
| embedding/reranking | batched input | no autoregressive output | compute throughput |
| RAG | repeated system/docs + query | medium | prefix reuse, composition |
| multi-turn agents | repeated prefixes + tool gaps | variable | state lifetime and reactivation |
| offline batch | broad | broad | throughput/cost, relaxed latency |
| multi-LoRA | shared base, many adapters | variable | adapter placement and batching |
| multimodal | image/audio/video tokens | variable | encode/prefill/decode separation |

### 1.2 Arrival process

Measure or generate:

- constant concurrency;
- open-loop Poisson;
- bursty / self-similar arrivals;
- diurnal trace;
- correlated agent fan-out;
- batch submission;
- retry storms;
- priority classes.

Closed-loop concurrency benchmarks hide queue overload because a client sends a new request only after one finishes.

### 1.3 Prefix structure

Classify:

- no shared prefix;
- identical system prompt;
- tree-structured chat history;
- shared retrieved documents;
- overlapping document chunks;
- multi-turn continuation;
- cross-request generated-prefix reuse;
- cross-replica or cross-model reuse.

Hit rate alone is insufficient. Record reusable token length, load latency, cache tier and whether reuse blocks prefill.

### 1.4 Output uncertainty

Schedulers rarely know output length exactly. Evaluate:

- fixed length;
- empirical distribution;
- heavy tail;
- early stop;
- tool-call interruption;
- speculative acceptance variability;
- structured-output constraints.

---

## 2. Service-level metrics

### 2.1 Latency

```text
queue_time       = admitted - arrival
tokenization     = queued - arrival portions
TTFT             = first_token - arrival
prefill_latency  = end_prefill - start_prefill
ITL_i            = token_i - token_{i-1}
TPOT             = decode_duration / generated_tokens
E2E              = final_token - arrival
```

Distinguish:

- server-side vs client-observed;
- warm vs cold;
- streaming flush delay;
- P50/P95/P99;
- per-request vs per-token percentiles.

### 2.2 Throughput

Report separately:

- input tokens/s;
- output tokens/s;
- total tokens/s;
- requests/s;
- useful tokens/s after rejected speculative tokens;
- per-GPU and per-dollar rates.

### 2.3 Goodput

```text
goodput = completed requests satisfying all declared SLOs / second
```

SLO can include:

- TTFT bound;
- ITL bound;
- E2E bound;
- deadline;
- accuracy/quality;
- availability.

An optimization that raises peak tokens/s while violating interactive ITL may reduce goodput.

### 2.4 Efficiency

- HBM bytes/output token;
- FLOPs/output token;
- interconnect bytes/token;
- KV bytes/request;
- energy/token;
- GPU-hours or cost per million tokens;
- utilization by phase;
- wasted compute from padding, preemption, rejected drafts or dropped work.

---

## 3. Request lifecycle

### Stage A — Frontend

Functions:

- auth, quota, rate limits;
- request validation;
- chat-template rendering;
- tokenization;
- model/adapter selection;
- structured-output grammar compilation;
- cancellation tracking.

Potential bottlenecks:

- CPU tokenizer;
- Python/event-loop overhead;
- JSON serialization;
- grammar construction;
- too many connections;
- backpressure propagation.

### Stage B — Routing and admission

Inputs:

- replica load;
- queue;
- KV/prefix locality;
- adapter locality;
- expert topology/load;
- SLO/deadline;
- memory headroom;
- predicted length;
- failure domain.

Actions:

- accept/reject/defer;
- select model and replica;
- choose prefill/decode workers;
- reserve capacity;
- prioritize class.

### Stage C — Prefill

Work:

- build Q/K/V or other state for prompt;
- compute prompt activations;
- allocate/cache state;
- produce first-token logits.

Characteristics:

- many tokens in parallel;
- attention can be quadratic in prompt length;
- large GEMMs, often compute-oriented;
- long request can block short requests without chunking.

### Stage D — Decode

Repeated loop:

```text
select active requests
→ build token batch
→ one model step
→ update KV/state
→ sample
→ finish/preempt/continue
```

Characteristics:

- one/few query tokens per sequence;
- repeated model-weight reads;
- context-dependent KV reads;
- kernel launch and CPU scheduling visible;
- synchronization each iteration;
- variable completion causes dynamic batch shape.

### Stage E — State retirement

When request ends/cancels:

- free blocks;
- retain reusable prefix;
- persist multi-turn state;
- update global index;
- publish cache events;
- release adapter/workspace;
- account usage.

Lifecycle bugs can leak memory or expose stale/cross-tenant state.

---

## 4. Scheduler taxonomy

### 4.1 Static batching

Wait for a batch, pad, run together.

Good for homogeneous offline workloads; poor for variable autoregressive completion.

### 4.2 Iteration-level / continuous batching

At each decode iteration, remove finished requests and insert new work.

Benefits:

- less padding;
- higher utilization;
- dynamic concurrency.

Costs:

- scheduler runs frequently;
- shapes change;
- graph capture and memory planning become harder;
- admission can harm existing requests' ITL.

### 4.3 Chunked prefill

Split a long prompt into token chunks and interleave with decode.

Control variables:

```text
max_num_batched_tokens
prefill_chunk_size
decode_priority
number_of_partial_prefills
long_prompt_threshold
```

Trade-offs:

- smaller chunks protect ITL;
- too small → more scheduling/launch overhead;
- too large → head-of-line blocking;
- attention work and KV allocation span iterations;
- chunk boundaries affect kernel shapes and prefix-cache behavior.

### 4.4 Prefill-first vs decode-first

- prefill-first improves admission/TTFT throughput but may hurt existing decode ITL;
- decode-first protects active sessions but can starve new requests;
- token-budget scheduling seeks a controlled balance.

### 4.5 Priority/deadline scheduling

Possible keys:

- arrival time / FCFS;
- shortest predicted remaining processing time;
- earliest deadline;
- TTFT urgency;
- decode ITL slack;
- customer priority;
- cache hit/locality;
- output-length prediction.

Length prediction error and starvation must be evaluated.

### 4.6 Preemption

Methods:

- swap KV/state to CPU;
- discard and recompute;
- stop admitting prefill;
- migrate request;
- checkpoint recurrent state;
- partial sequence suspension.

Costs:

- copy or recomputation;
- memory fragmentation;
- latency spikes;
- cache/index consistency;
- speculative state rollback.

### 4.7 Multi-LoRA batching

Scheduler considers:

- adapter residency;
- adapter rank/size;
- per-batch adapter diversity;
- fused multi-LoRA kernel;
- load/eviction;
- base-model cache sharing.

### 4.8 Structured-output scheduling

Grammar-constrained decoding adds:

- per-request automaton state;
- logits mask generation;
- CPU/GPU synchronization;
- variable allowed-token sets;
- batching incompatibilities.

---

## 5. Memory and state taxonomy

### 5.1 Memory classes

| State | Lifetime | Mutability | Sharing |
|---|---|---|---|
| model weights | deployment | immutable | all requests |
| quant scales/metadata | deployment | immutable | all requests |
| adapters | tenant/model session | immutable/read-mostly | selected requests |
| KV cache | request/prefix | append-only then immutable blocks | prefix-shareable |
| recurrent/SSM state | request step | mutable | snapshot-shareable with care |
| activation workspace | kernel/iteration | temporary | reusable buffer |
| graph workspace | shape/config | planned | runtime-shared |
| sampling/grammar state | request | mutable | not generally shared |

### 5.2 Contiguous KV allocation

Reserve/grow contiguous tensors per sequence.

Problems:

- over-reservation;
- fragmentation;
- relocation/copy;
- variable lengths.

### 5.3 Paged/block KV

Divide KV into fixed-size logical blocks and map sequences through block tables.

Benefits:

- non-contiguous physical allocation;
- low external fragmentation;
- copy-on-write prefix sharing;
- easier block-level eviction/migration.

Costs:

- indirection;
- block-table metadata;
- internal fragmentation at tails;
- attention kernel must understand pages;
- block size trade-off.

### 5.4 Virtual-memory-backed KV

Reserve virtual address space and map physical pages on demand, preserving contiguous virtual tensors.

Trade-off:

- reuse standard kernels more easily;
- OS/driver mapping overhead;
- page-size and allocator behavior;
- portability;
- mapping/unmapping synchronization.

### 5.5 Prefix cache

Key design dimensions:

- hash key: tokens, model, adapter, position/mask, dtype;
- data structure: hash map, radix tree, trie;
- block granularity;
- exact vs approximate match;
- copy-on-write;
- eviction;
- tenant/security boundary;
- replica/global directory;
- invalidation/versioning.

### 5.6 Multi-tier state

Potential tiers:

```text
GPU HBM
→ host DRAM
→ local SSD
→ remote memory / KV store
→ recompute
```

Policy dimensions:

- admission;
- promotion/demotion;
- eviction;
- prefetch;
- pinning;
- TTL;
- bandwidth reservation;
- global index;
- consistency and cancellation.

For each tier, report hit rate **and** service time distribution.

### 5.7 KV reduction methods

| Method | Changes | Exact? | Main risk |
|---|---|---|---|
| GQA/MQA/MLA | model architecture/state representation | model-defined | needs trained model/backend |
| lower KV precision | representation | approximate unless native | quality + dequant |
| token eviction | retained positions | approximate | lost information |
| sparse attention | accessed positions | usually approximate/model-defined | selection and gather |
| sliding window | attention semantics | model-defined | no distant attention |
| low-rank/compression | representation | approximate | decode overhead |
| offload | location only | exact | transfer latency |
| recomputation | regenerate state | exact if deterministic | compute latency |

---

## 6. Kernel/runtime taxonomy

### 6.1 Core operators

- GEMM / matvec;
- QKV projection;
- attention prefill;
- paged decode attention;
- MLA transforms;
- RMSNorm/LayerNorm;
- RoPE;
- gated FFN;
- fused MoE;
- sampling/top-k/top-p;
- quant/dequant;
- KV copy/reshape;
- collective communication.

### 6.2 Prefill attention

Optimization themes:

- IO-aware tiling;
- avoid materializing score matrix;
- causal masking;
- variable lengths;
- packed sequences;
- split-K / sequence parallel;
- FP8/low-precision;
- sparse patterns.

### 6.3 Decode attention

Characteristics:

- `Tq` usually one;
- many sequences with different `Tk`;
- paged/fragmented KV;
- low arithmetic intensity;
- KV dtype and head-sharing dominate bytes;
- batch metadata and launch overhead matter.

### 6.4 CUDA Graph / graph capture

Benefits:

- reduce launch overhead;
- stabilize execution;
- bypass Python/CPU gaps.

Challenges:

- dynamic batch sizes;
- dynamic sequence/block tables;
- memory address stability;
- MoE routing shapes;
- structured decoding;
- speculative branch lengths;
- multi-LoRA/adapters.

Strategies:

- pad to captured sizes;
- piecewise graphs;
- multiple graph buckets;
- graph only model core;
- dynamic-shape compilation.

### 6.5 Kernel selection

No single backend wins all shapes. Runtime may choose among:

- FlashAttention;
- FlashInfer;
- Triton kernels;
- vendor-specific generated kernels;
- CUTLASS/CuTe;
- ROCm AITER/CK;
- custom MLA/MoE kernels.

A research artifact must log selected backend, not just framework version.

---

## 7. Decoding algorithm taxonomy

### 7.1 Baseline autoregressive

- greedy;
- temperature sampling;
- top-k/top-p/min-p;
- beam search;
- best-of / parallel sampling.

Sampling policy changes output length, branching and speculative acceptance.

### 7.2 Draft-model speculative decoding

Small draft proposes multiple tokens; target verifies in parallel.

Cost model:

```text
speedup depends on
draft latency
accepted tokens per target step
verification efficiency
target batch interference
extra memory
```

### 7.3 Self-speculative

- layer skipping/early exit;
- intermediate features;
- MTP/Medusa heads;
- EAGLE-like feature draft;
- model-internal proposal.

### 7.4 Retrieval/prompt-based proposals

- n-gram;
- prompt lookup;
- suffix decoding;
- repeated code/text patterns.

Cheap and useful when workload has repetition; weak on novel text.

### 7.5 Tree verification

Verify multiple branches/tokens with tree attention. Must manage:

- position ids;
- causal mask/tree topology;
- temporary KV;
- accepted-path compaction;
- graph shapes.

### 7.6 Serving interaction

Speculation is not an isolated per-request optimization:

- draft consumes GPU/CPU capacity;
- verification changes token budget;
- acceptance varies by request;
- longer target steps affect fairness;
- rejected tokens are wasted work;
- draft/target can be disaggregated;
- batching can lower acceptance benefit.

---

## 8. Distributed inference taxonomy

### 8.1 Tensor parallelism (TP)

Shard matrix dimensions across devices.

Pros:

- reduces weight per rank;
- aggregates bandwidth/compute.

Costs:

- per-layer all-reduce/all-gather/reduce-scatter;
- decode has small messages and frequent collectives;
- topology critical.

### 8.2 Pipeline parallelism (PP)

Shard layers.

Pros:

- low activation traffic relative to TP;
- fits deeper models.

Costs:

- pipeline bubbles;
- microbatch scheduling;
- stage imbalance;
- KV partitioned across stages;
- interactive latency.

### 8.3 Data parallelism / replicas

Independent model replicas; cluster router chooses replica.

Pros:

- simple scale-out;
- no model-step cross-replica communication.

Costs:

- duplicate weights;
- fragmented KV/locality;
- load routing and autoscaling.

### 8.4 Expert parallelism

See [`04-moe-deep-dive.md`](04-moe-deep-dive.md). Adds two all-to-all-like phases per MoE layer.

### 8.5 Context/sequence parallelism

Shard long sequence work/state. Relevant to long prefill and very large contexts.

### 8.6 Prefill/decode disaggregation

Separate worker pools:

```text
prefill worker builds KV
→ transfer/register KV
→ decode worker continues generation
```

Potential benefits:

- independently scale compute-heavy prefill and bandwidth/latency-heavy decode;
- isolate interference;
- specialize kernels/configurations.

Costs:

- KV transfer;
- extra queue and routing;
- failure/cancellation semantics;
- cache/index consistency;
- weak interconnect can erase gains;
- balancing two coupled pools.

### 8.7 Encode/prefill/decode disaggregation

Multimodal systems may separate encoder from prefill/decode and manage embedding/feature caches in addition to KV.

### 8.8 Draft/target disaggregation

Place speculative draft and target separately. Trade draft resource isolation against communication and synchronization.

### 8.9 Remote KV/state layer

Separate state storage/transfer from compute. Components:

- block IDs and global index;
- memory registration;
- transfer engine;
- multi-tier allocator;
- routing hints;
- lifecycle/events;
- backpressure.

---

## 9. Cluster routing and orchestration

### 9.1 Routing signals

- queue/load;
- prefix/KV locality;
- model/adapter locality;
- predicted latency;
- token length;
- SLO/deadline;
- GPU type/topology;
- expert locality;
- failure/maintenance state;
- power/cost.

### 9.2 Routing objectives

Possible objective:

```text
minimize predicted completion time
+ queue cost
+ cache miss cost
+ migration/transfer cost
+ SLO violation penalty
+ imbalance penalty
```

No single scalar is correct for all services; declare weights and policy.

### 9.3 Flow control

Layers:

- client limits;
- gateway rate limits;
- per-model admission;
- per-worker token budgets;
- KV capacity guard;
- transfer/network backpressure;
- batch queue;
- priority/deadline protection.

Without end-to-end flow control, a fast frontend can overload GPU queues and inflate P99.

### 9.4 Autoscaling

Signals:

- request rate;
- input/output token rate;
- queued tokens;
- TTFT/ITL slack;
- KV occupancy;
- prefix hit rate;
- prefill/decode pool imbalance;
- expert load;
- GPU health.

Cold-start costs:

- container scheduling;
- weight download/load;
- graph/kernel warmup;
- KV directory registration;
- cache warmup.

Scale-to-zero is often incompatible with tight latency unless weights/state stay warm elsewhere.

### 9.5 Fault tolerance

Failure units:

- request/front-end process;
- model worker;
- one GPU/rank in TP/EP group;
- node;
- KV store/tier;
- network rail;
- scheduler/router.

Questions:

- can a partially decoded request resume elsewhere;
- what state must transfer;
- replay is deterministic under sampling;
- prefix directory has stale entries;
- cancellation reaches all phases;
- distributed group failure tears down how much capacity.

---

## 10. Framework/layer map

| Project | Primary layer | Best use in study | Do not assume |
|---|---|---|---|
| vLLM | execution engine + serving | scheduler, paged KV, model runner, distributed features | every backend/path behaves identically |
| SGLang | runtime + structured programs + serving | radix/prefix cache, scheduler, broad optimized paths | benchmark defaults match vLLM |
| TensorRT-LLM | NVIDIA optimized engine/runtime | low precision, generated kernels, vendor path | portability outside supported stack |
| FlashAttention | attention algorithm/kernel | IO-aware prefill kernels | it is a complete serving engine |
| FlashInfer | serving kernel library | paged/decode attention, sampling, MoE kernels | kernel microbench equals online serving |
| LMCache | KV/state layer | offload, sharing, remote/tiered KV | cache hit is free |
| Mooncake | KV-centric distributed serving/store | disaggregated KV and transfer | its topology matches commodity cluster |
| Dynamo | datacenter orchestration above engines | distributed routing, disaggregation, KV, scaling | it replaces vLLM/SGLang/TRT-LLM |
| llm-d | Kubernetes production stack above engines | KV-aware routing, wide EP, operational workflows | a single monolithic engine |
| AIBrix | pluggable inference infrastructure | autoscaling/routing/cache research | all components are production-equivalent |
| Sarathi-Serve | research serving engine | chunked prefill | current full production baseline |
| DistServe | research system | PD disaggregation | transfer/topology cost is universal |
| Vidur | simulator | trace/config exploration | simulator results replace hardware validation |
| ServeGen | workload generation | realistic workload traces | generated trace is ground truth for every service |

---

## 11. Local code-reading paths

### vLLM

- `RESOURCES/repos/vllm/vllm/v1/core`
- `RESOURCES/repos/vllm/vllm/v1/worker`
- `RESOURCES/repos/vllm/vllm/v1/attention`
- `RESOURCES/repos/vllm/vllm/v1/kv_offload`
- `RESOURCES/repos/vllm/vllm/distributed/kv_transfer`
- `RESOURCES/repos/vllm/vllm/model_executor/layers`
- `RESOURCES/repos/vllm/benchmarks`

### SGLang

- `RESOURCES/repos/sglang/python/sglang/srt/managers`
- `RESOURCES/repos/sglang/python/sglang/srt/managers/scheduler_components`
- `RESOURCES/repos/sglang/python/sglang/srt/mem_cache`
- `RESOURCES/repos/sglang/python/sglang/srt/layers/attention`
- `RESOURCES/repos/sglang/python/sglang/srt/layers/moe`
- `RESOURCES/repos/sglang/benchmark`

### KV/disaggregation

- `RESOURCES/repos/lmcache/docs/design`
- `RESOURCES/repos/lmcache/lmcache`
- `RESOURCES/repos/lmcache/csrc/storage_backends`
- `RESOURCES/repos/mooncake/mooncake-store`
- `RESOURCES/repos/dynamo/components`
- `RESOURCES/repos/dynamo/docs`
- `RESOURCES/repos/llm-d/guides`
- `RESOURCES/repos/distserve`

### Kernels

- `RESOURCES/repos/flash-attention`
- `RESOURCES/repos/flashinfer`
- `RESOURCES/repos/cutlass`
- `/home/junyao/code/study/tutorials/gpu-systems/triton`

### Benchmark/simulation

- `RESOURCES/repos/vidur`
- `RESOURCES/repos/servegen`
- `RESOURCES/repos/nccl-tests`
- `RESOURCES/repos/vllm/benchmarks`
- `RESOURCES/repos/sglang/benchmark`

---

## 12. Evaluation matrix

Every proposed optimization should cover the relevant axes:

| Axis | Minimum levels |
|---|---|
| prompt length | short / medium / long |
| output length | short / long / heavy-tail trace |
| arrival | low / near-saturation / overloaded / burst |
| prefix reuse | none / high / mixed |
| batch | offline large / online dynamic |
| model | dense / MoE if applicable |
| precision | baseline + optimized |
| hardware | topology and GPU generation stated |
| SLO | interactive + throughput target |
| metrics | TTFT, ITL, E2E, goodput, memory, network |
| statistics | warmup, repetitions, P50/P95/P99 |
| ablation | each mechanism independently |

### Required reproducibility record

```text
model name + exact config/checkpoint
framework commit
kernel/backend selection
driver/CUDA/ROCm/library versions
GPU count/type/topology
CPU/RAM/NIC/storage
launch command and environment
workload trace and seed
warmup and measurement window
all scheduling/cache/parallelism flags
```

---

## 13. Exit gate

You should be able to take one latency trace and decompose it into:

1. frontend/tokenizer;
2. queue/admission;
3. prefill;
4. state allocation/transfer;
5. decode model step;
6. scheduler/launch gaps;
7. sampling/streaming;
8. communication;
9. tail from imbalance or interference.

You should also be able to locate each component in at least one production engine and design a workload that can
falsify your bottleneck hypothesis.
