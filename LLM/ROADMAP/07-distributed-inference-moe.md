# Distributed Inference and Mixture-of-Experts

> Goal: understand dense and sparse distributed inference from topology to end-to-end serving.
>
> Snapshot: 2026-07-31
>
> Read after: [`06-kv-scheduling-serving.md`](06-kv-scheduling-serving.md)

[Roadmap index](README.md) ·
[Overview](00-roadmap.md) ·
[Serving systems](06-kv-scheduling-serving.md) ·
[Production reliability](08-production-reliability.md) ·
[Competency gates](COMPETENCY-GATES.md)

---

## Document Guide

| Sections | Focus |
|---|---|
| 0 | dense distributed-inference foundations |
| 1–3 | mathematical model, routing taxonomy, and load balancing |
| 4–7 | execution pipeline, prefill/decode behavior, kernels, and communication |
| 8–10 | parallelism composition, placement, offload, and online serving |
| 11–12 | architecture case studies and exact code-reading paths |
| 13 | validation ladder |
| 14–16 | research opportunities, invalid conclusions, and exit criterion |

---

## 0. Distributed Inference Foundations

### 0.1 Why distribute inference

Distribute a model or workload only for a declared reason:

- model weights do not fit on one accelerator;
- KV/state or workspace limits concurrency;
- one device cannot satisfy latency;
- replicas are needed for throughput or availability;
- prefill and decode require different resources;
- MoE experts exceed local capacity;
- state locality or hardware heterogeneity favors placement.

Every additional rank adds communication, synchronization, placement, and failure complexity.

### 0.2 Distributed cost model

For each parallel region, record:

```text
local parameter bytes
local activation/state bytes
compute per rank
messages and payload bytes
collective rounds
critical-path imbalance
temporary communication buffers
serialization and synchronization
```

A basic transfer estimate:

```text
transfer_time
≈ software_latency
 + payload_bytes / effective_bandwidth
 + contention
 + synchronization_tail
```

Effective bandwidth depends on message size, topology, peers, protocol, and concurrent compute.

### 0.3 Data parallelism / replicas

Each replica owns a full model and serves different requests.

Strengths:

- high aggregate throughput;
- failure/placement flexibility;
- no per-layer model collective.

Limits:

- full weights per replica;
- cache/model locality affects routing;
- one replica still must fit and meet latency.

For online inference, data parallelism usually means request-level replication rather than
training-gradient synchronization.

### 0.4 Tensor parallelism

Partition large matrices across ranks. Typical layer execution uses collectives such as:

- all-reduce;
- all-gather;
- reduce-scatter.

Benefits:

- divides weight capacity and compute;
- can reduce single-request latency when communication is fast.

Costs:

- per-layer synchronization;
- smaller local GEMMs;
- topology sensitivity;
- one slow rank delays the group.

### 0.5 Pipeline parallelism

Partition layers into stages:

```text
stage 0 → stage 1 → ... → stage P-1
```

Benefits:

- divides weight capacity;
- uses point-to-point activation transfer.

Costs:

- pipeline bubbles;
- stage imbalance;
- activation transfer;
- request/microbatch scheduling complexity;
- state ownership across stages.

### 0.6 Context and sequence parallelism

Partition sequence/token work where attention or long-context memory exceeds one device. Depending
on the method, ranks exchange:

- Q/K/V blocks;
- partial attention outputs/statistics;
- activations;
- KV/state.

The communication pattern differs between prefill and decode.

### 0.7 Expert parallelism

Partition MoE experts across ranks. Tokens move to selected experts and outputs return:

```text
route
→ dispatch / all-to-all
→ expert compute
→ combine / all-to-all
```

The rest of this document develops this path in detail.

### 0.8 Collectives

Know the semantics and likely use of:

| Primitive | Typical use |
|---|---|
| all-reduce | combine partial outputs |
| all-gather | assemble partitioned activations/weights |
| reduce-scatter | reduce and retain partitions |
| all-to-all | expert/token redistribution |
| broadcast | distribute shared data |
| point-to-point | pipeline/state transfer |

Benchmark latency and bandwidth across realistic message sizes. Large-message peak bandwidth does
not predict decode-time small-message latency.

### 0.9 Topology

Distinguish:

```text
within accelerator package
→ within node over NVLink/NVSwitch/xGMI
→ cross-socket PCIe
→ cross-node InfiniBand/RoCE/Ethernet
→ storage/state paths
```

Placement should keep frequent, latency-sensitive communication on stronger links when capacity and
failure constraints permit.

### 0.10 Parallelism composition

Real deployments combine axes:

```text
replicas × pipeline × tensor/context × expert parallelism
```

Verify that the product of degrees matches the physical rank count and that each axis has a clear
ownership/communication boundary.

### 0.11 Disaggregation

Possible separations:

- prefill workers versus decode workers;
- compute versus KV/state storage;
- encoder versus decoder;
- draft versus target model;
- router/control plane versus model workers.

Disaggregation is beneficial only when specialization/locality gains exceed transfer, queueing,
serialization, and failure-management cost.

### 0.12 Required distributed evidence

- per-rank memory ledger;
- parallelism calculator;
- topology diagram;
- collective/message-size model;
- measured-versus-predicted communication;
- load/straggler distribution;
- prefill/decode comparison;
- break-even boundary against a colocated or smaller-parallelism baseline.

---

## MoE Transition

The basic appeal of MoE is partially decoupling **model capacity** from **compute per token**:

```text
total parameters:     very large
activated parameters: only experts selected for this token
```

An inference system, however, must execute:

```text
route
→ count tokens per expert
→ permute/dispatch tokens
→ communicate across EP ranks
→ run many unequal expert GEMMs
→ communicate/combine results
→ restore token order
```

Therefore:

> MoE provides conditional computation, but not conditional storage.
> Sparse FLOPs do not imply sparse memory traffic, communication, or latency.

---

## 1. Minimal mathematical model

For token hidden vector `x ∈ R^d` and `E` experts:

```text
router_logits = W_r x
router_prob   = softmax(router_logits)
S(x)          = TopK(router_prob, k)
y             = Σ_{e ∈ S(x)} g_e(x) · Expert_e(x)
```

A gated expert commonly takes the form:

```text
Expert_e(x) = W_down,e (SiLU(W_gate,e x) ⊙ W_up,e x)
```

Parameters that must be recorded:

| Symbol | Meaning |
|---|---|
| `E` | routed expert count |
| `k` | experts selected per token |
| `Es` | always-on shared expert count |
| `d` | hidden dimension |
| `d_ff,e` | one expert intermediate dimension |
| `L_moe` | number of MoE layers |
| `C` | expert capacity, if capacity-limited |
| `P_total` | all parameters |
| `P_active` | parameters activated per token |
| `EP` | expert-parallel degree |
| `TPe` | tensor-parallel degree inside each expert |

Active-expert FLOPs are roughly proportional to `k × d × d_ff,e`, but system cost also includes
routing, permutation, communication, padding, quantization/dequantization, and combination.

---

## 2. MoE taxonomy

### 2.1 Where MoE appears

| Placement | Description | Systems implication |
|---|---|---|
| every FFN layer | all dense FFNs become MoE | maximum expert traffic |
| alternating layers | every other / periodic layer is MoE | dense and sparse kernels interleave |
| selected late layers | only subset of depth is MoE | layer-dependent placement |
| attention experts | experts also alter attention path | state and kernel complexity grows |
| modality experts | route by text/image/audio or learned tokens | heterogeneous expert shapes |
| shared + routed FFN | always-on shared experts plus sparse experts | dense compute remains on critical path |

### 2.2 Token-choice routing

Each token chooses experts.

Variants:

- top-1: Switch-style;
- top-2: GShard/Mixtral-style;
- top-k;
- threshold routing;
- group-limited top-k;
- node-limited routing;
- sigmoid vs softmax scores;
- normalized vs unnormalized top-k weights.

Pros:

- fixed `k` means predictable compute per token;
- simple semantic model.

Cons:

- expert load can be highly skewed;
- capacity overflow or padding;
- one hot expert can determine all-to-all completion time.

### 2.3 Expert-choice routing

Each expert chooses a fixed-capacity subset of tokens.

Properties:

- expert load is directly bounded;
- tokens may receive a variable number of experts;
- semantics and implementation differ from fixed top-k token choice;
- online autoregressive decode may not have a large enough token pool for attractive batching.

### 2.4 Shared experts

Shared experts always process tokens; routed experts capture specialized knowledge.

System consequences:

- shared expert behaves like a dense FFN on the critical path;
- shared expert can be replicated or tensor-parallelized separately;
- routed expert traffic is not the entire MoE layer cost;
- shared expert compute may overlap with routed dispatch/communication.

### 2.5 Fine-grained experts

Split large experts into more, narrower experts and select more of them.

Potential benefits:

- richer expert combinations;
- stronger specialization;
- more placement flexibility.

Systems costs:

- smaller GEMMs;
- more routing indices;
- more distinct experts touched per batch;
- higher risk that kernel/communication overhead dominates;
- top-k may be numerically larger even if active FLOPs stay similar.

PEER pushes this direction to an extreme by using product-key retrieval to select a few experts
from as many as a million tiny experts. Its systems relevance is not the headline expert count, but
the fact that expert indexing, metadata locality, retrieval cost, micro-GEMM efficiency, and expert
placement become first-class concerns. Do not reuse the cost model of tens or hundreds of large FFN
experts for retrieval-routed tiny experts.

### 2.6 Soft / dense mixtures

Soft MoE-like designs combine expert slots without hard sparse top-k. They are relevant algorithmically but do not
automatically give sparse inference. Keep them separate from sparse routed MoE in performance comparisons.

---

## 3. Routing and load balancing

### 3.1 Router pipeline

```text
1. compute router logits
2. optional bias / temperature / group mask
3. select top-k experts
4. compute routing weights
5. histogram tokens per expert
6. enforce capacity / drop / pad / reroute
7. build source↔destination indices
```

Router compute is small relative to expert GEMM, but selection, histogram and prefix-sum kernels can be visible in
decode because the token batch is small.

### 3.2 Capacity

Classic capacity approximation:

```text
capacity_per_expert ≈ capacity_factor × tokens × k / E
```

Choices:

- drop overflow tokens;
- residual bypass;
- route to backup expert;
- pad to capacity;
- dropless variable-size execution.

`capacity_factor` trades:

- correctness/quality;
- memory workspace;
- wasted padded compute;
- load balance;
- static shape convenience.

### 3.3 Auxiliary load-balancing loss

Training commonly adds a balance term so routing probability and actual assignment distribute across experts.

Risk:

- too weak → routing collapse/hot experts;
- too strong → model optimizes balance instead of specialization;
- training-balanced aggregate statistics do not guarantee per-request or per-step inference balance.

### 3.4 Auxiliary-loss-free balancing

Aux-loss-free strategies adjust per-expert routing bias using observed load rather than adding a strong auxiliary
training objective. For systems researchers, the important point is not only model quality:

- bias update changes expert popularity distribution;
- aggregate balance may still hide temporal bursts;
- inference traces should measure per-layer, per-step and per-domain routing;
- runtime cannot assume uniform routing just because the training paper reports balance.

### 3.5 Group- or node-limited routing

Router may constrain selection to experts within a group or limited nodes.

This is explicit algorithm–topology co-design:

- fewer cross-node destinations;
- potentially lower quality or routing flexibility;
- larger local hot spots;
- topology group must match actual NVLink/NVSwitch/RDMA layout.

### 3.6 What to log

For every MoE layer and decode step:

```text
tokens_per_expert
max / mean load
p50 / p95 / p99 expert load
coefficient_of_variation
entropy of assignments
number of active experts
number of remote destinations
overflow / dropped / rerouted tokens
router score margin between kth and (k+1)th expert
```

Also aggregate by:

- prompt vs decode;
- request/domain/language;
- context length;
- batch size;
- sampling temperature;
- model layer;
- replica and node;
- time window.

---

## 4. End-to-end execution pipeline

### 4.1 Local logical order

```text
hidden states
  → router logits / top-k
  → count + prefix sum
  → permute tokens by expert
  → expert MLPs
  → weighted combine
  → unpermute to original token order
```

### 4.2 Expert-parallel order

```text
local tokens
  → route to globally owned experts
  → dispatch all-to-all
  → local grouped expert GEMM
  → combine all-to-all
  → restore local token order
```

### 4.3 Byte model

Let:

- `N` be local tokens entering a MoE layer;
- `k` selected experts;
- `d` hidden width;
- `b_act` bytes per dispatched activation;
- `b_meta` metadata per assignment;
- `r_remote` fraction sent to remote ranks.

One-way logical activation traffic:

```text
dispatch_bytes ≈ N × k × d × b_act × r_remote
combine_bytes  ≈ similar order
metadata_bytes ≈ N × k × b_meta
```

The actual link bytes depend on collective algorithm, topology, local traffic and protocol overhead.

### 4.4 Latency model

```text
T_moe ≈ T_router
      + T_permute
      + T_dispatch
      + T_expert_compute
      + T_combine
      + T_unpermute
      - T_overlap
      + T_imbalance_tail
```

The max-loaded expert/rank often determines the layer barrier:

```text
T_expert_compute ≈ max_rank(compute_for_local_experts)
T_dispatch       ≈ max_rank(communication_path)
```

Average load is therefore insufficient.

---

## 5. Prefill and decode are different MoE workloads

### Prefill

- many tokens per request;
- larger expert batches;
- grouped GEMM has better arithmetic intensity;
- router distribution over a prompt may average out;
- communication is high-throughput oriented;
- overlap is easier;
- chunking changes expert batch and cross-request mixing.

### Decode

- one new token per active sequence per iteration;
- number of tokens ≈ active batch size;
- each token may touch `k` experts at every MoE layer;
- many experts receive tiny or zero batches;
- launch, sort and communication latency dominate;
- routing imbalance changes at every step;
- ITL is a repeated barrier, so microseconds matter.

### Consequence

MoE libraries often expose separate high-throughput and low-latency paths. Never evaluate only prefill or one large
synthetic token batch and conclude that online decode is solved.

---

## 6. Expert compute kernels

### 6.1 Naive per-expert GEMM

Launch one GEMM per active expert.

Failure mode:

- many launches;
- tiny `M` dimension;
- poor tensor-core utilization;
- CPU/launch overhead;
- difficult CUDA Graph capture.

Useful only as correctness baseline or when very few experts are active with large token counts.

### 6.2 Padded batched GEMM

Pad experts to a common capacity and batch the operation.

Trade-off:

- regular shape and easier kernels;
- wasted compute proportional to imbalance/padding;
- larger workspace;
- stable graph shape can help runtime overhead.

### 6.3 Grouped GEMM

Process a list of different expert matrix problems in one launch.

Key dimensions:

- number of groups;
- tokens per group (`M_e`);
- common or variable `N/K`;
- weight layout;
- quantization format;
- scheduling policy across SMs;
- empty experts;
- alignment and tile waste.

Grouped GEMM is a central MoE primitive, but its benefit depends on the distribution of `M_e`, not just total tokens.

### 6.4 Sparse block formulation

MegaBlocks-style approaches reformulate dropless MoE with block-sparse operations, avoiding token dropping and
reducing padding waste. Inspect:

- block size;
- token blocking strategy;
- sparse metadata;
- gather/scatter;
- block-sparse matrix multiply;
- load imbalance within blocks.

### 6.5 Fused MoE

Potential fusion:

```text
permute
→ gate/up projection
→ activation × gate
→ down projection
→ weighted combine
```

More aggressive implementations may overlap communication and compute or fuse dispatch/combine with expert kernels.

Risks:

- architecture/precision-specific;
- complex workspace and graph capture;
- difficult portability;
- benefits may vanish for different `E`, `k`, hidden size or GPU generation.

### 6.6 Quantized experts

MoE quantization has unique issues:

- total expert weights dominate capacity, so compression is attractive;
- each expert may have little calibration data;
- expert popularity is skewed;
- per-expert scales add metadata;
- small/irregular expert GEMM needs matching low-bit grouped kernels;
- dequant overhead may dominate tiny decode expert batches;
- shared and routed experts may want different precision.

Always report:

- total model bytes;
- active bytes per step;
- hot-expert cache residency;
- quant/dequant kernels;
- expert-wise and task-wise quality.

---

## 7. Dispatch and communication

### 7.1 Core operations

- count/histogram destinations;
- prefix sums and offsets;
- token permutation;
- all-to-all / all-to-all-v;
- low-precision dispatch;
- reverse combine;
- weighting and reduction;
- event/synchronization management.

### 7.2 Intra-node

Typical fabrics:

- PCIe;
- NVLink;
- NVSwitch;
- shared host memory or peer access.

Questions:

- direct peer copy or collective;
- topology-aware rank placement;
- how many SMs communication kernels occupy;
- communication overlaps tensor-core compute or competes with it;
- copy engine vs SM-driven transfer;
- CUDA Graph compatibility.

### 7.3 Inter-node

Typical stack:

- GPU Direct RDMA;
- NIC / rail topology;
- NCCL or custom communication;
- NVSHMEM / UCX / NIXL-like mechanisms;
- direct low-latency kernel path vs high-throughput buffered path.

Measure:

```text
message size distribution
logical vs physical bytes
per-peer fanout
serialization latency
link bandwidth
SM occupation by communication
QP / connection count
straggler rank
```

### 7.4 Hierarchical all-to-all

Possible structure:

```text
local aggregate
→ inter-node exchange
→ intra-node scatter
```

or the reverse depending on ownership. Benefits depend on routing locality, topology and message size. Hierarchical
communication can reduce network peers but adds local stages and synchronization.

### 7.5 Overlap

Overlap opportunities:

- shared expert compute with routed dispatch;
- local experts with remote token arrival;
- dispatch of one chunk with compute of another;
- combine of completed experts with remaining compute;
- dense attention of another microbatch with MoE communication.

Correct overlap analysis needs a timeline. `communication_time + compute_time` from isolated benchmarks does not show
critical-path overlap.

---

## 8. Parallelism composition

### 8.1 Expert parallelism (EP)

Experts are partitioned across ranks.

Benefits:

- total expert capacity scales with devices;
- each rank stores fewer experts.

Costs:

- dispatch/combine all-to-all;
- imbalance;
- topology sensitivity.

### 8.2 Expert tensor parallelism (`TPe`)

Shard an individual expert across ranks.

Use when:

- one expert is too large;
- sharding helps balance;
- expert batch is large enough.

Costs:

- collectives inside expert GEMM;
- can conflict with EP topology;
- fine-grained experts often make `TPe > 1` inefficient.

### 8.3 Tensor parallelism for dense layers

Attention/shared dense layers may use TP while routed experts use EP. The rank groups can be:

- identical;
- orthogonal;
- nested;
- topology-aware.

Group construction determines whether a layer changes communication domain between attention and MoE.

### 8.4 Data parallelism

Replicate the model/experts and split requests.

Inference benefits:

- independent replicas;
- no cross-replica token dispatch;
- routing/load balancing can select replicas.

Costs:

- duplicate total expert weights;
- lower memory efficiency;
- cache locality and load routing become cluster concerns.

### 8.5 Pipeline parallelism

Split layers into stages.

MoE-specific concerns:

- per-stage expert weight imbalance;
- variable stage time from routing;
- microbatch bubbles;
- KV state partition by layers;
- pipeline scheduling interacts with EP collectives.

### 8.6 Context/sequence parallelism

Long-context prefill can partition sequence, but MoE routing redistributes tokens again. Analyze both communication
dimensions and whether group transitions require reshaping/collectives.

### 8.7 A placement notation

Record a deployment as:

```text
replicas × PP × TP_dense × EP × TP_expert
```

and state topology mapping:

```text
within NVSwitch domain
across nodes / rails
```

"16 GPUs" is not a reproducible parallel configuration.

---

## 9. Expert placement, replication and offload

### 9.1 Static equal-count placement

Put equal numbers of experts per rank.

Simple, but assumes expert popularity and compute are uniform.

### 9.2 Popularity-aware placement

Place hot experts to reduce remote traffic or balance rank load.

Need to answer:

- popularity measured over what time/domain;
- model layer handled independently or jointly;
- migration frequency and cost;
- robustness to workload drift;
- conflict with topology and memory capacity.

### 9.3 Expert replication

Replicate hot experts on multiple ranks.

Benefits:

- less contention;
- shorter routes;
- better balance.

Costs:

- more HBM;
- routing must choose replica;
- updates are irrelevant for frozen inference but deployment consistency remains;
- replica selection may harm batching.

### 9.4 Expert caching/offload

Store some experts in CPU/SSD/remote memory and load on demand.

Hard questions:

- can next-layer experts be predicted early;
- miss latency relative to ITL;
- prefetch accuracy;
- hot set stability;
- eviction policy;
- expert weight granularity;
- overlap with current-layer compute;
- what happens under concurrent requests.

Pre-gating changes the algorithm so future expert needs can be known earlier; this is a good example of algorithm–system
co-design rather than a transparent runtime optimization.

### 9.5 Disaggregated experts

Remote expert serving separates dense backbone and experts or assigns expert services to other nodes.

Potential advantages:

- independent capacity scaling;
- centralized hot-expert replication;
- heterogeneous hardware.

Risks:

- a network RPC/collective on every MoE layer and decode step;
- failure amplification;
- synchronization/tail latency;
- backpressure;
- loss of grouped GEMM across locally batched tokens.

---

## 10. MoE-aware online serving

Traditional request scheduling sees:

```text
prompt length, output length, KV occupancy, SLO
```

MoE-aware scheduling may also need:

```text
predicted expert set
expert popularity signature
current per-rank expert queues
EP communication pressure
hot-expert residency
prefill vs decode expert batch opportunity
```

Research directions:

- group requests with complementary expert loads;
- batch requests sharing experts to enlarge GEMMs;
- route replicas by expert locality;
- use prefill routing as a predictor for decode;
- separate prefill/decode EP configurations;
- SLO-aware control of EP degree;
- expert-aware admission and backpressure;
- schedule shared expert compute to overlap communication.

Beware: grouping for expert locality can increase queueing delay or reduce KV prefix locality.

---

## 11. Architecture case studies

### GShard

- token-choice top-2 routing;
- capacity and overflow behavior;
- foundational large-scale conditional-compute/sharding design.

Study for routing/capacity concepts, not as the final modern inference implementation.

### Switch Transformer

- top-1 routing;
- simplified sparse routing;
- highlights training stability and capacity trade-offs.

### Mixtral

- eight FFN experts, top-2 per token in the original 8×7B design;
- good entry point because attention resembles Mistral and only FFN becomes sparse;
- useful controlled dense-vs-MoE code reading.

### DeepSeekMoE / DeepSeek-V2/V3

- fine-grained routed experts;
- shared experts;
- MLA reduces attention state while MoE expands parameter capacity;
- V3 reports auxiliary-loss-free balancing and MTP;
- serving therefore combines MLA kernels, expert parallelism, low-precision compute and communication co-design.

### Qwen MoE

- dense and MoE variants within one family;
- useful for controlled systems evaluations;
- inspect actual config because expert count/top-k differs across releases.

### GPT-OSS

- modern open-weight MoE reference;
- useful for comparing backend support across Transformers, vLLM, SGLang, TensorRT-LLM and Tutel;
- do not assume identical expert kernels or quant formats across engines.

---

## 12. Local repositories and exact reading paths

### Model semantics

- `RESOURCES/repos/transformers/src/transformers/models/mixtral`
- `RESOURCES/repos/transformers/src/transformers/models/qwen2_moe`
- `RESOURCES/repos/transformers/src/transformers/models/qwen3_moe`
- `RESOURCES/repos/transformers/src/transformers/models/deepseek_v3`
- `RESOURCES/repos/transformers/src/transformers/models/gpt_oss`
- `RESOURCES/repos/deepseek-v3/inference/model.py`
- `RESOURCES/repos/deepseek-v3/inference/configs`

### Full MoE framework

- `RESOURCES/repos/megatron-lm/megatron/core/transformer/moe/README.md`
- `RESOURCES/repos/megatron-lm/megatron/core/transformer/moe/router.py`
- `RESOURCES/repos/megatron-lm/megatron/core/transformer/moe/token_dispatcher.py`
- `RESOURCES/repos/megatron-lm/megatron/core/transformer/moe/token_dispatcher_inference.py`
- `RESOURCES/repos/megatron-lm/megatron/core/transformer/moe/experts.py`
- `RESOURCES/repos/deepspeed/deepspeed/moe`
- `RESOURCES/repos/tutel/tutel`

### Sparse/grouped expert compute

- `RESOURCES/repos/megablocks/megablocks/layers/dmoe.py`
- `RESOURCES/repos/megablocks/megablocks/layers/router.py`
- `RESOURCES/repos/megablocks/megablocks/ops`
- `RESOURCES/repos/deepgemm`
- `RESOURCES/repos/cutlass`

### Expert communication

- `RESOURCES/repos/deepep`
- `RESOURCES/repos/flux`
- `RESOURCES/repos/nccl-tests`

### Production engine paths

- `RESOURCES/repos/vllm/vllm/model_executor/layers/fused_moe`
- `RESOURCES/repos/vllm/vllm/distributed`
- `RESOURCES/repos/sglang/python/sglang/srt/layers/moe`
- `RESOURCES/repos/sglang/python/sglang/srt/layers/moe/token_dispatcher`
- `RESOURCES/repos/sglang/python/sglang/srt/layers/moe/ep_moe`
- `RESOURCES/repos/tensorrt-llm/tensorrt_llm`

---

## 13. Validation Ladder

### M0 — Router only

Input synthetic hidden states; measure:

- routing distribution;
- top-k kernel;
- histogram/prefix sum;
- metadata bytes;
- batch-size scaling.

### M1 — Single-GPU experts

Compare:

- per-expert GEMM;
- padded batched GEMM;
- grouped GEMM;
- sparse block kernel;
- fused MoE.

Sweep:

```text
tokens, experts, top-k, hidden, expert width, skew, dtype
```

### M2 — Intra-node EP

Measure:

- all-to-all latency and bandwidth;
- local/remote fraction;
- NVLink/NVSwitch topology;
- communication SM usage;
- overlap;
- max-rank tail.

### M3 — Multi-node EP

Sweep:

- nodes;
- EP degree;
- message size;
- routing locality;
- rail/NIC mapping;
- low-latency vs high-throughput path.

### M4 — End-to-end offline inference

Separate prefill and decode. Report kernel timeline, expert batch distributions and tokens/s.

### M5 — Online serving

Use Poisson and bursty traces. Report:

- TTFT;
- P50/P95/P99 ITL;
- goodput under SLO;
- per-rank load;
- network utilization;
- expert batch sizes;
- cost/token.

### M6 — Workload shift

Change domain/language/reasoning length over time. Evaluate whether expert placement, replication or learned scheduling
remains robust.

---

## 14. Research opportunities

### Direction A — Decode-first MoE kernel

Problem: current grouped kernels excel at large token batches; online decode produces many tiny expert groups.

Potential work:

- persistent kernel;
- expert work stealing;
- shape-specialized code generation;
- fuse routing and grouped GEMM;
- hybrid padding/grouping;
- latency-aware tile scheduling.

### Direction B — Topology-aware routing without quality collapse

Jointly optimize:

```text
router score
network locality
expert load
quality constraint
```

Need training/inference compatibility and domain-shift evaluation.

### Direction C — Expert-aware continuous batching

Schedule requests to enlarge expert batches while preserving TTFT/ITL SLO and KV locality.

### Direction D — Dynamic expert replication

Treat experts as hot records:

- online popularity estimation;
- replica placement;
- migration;
- cache admission/eviction;
- consistency/versioning;
- topology-aware routing.

This direction maps naturally to database caching and skew handling.

### Direction E — Prefill/decode asymmetric EP

Prefill wants throughput; decode wants low latency. Explore different EP layouts, kernels, quantization and worker pools
for the two phases.

### Direction F — Portable EP communication

Deep vendor-specific paths can be fast but brittle. Research abstractions that preserve low latency across GPU/NIC
generations and NVIDIA/AMD/other platforms.

### Direction G — MoE observability and prediction

Build trace schemas and models for:

- expert load;
- layer critical path;
- routing drift;
- tail prediction;
- admission/autoscaling.

### Direction H — Joint KV + expert locality

Cluster routing has two stateful locality signals:

1. prefix/KV locality;
2. expert weight/load locality.

Optimizing one can harm the other. This is a strong systems scheduling problem.

---

## 15. Common invalid conclusions

| Invalid conclusion | Why |
|---|---|
| "Only 20B active, so it costs like a dense 20B" | total expert weights, dispatch and communication remain |
| "Routing is balanced on average" | P99 per-step/per-rank imbalance determines barriers |
| "Grouped GEMM is faster" | depends on expert token-size distribution and dtype |
| "All-to-all bandwidth is saturated" | decode may be latency-bound with tiny messages |
| "Quantization saves 4× memory, so latency drops 4×" | dequant/kernel support/communication can dominate |
| "Expert offload fits the model" | miss latency may destroy ITL |
| "More EP always improves throughput" | smaller local GEMMs and larger communication domain |
| "Prefill result represents serving" | decode has different shapes and critical path |
| "Uniform synthetic routing is fair" | it removes the actual skew problem |
| "Mean tokens/s improved" | tail ITL and goodput may regress |

---

## 16. Reading order and exit gate

### Papers

1. GShard;
2. Switch Transformer;
3. ST-MoE / Expert Choice;
4. DeepSeekMoE;
5. Mixtral / PEER to compare large experts with extremely fine-grained experts;
6. auxiliary-loss-free balancing;
7. DeepSpeed-MoE;
8. Tutel;
9. MegaBlocks;
10. Pre-gated MoE / Fiddler / MoEShard / UCCL-EP as systems directions.

PDFs are in [`../MOE/`](../MOE/README.md).

### Exit gate

You should be able to:

1. derive tokens-per-expert and dispatch bytes;
2. explain capacity, dropless execution and load-balancing strategies;
3. distinguish top-1/top-2/top-k, expert choice, shared experts and fine-grained experts;
4. draw EP dispatch → compute → combine and mark barriers;
5. explain why prefill and decode need different MoE paths;
6. map router, dispatcher, grouped GEMM and communication to local code;
7. design an evaluation that includes skew, topology, tail latency and quality.

---

**Previous:** [`06-kv-scheduling-serving.md`](06-kv-scheduling-serving.md) ·
**Next:** [`08-production-reliability.md`](08-production-reliability.md) ·
**Research map:** [`09-bottleneck-research.md`](09-bottleneck-research.md)
