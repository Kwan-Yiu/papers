# Modern LLM Architecture for Systems Researchers

> Purpose: read an architecture as a cost model, not as a list of model names.
>
> Snapshot: 2026-07-31
>
> Companion: [`09-distributed-inference-moe.md`](09-distributed-inference-moe.md), [`08-kv-scheduling-serving.md`](08-kv-scheduling-serving.md)

[Roadmap index](README.md) ·
[Overview](00-roadmap.md) ·
[Transformer foundations](02-transformer-foundations.md) ·
[Training systems](04-training-post-training-systems.md) ·
[Single-node optimization](05-single-node-inference-optimization.md) ·
[Competency gates](COMPETENCY-GATES.md)

---

For an inference-systems researcher, the important question is not whether Llama, Qwen, or
DeepSeek is stronger. Ask instead:

1. How many weight bytes and FLOPs does each token require?
2. How much state does each historical token leave behind?
3. Which operators form large GEMMs, and which degrade into small fragmented kernels?
4. Which state must remain in HBM, and which can be paged, compressed, migrated, or recomputed?
5. Which synchronization and communication costs appear on one GPU, one node, and multiple nodes?
6. How does an architecture delta change prefill, decode, batching, and SLO behavior?

---

## Document Guide

| Sections | Focus |
|---|---|
| 1–2 | canonical decoder and architecture dimensions |
| 3–4 | model-family deltas and architecture-to-bottleneck mapping |
| 5–6 | model inspection checklist and code-reading paths |
| 7–8 | controlled evaluations and exit criterion |

---

## 1. Canonical decoder block

A common pre-norm decoder block can be abstracted as:

```text
x1 = x + Attention(Norm(x))
x2 = x1 + FFN(Norm(x1))
```

For batch `B`, current query length `Tq`, existing context `Tk`, and hidden size `d_model`:

```text
x        : [B, Tq, d_model]
Q        : [B, n_q_heads,  Tq, head_dim]
K, V     : [B, n_kv_heads, Tk, head_dim]
attn     : softmax(QK^T / sqrt(head_dim) + mask)
output   : [B, Tq, d_model]
```

Start systems analysis with a **shape ledger**:

| Object | Shape / scale | Lifetime | Main cost |
|---|---|---|---|
| Weights | `O(n_layers × d_model²)` | model lifetime | capacity + HBM reads |
| Activations | `O(B × Tq × d_model)` | one layer / step | workspace + bandwidth |
| Attention scores | `O(B × heads × Tq × Tk)` | attention kernel | compute + temporary I/O |
| KV cache | `O(B × layers × n_kv_heads × Tk × head_dim)` | request lifetime | HBM capacity + bandwidth |
| Logits | `O(B × Tq × vocab)` | output step | large vocab GEMM |
| MoE routing metadata | `O(tokens × top_k)` | one MoE layer | sort/permute/dispatch |

---

## 2. Architecture dimensions

Do not organize models by brand. Compare them across the following architecture dimensions.

### 2.1 Sequence topology

| Topology | Typical use | Inference state | Systems consequence |
|---|---|---|---|
| Encoder-only | embedding/reranking/classification | usually no autoregressive KV | large parallel batch, compute-oriented |
| Encoder-decoder | translation, some multimodal models | encoder states + decoder KV | cross-attention state and two execution phases |
| Decoder-only | mainstream text LLM | growing self-attention KV | stateful, iterative decode |
| Prefix-LM | mixed bidirectional/causal segments | mask-dependent | cache reuse must preserve mask semantics |
| Recurrent / SSM | Mamba, RWKV-like | fixed-size recurrent state | constant state per layer, sequential state update |
| Hybrid | attention + recurrence/SSM | KV for some layers + recurrent state | heterogeneous cache and kernel scheduling |

Master decoder-only execution first, then compare hybrid architectures because they directly
challenge the serving abstraction that all persistent state consists of KV blocks.

### 2.2 Tokenization and vocabulary

Categories:

- byte-level BPE;
- SentencePiece/BPE;
- unigram;
- byte fallback;
- multilingual vocabulary;
- multimodal codebook/tokenizer;
- separate or unified input/output vocabularies.

Systems consequences:

- The same character length can produce different token counts, directly changing prefill cost,
  KV footprint, and billing.
- A large vocabulary increases embedding/output-projection parameters and the logits GEMM.
- Tokenization can become a CPU bottleneck at high QPS or with small models.
- A prompt-cache key must use the normalized token sequence rather than the raw string.
- Differences in chat templates, special tokens, and tool tokens can prevent cache sharing between
  apparently identical prompts.

### 2.3 Embedding and output head

Inspect:

- whether the input embedding and LM head share weights;
- how vocabulary parallelism partitions the head;
- whether logits are generated for every position or only the final token / selected positions;
- whether sampling runs on the GPU or CPU;
- whether structured decoding applies an additional logits mask;
- whether speculative verification emits logits for multiple positions at once.

A very large vocabulary head can be a material bandwidth and latency cost during small-batch decode.

### 2.4 Normalization and residual layout

| Variant | Where norm happens | What to learn |
|---|---|---|
| Post-LN | norm after the sublayer | original Transformer; harder deep-model training |
| Pre-LN | norm before the sublayer | common in modern decoders; more direct residual path |
| RMSNorm | only RMS scaling, no mean centering | less reduction work; common in modern LLMs |
| Parallel residual | attention/MLP from the same normalized input | changes the dependency graph and fusion opportunities |
| Sandwich / extra norm | norm before and after the sublayer | stability vs additional-kernel trade-off |

The systems question is not "RMSNorm omits a mean, so it must be fast." Inspect:

- fusion with residual, quantization, and activation paths;
- whether hidden size and batch/token count amortize launch overhead;
- whether the kernel needs a cross-CTA reduction;
- normalization precision and accumulator type;
- additional synchronization under tensor parallelism.

### 2.5 Positional information

| Family | Mechanism | State/cost implication |
|---|---|---|
| Learned absolute | learned position embedding | fixed trained range; small lookup cost |
| Sinusoidal | add fixed features | no learned table |
| RoPE | rotate Q/K by position | affects Q/K kernel and KV compatibility |
| ALiBi | add head-specific linear bias | no position embedding table |
| Relative bias | bias by relative distance/bucket | extra score/bias handling |
| RoPE scaling | NTK-style, YaRN, LongRoPE | extends context but does not remove KV/attention cost |

Distinguish:

1. **the representation supports larger positions**;
2. **the model retains quality at longer positions**;
3. **the system can hold the larger KV state**;
4. **attention complexity is reduced**.

RoPE scaling usually addresses only the first two points. It does not automatically solve KV
capacity or attention complexity.

### 2.6 Attention head sharing: MHA → GQA → MQA → MLA

Definitions:

- `Hq`: query heads;
- `Hkv`: KV heads;
- `Dh`: head dimension;
- `L`: layers;
- `S`: cached sequence length;
- `b`: bytes per KV element.

Approximate KV size per request:

```text
KV_bytes = 2 × L × Hkv × Dh × S × b
```

| Variant | `Hkv` | Main idea | Decode consequence |
|---|---:|---|---|
| MHA | `Hq` | every Q head owns K/V head | largest KV, strongest independence |
| GQA | `1 < Hkv < Hq` | groups of Q heads share K/V | compromise between quality and KV/bandwidth |
| MQA | `1` | all Q heads share one K/V | minimal conventional KV |
| MLA | latent dimensions, not plain `Hkv` | cache compressed latent representation; reconstruct/project as needed | much smaller state, different kernels and parallel decomposition |

Measure independently:

- KV-capacity reduction;
- decode-attention read bytes;
- K/V-projection cost;
- latent reconstruction/dequantization cost;
- tensor-parallel communication;
- kernel support and shape efficiency.

An eightfold reduction in KV does not guarantee an eightfold end-to-end speedup. Weight GEMMs,
scheduling, sampling, and communication remain.

External tutorial bridge:

- [LLMs-from-scratch — GQA](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/04_gqa);
- [LLMs-from-scratch — MLA](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/05_mla).

### 2.7 Cross-layer KV and attention sharing

Do not treat all cross-layer methods as the same mechanism. Use the external
[Cross-Layer KV Sharing tutorial](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/10_kv-sharing),
then compare:

| Source | Shared object | Systems question |
|---|---|---|
| [CLA](../ARCHITECTURE/CLA.pdf) | K/V heads across adjacent layers | KV capacity and decode reads |
| [MLKV](../ARCHITECTURE/MLKV.pdf) | KV heads across head and layer groups | memory-quality frontier |
| [YOCO](../ARCHITECTURE/YOCO.pdf) | global KV built once and reused by a cross-decoder | cache-once architecture and prefill |
| [LiSA](../ARCHITECTURE/LISA.pdf) | aligned/approximated attention maps | attention compute and approximation |

Reference paths:

```text
RESOURCES/repos/pythia-mlkv/
RESOURCES/repos/flash-linear-attention/fla/models/yoco/
```

Record whether the model was trained with sharing, converted, or uptrained, and separate KV sharing
from attention-map sharing.

### 2.8 Attention span and sparsity

| Pattern | Query attends to | Complexity intuition | Serving concern |
|---|---|---|---|
| Full causal | all previous tokens | prefill quadratic, decode linear in context | highest quality baseline |
| Sliding window | recent `W` tokens | bounded local attention | evictable old local KV, but global layers/tokens complicate |
| Dilated/block sparse | selected blocks | pattern-dependent | layout and kernel regularity decide real speed |
| Global + local | local window + special/global positions | reduced working set | metadata and two-path kernels |
| Retrieval attention | retrieved subset of KV | depends on index/search | index build, recall, random access |
| Native sparse attention | learned/architected selected regions | lower attended tokens | training-inference co-design; exactness differs |
| Linear attention | feature-map/recurrent formulation | linear or recurrent | numerical stability and model quality boundaries |

Record the semantics of every sparse method:

- exact or approximate;
- static, query-dependent, or learned selection;
- CPU or GPU selection metadata;
- whether gathered KV is contiguous;
- whether prefill and decode use the same pattern;
- whether quality is measured by perplexity, task accuracy, or long-context retrieval.

### 2.9 Feed-forward network

A common dense FFN is:

```text
FFN(x) = W_down activation(W_up x)
```

Gated FFN:

```text
SwiGLU(x) = W_down (SiLU(W_gate x) ⊙ (W_up x))
```

Comparison dimensions:

- activation: ReLU / GELU / SiLU;
- gated or ungated;
- intermediate width;
- bias or bias-free;
- dense or conditional experts;
- fused gate/up projection;
- weight/activation precision;
- TP split dimension;
- quantized-kernel availability.

FFN/GEMM often accounts for most weight reads during decode. Once attention is optimized, the FFN
can become the new dominant bottleneck.

### 2.10 Dense vs sparse MoE FFN

MoE replaces one FFN with multiple experts and a router:

```text
scores = Router(x)
chosen = TopK(scores)
y = Σ gate_weight(e, x) × Expert_e(x)
```

Record these architecture parameters separately:

```text
total_parameters
activated_parameters_per_token
number_of_experts
top_k
shared_experts
expert_intermediate_size
moe_layer_frequency
capacity_factor / drop policy
```

The complete MoE systems path is in
[`09-distributed-inference-moe.md`](09-distributed-inference-moe.md). The central warning is:

> Sparse FLOPs do not imply sparse memory, communication, or latency.

Even when each token activates only a few experts, all expert weights must still be placed, loaded,
or accessed remotely. Dynamic routing also creates irregular small GEMMs, all-to-all traffic, and
tail imbalance.

### 2.11 Recurrence, SSM, retention and hybrid blocks

| Family | Persistent state | Parallel prefill | Incremental decode |
|---|---|---|---|
| Transformer attention | KV grows with context | highly parallel but attention is context-dependent | read growing KV |
| Linear attention | fixed/structured summary state | scan or associative formulation | constant-size update |
| SSM / Mamba | per-layer state | selective scan | recurrent state update |
| RWKV-like | recurrent accumulators | parallel training formulation | RNN-like constant state |
| RetNet | retention state | parallel/recurrent/chunkwise modes | recurrent |
| Hybrid local attention + recurrence | KV for local layers/windows + recurrent state | mixed | mixed kernels and state |

New systems questions:

- How should the cache manager handle state types with different lifetimes?
- How should recurrent state be compacted and migrated under continuous batching?
- Does request preemption need to save only a fixed-size state?
- Can prefix sharing reuse a recurrent-state snapshot?
- How should speculative rollback restore state?
- Should heterogeneous layers be placed or scheduled separately?

### 2.12 Multi-token heads and speculative-friendly architecture

Categories:

- independent multi-token prediction heads;
- Medusa-style multiple decoding heads;
- EAGLE-style feature-level draft;
- built-in MTP module;
- tree attention verification;
- n-gram / suffix / prompt-lookup proposals.

Key quantities:

```text
acceptance_length
draft_cost
verification_cost
extra_weight_bytes
extra_KV_or_state
rollback_cost
batch_interference
```

MTP can be a training objective or an inference draft source. Do not conflate the two roles.

### 2.13 Dynamic depth and conditional layer execution

Mixture-of-Depths, early exit, and layer skipping extend conditional computation from choosing an
FFN expert to deciding whether a token executes a block. Record separately:

```text
layer_capacity
tokens_selected_per_layer
selection_stability
per_token_depth
skipped_state_semantics
compaction_and_scatter_cost
quality_vs_compute_budget
```

This creates ragged token sets: every layer may have a different number of active tokens, requiring
gather/compact, execution, and scatter while disrupting fixed batch shapes. The potential benefit is
lower block FLOPs; the costs are routing, data movement, small GEMMs, and warp under-utilization.
Report measured kernel time rather than only theoretical FLOPs.

### 2.14 Diffusion and masked iterative generation

An autoregressive LM appends one or a few tokens per step. A masked diffusion LM begins with a
masked sequence and updates multiple positions through repeated denoising/refinement. The primary
systems dimensions are:

- the number of refinement steps and active masks per step;
- full-sequence forward passes vs updates only at changed positions;
- safe KV/state reuse under bidirectional attention;
- remasking, confidence selection, and stopping policies;
- continuous batching of requests at different refinement progress;
- conversion of a latency budget into a dynamic step budget;
- whether parallel token updates offset repeated full-model execution.

This is not merely a sampler change: generation dependencies, cache semantics, batch shapes, and SLO
control all change. Read MDLM and LLaDA first, then compare against an autoregressive baseline at
matched model quality, output length, and end-to-end latency budget.

### 2.15 Precision as part of architecture execution

Distinguish:

- storage dtype;
- compute/input dtype;
- accumulator dtype;
- communication dtype;
- KV dtype;
- router/gate dtype;
- logits/sampling dtype.

Common combinations include BF16/FP16, FP8, INT8, INT4, NVFP4/MXFP4, and mixed-precision paths.
Whether quantization helps depends on:

- backend support for the model architecture;
- whether GEMM shapes hit specialized Tensor Core kernels;
- scale/zero-point metadata;
- dequant fusion;
- calibration and outlier strategy;
- grouped-GEMM support for small MoE expert matrices;
- quality constraint.

---

## 3. Model-family delta map

This table records only systems-relevant deltas. It does not rank model capability.

| Family | Architecture delta to inspect | Primary systems question |
|---|---|---|
| GPT-style dense decoder | full causal attention, dense FFN, LayerNorm variants | establish canonical baseline |
| Llama 2/3 | RoPE, RMSNorm, SwiGLU, GQA in larger variants | standard modern dense serving path |
| Mistral | GQA + sliding-window attention | local KV policy and window kernels |
| Mixtral | Mistral-style attention + sparse top-2 MoE FFN | expert weight capacity, dispatch and small GEMM |
| Qwen2/Qwen3 | dense and MoE variants; broad model/config range | one family for dense-vs-MoE controlled study |
| DeepSeek-V2/V3 | MLA + fine-grained/shared-expert MoE; V3 adds aux-loss-free balance and MTP | compressed state + wide EP + MTP co-design |
| Kimi Linear | KDA linear attention + periodic MLA + sparse MoE | fixed recurrent state, hybrid layer schedule and long-context kernels |
| GPT-OSS | open-weight MoE model and optimized reference paths | modern MoE backend portability |
| Mamba/Mamba-2 | selective SSM / state-space duality | fixed state, scan kernels, hybrid serving |
| RWKV | Transformer-trainable recurrent execution | constant state and recurrent batching |
| Griffin / recurrent hybrid | local attention + gated recurrence | heterogeneous layer state manager |
| Mixture-of-Depths / early-exit | token-dependent layer execution | ragged batches, compaction and quality-aware compute budgets |
| MDLM / LLaDA | masked iterative denoising instead of left-to-right decode | refinement scheduling, state reuse and latency/quality control |

### Recommended controlled comparisons

1. Llama-like MHA/GQA variants with equal hidden/layer scale;
2. Mistral dense vs Mixtral MoE;
3. Qwen dense vs Qwen MoE at similar active parameter scale;
4. standard GQA vs DeepSeek MLA;
5. conventional per-layer KV vs CLA/MLKV/YOCO-style layer sharing;
6. attention-only vs Mamba/DeltaNet/Kimi Linear hybrid with equal workload;
7. MTP/speculation on and off under the same arrival trace;
8. fixed depth vs token-conditional depth with matched quality;
9. autoregressive vs diffusion generation with matched quality, output length and SLO.

Do not compare models with unrelated quality and scale and attribute the entire result to one
architecture feature.

---

## 4. Architecture → bottleneck matrix

| Feature | Weight capacity | KV/state capacity | HBM bytes/token | Compute | Communication | Scheduling irregularity |
|---|---:|---:|---:|---:|---:|---:|
| MHA | baseline | high | high for long context | baseline | TP-dependent | low |
| GQA/MQA | same-ish | lower | lower | similar | may change TP layout | low |
| MLA | projection-dependent | much lower latent state | lower state read, extra transforms | changed | implementation-dependent | medium |
| Cross-layer KV sharing | similar | lower by sharing group | fewer layer-specific KV reads | changed projections/reuse | implementation-dependent | medium |
| Sliding window | same | bounded for local layers | bounded local reads | lower at long context | low | window metadata |
| Sparse/retrieval attention | same | stored KV may remain large | selected reads | selection + attention | possibly remote | high |
| Dense FFN | high | none persistent | high weight reads | regular GEMM | TP collectives | low |
| MoE FFN | very high total | none persistent | active experts + routing | irregular/grouped GEMM | EP all-to-all | high |
| Quantized weights | lower | unchanged | lower if fused | dequant/low-bit | lower weight transfer | backend-specific |
| SSM/recurrent | similar model-dependent | fixed state | fixed state access | scan/update | layer/TP-dependent | mixed layers |
| MTP heads | extra weights | possibly extra state | extra reads | proposal/verification | batch-dependent | acceptance variance |
| Dynamic depth | unchanged total | activation metadata | selected layers only | conditional block compute | placement-dependent | ragged per-layer tokens |
| Diffusion LM | model-dependent | sequence/refinement state | repeated sequence reads | parallel updates × steps | model-parallel dependent | variable refinement progress |

---

## 5. Fixed Checklist for Inspecting an Unfamiliar Model

### Step 1 — Read config, not marketing

Extract:

```text
vocab_size
hidden_size
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
intermediate_size
max_position_embeddings
rope_theta / rope_scaling
sliding_window
num_experts
num_experts_per_tok
shared_expert_intermediate_size
moe_layer_freq / decoder_sparse_step
tie_word_embeddings
torch_dtype / quantization_config
```

### Step 2 — Reconstruct one block

Write down:

- norm order;
- Q/K/V projection shapes;
- attention pattern;
- output projection;
- FFN or MoE;
- residual structure;
- any recurrent state;
- any MTP/speculative heads.

### Step 3 — Compute memory

At minimum:

```text
weight_bytes
KV_bytes_per_token
KV_bytes_per_request_at_{4K,32K,128K}
activation_workspace_estimate
expert_weight_bytes_per_rank
```

### Step 4 — Predict prefill/decode behavior

Before profiling, state falsifiable predictions:

- prefill compute- or bandwidth-bound;
- decode weight- or KV-bandwidth-bound;
- expected dominant kernels;
- expected TP/EP collectives;
- context or batch point at which the bottleneck changes.

### Step 5 — Find backend-specific implementation

Model definition tells semantics; serving engine tells the actual execution. Compare both.

---

## 6. Local code-reading paths

### Semantics first

- `RESOURCES/repos/transformers/src/transformers/models/llama`
- `RESOURCES/repos/transformers/src/transformers/models/mistral`
- `RESOURCES/repos/transformers/src/transformers/models/mixtral`
- `RESOURCES/repos/transformers/src/transformers/models/qwen3`
- `RESOURCES/repos/transformers/src/transformers/models/qwen3_moe`
- `RESOURCES/repos/transformers/src/transformers/models/deepseek_v3`
- `RESOURCES/repos/transformers/src/transformers/models/gpt_oss`
- `RESOURCES/repos/transformers/src/transformers/models/mamba`
- `RESOURCES/repos/transformers/src/transformers/models/mamba2`
- `RESOURCES/repos/transformers/src/transformers/models/rwkv`

### Minimal reference implementations

- `RESOURCES/repos/deepseek-v3/inference/model.py`
- `RESOURCES/repos/deepseek-v3/inference/kernel.py`
- `RESOURCES/repos/deepseek-v3/inference/configs`
- `RESOURCES/repos/mamba/mamba_ssm`
- `RESOURCES/repos/flash-linear-attention/fla`
- `RESOURCES/repos/kimi-linear`
- `RESOURCES/repos/pythia-mlkv`
- `/home/junyao/code/study/tutorials/llm/LLMs-from-scratch`
- `/home/junyao/code/study/tutorials/transformer/annotated-transformer`

### Production execution

- `RESOURCES/repos/vllm/vllm/model_executor/models`
- `RESOURCES/repos/vllm/vllm/model_executor/layers/attention`
- `RESOURCES/repos/vllm/vllm/model_executor/layers/fused_moe`
- `RESOURCES/repos/vllm/vllm/model_executor/layers/mamba`
- `RESOURCES/repos/vllm/vllm/v1/attention`
- `RESOURCES/repos/sglang/python/sglang/srt/models`
- `RESOURCES/repos/sglang/python/sglang/srt/layers/attention`
- `RESOURCES/repos/sglang/python/sglang/srt/layers/moe`

---

## 7. Controlled Architecture Evaluations

### A1 — KV head sweep

Hold layers/hidden/context constant and sweep `Hkv`.

Measure:

- KV bytes;
- max concurrency;
- attention bandwidth;
- decode ITL;
- end-to-end goodput;
- quality if weights are actually retrained/uptrained.

### A2 — Full vs sliding-window attention

Sweep context length and window.

Measure:

- prefill time;
- per-token decode time;
- retained KV;
- global/local layer effects;
- accuracy on local and long-range tasks.

### A3 — Dense vs MoE at equal active compute

Report total and active parameters separately. Sweep batch and EP degree.

Measure:

- weight capacity;
- tokens per expert distribution;
- grouped GEMM sizes;
- all-to-all bytes;
- P50/P99 ITL;
- cost/token.

### A4 — GQA vs MLA

Use backend-supported implementations.

Measure:

- cached representation bytes;
- projection/reconstruction kernels;
- attention backend;
- TP communication;
- prefill/decode crossover.

### A5 — Per-layer KV vs cross-layer sharing

- conventional per-layer KV versus CLA/MLKV-style groups;
- training/uptraining requirement;
- KV bytes and decode traffic;
- quality and long-context behavior;
- backend support.

### A6 — Attention vs SSM/hybrid state

Use the same prompt/output distributions.

Measure:

- state bytes by context;
- prefill throughput;
- decode latency;
- preemption/save/restore cost;
- prefix reuse feasibility;
- quality at short and long range.

### A7 — MTP/speculative head

Sweep workload, temperature and output length.

Measure:

- acceptance-length distribution;
- verification batch size;
- extra model memory;
- latency variance;
- throughput under load;
- benefit after scheduling interference.

---

## 8. Exit gate

After completing this chapter, you should be able to analyze a new model configuration:

1. draw its blocks and persistent state;
2. calculate weight, KV/state, and active-parameter costs;
3. identify required attention, MoE, and recurrent kernels;
4. predict the main prefill and decode bottlenecks;
5. find the corresponding Hugging Face semantics, reference implementation, and production-engine
   paths;
6. design a controlled comparison that does not conflate model size, quality, and architecture
   feature.

## Primary papers

Local PDFs are in [`../ARCHITECTURE/`](../ARCHITECTURE/README.md). Key papers include:

- MQA, GQA, CLA, MLKV, YOCO, RoPE, RMSNorm, GLU variants, ALiBi;
- Mistral, Llama 3, Qwen3, DeepSeek-V2/V3;
- Linear Transformer, Based, GLA, DeltaNet, Kimi Linear, Mamba/Mamba-2, RetNet, RWKV, Griffin;
- YaRN, LongRoPE, Multi-token Prediction.

---

**Previous:** [`02-transformer-foundations.md`](02-transformer-foundations.md) ·
**Next:** [`04-training-post-training-systems.md`](04-training-post-training-systems.md) ·
**Inference optimization:** [`05-single-node-inference-optimization.md`](05-single-node-inference-optimization.md) ·
**MoE deep dive:** [`09-distributed-inference-moe.md`](09-distributed-inference-moe.md)
