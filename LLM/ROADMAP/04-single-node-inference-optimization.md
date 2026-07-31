# Single-Node LLM Inference Optimization — Curated Reading Map

> **Purpose:** connect architecture-level compression, attention algorithms, low precision, GPU
> kernels, compiler behavior, and inference-engine execution.
>
> **Method:** learn each mechanism from an external English tutorial or official document, verify
> the primary paper, inspect a reference implementation, and then locate the production backend.
>
> **Boundary:** this document is a navigator, not a self-contained optimization tutorial.

[Roadmap](00-roadmap.md) ·
[Modern architecture](03-modern-llm-architecture.md) ·
[Single-node engine](05-single-node-inference-engine.md) ·
[Repository atlas](../RESOURCES/GITHUB-REPO-ATLAS.md) ·
[Source provenance](../RESOURCES/SOURCES.md)

---

## How to Use This Map

For every optimization, keep four questions separate:

1. **Semantics:** is it exact, model-defined, retrained/uptrained, or approximate?
2. **Resource:** does it reduce weight bytes, KV/state bytes, HBM traffic, FLOPs, launches, or
   allocation waste?
3. **Implementation:** does the target hardware and engine have a fused kernel for the resulting
   dtype, layout, and shape?
4. **End-to-end effect:** does the change improve TTFT, TPOT/ITL, throughput, capacity, or cost for
   the declared workload?

Use this evidence chain:

```text
external explanation
→ primary paper
→ minimal implementation
→ production implementation
→ predicted resource change
→ measured profiler/serving evidence
→ quality and regression boundary
```

Do not infer end-to-end speedup from a compression ratio, FLOP count, or standalone kernel result.

---

## 1. Field Map

The examples in this area belong to different optimization layers. Read them as a system, not as a
flat list of technique names.

| Layer | Representative mechanisms | Primary resource affected | Semantic class |
|---|---|---|---|
| cost model and measurement | tensor ledger, Roofline, memory snapshot, profiler | diagnosis only | exact observation/model |
| numerical representation | weight, activation, KV, attention quantization | capacity, bandwidth, tensor-core path | approximate or trained |
| head/state architecture | MQA, GQA, MLA | KV capacity and decode traffic | model-defined |
| cross-layer state sharing | CLA, MLKV, YOCO | KV capacity across layers | model-defined/uptrained |
| retention and span | sliding window, sinks, token eviction, retrieval heads | retained KV and attended tokens | model-defined or approximate |
| sparse attention | block, dynamic, native, retrieval sparsity | attention FLOPs and KV reads | model-defined or approximate |
| linear/recurrent attention | Linear Transformer, GLA, DeltaNet, KDA | fixed recurrent state, scan/recurrent kernels | model-defined |
| exact attention kernel | FlashAttention, FlashInfer | HBM traffic and kernel efficiency | exact |
| runtime memory | paging, prefix reuse, graph pools, workspaces | fragmentation, allocation, reuse | exact |
| compiler/kernel | fusion, specialization, Triton, CUTLASS/CuTe | launches, memory traffic, tensor-core use | exact |

### Required distinctions

- FlashAttention is not sparse attention and does not by itself shrink model KV semantics.
- PagedAttention changes physical KV allocation, not which tokens the model attends to.
- GQA/MQA share KV across query heads; CLA/MLKV share KV across layers.
- MLA stores a latent representation rather than a conventional reduced number of KV heads.
- Sliding-window attention is model-defined when trained that way; post-hoc token eviction is
  approximate.
- Linear attention changes the attention formulation and persistent state; it is not a drop-in
  kernel replacement for an arbitrary softmax-attention checkpoint.
- Weight quantization, activation quantization, and KV quantization have different distributions,
  calibration choices, kernels, and quality risks.

---

## 2. External Tutorial Spine

### 2.1 Architecture and memory tutorials

Use the current bonus material in Sebastian Raschka's
[LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) before reading the research papers.
The repository is also available in the shared study workspace.

| Order | External tutorial/code | Read or run | Why it belongs here |
|---:|---|---|---|
| 1 | [Performance Analysis](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/02_performance-analysis) | `README.md`, `flops-analysis.ipynb` | parameter/FLOP analysis bridge |
| 2 | [KV Cache](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/03_kv-cache) | `README.md`, baseline and optimized cache files | cache semantics, preallocation, generation |
| 3 | [Grouped-Query Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/04_gqa) | tutorial, implementation, memory estimator | MHA→GQA memory effect |
| 4 | [Multi-Head Latent Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/05_mla) | tutorial, implementation, memory estimator | GQA versus latent state |
| 5 | [Sliding Window Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/06_swa) | tutorial, implementation, tests, estimator | bounded attention span and KV |
| 6 | [Gated DeltaNet](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet) | full tutorial and memory comparison | linear/recurrent attention entry |
| 7 | [DeepSeek Sparse Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa) | tutorial, implementation, tests | current sparse-attention example |
| 8 | [Cross-Layer KV Sharing](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/10_kv-sharing) | tutorial, implementation, tests, estimator | head sharing versus layer sharing |

Local reused source:

```text
/home/junyao/code/study/tutorials/llm/LLMs-from-scratch/ch04/
```

Do not copy these tutorials into this roadmap. Use them to establish mechanics, then move to the
papers and production paths below.

### 2.2 System-level orientation

| Priority | External resource | Exact target | Extract |
|---|---|---|---|
| Core | [How To Scale Your Model — Transformer Math](https://jax-ml.github.io/scaling-book/transformers/) | forward-pass parameter/FLOP accounting | tensor and operator ledger |
| Core | [How To Scale Your Model — Rooflines](https://jax-ml.github.io/scaling-book/roofline/) | full chapter | compute, bandwidth, capacity bounds |
| Core | [How To Scale Your Model — Inference](https://jax-ml.github.io/scaling-book/inference/) | prefill, generation, KV and quantization sections | batch/context-dependent bottlenecks |
| Core | [Efficiently Scaling Transformer Inference](../PERF/TRANSFORMERINFER.pdf) | analytical model and evaluation | phase-specific cost model |
| Core | [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | overhead, fusion, memory and compilation | framework-to-kernel reasoning |
| Branch | [Lilian Weng — Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | memory, parallelism and decoding sections | broad inference orientation |

---

## 3. Cost Model and Measurement

### 3.1 Predict before profiling

Use the sources above to produce a cost record. The roadmap does not substitute its own memory or
latency formula.

Required fields:

```text
model/config revision
batch and sequence distribution
prefill or decode
tensor shapes and layouts
weight / activation / KV / workspace dtypes
parameter bytes
KV/state bytes per token and per request
workspace and allocator-reserved bytes
operator FLOPs and estimated HBM bytes
arithmetic intensity
launches and synchronization points
predicted limiting resource
```

### 3.2 Memory measurement path

| Order | Official resource | Use |
|---:|---|---|
| 1 | [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management) | allocated versus reserved memory and allocator behavior |
| 2 | [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/main/torch_cuda_memory) | allocation history and memory snapshots |
| 3 | [PyTorch Memory Visualizer](https://pytorch.org/memory_viz) | inspect snapshot segments, blocks, and allocation traces |
| 4 | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | CPU/GPU timeline, transfers, launches, memory-usage trace |
| 5 | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | per-kernel memory and compute evidence |

Record at least:

- model weights after loading;
- runtime-reserved pools;
- KV growth against generated tokens;
- temporary prefill workspace;
- CUDA Graph private pools when enabled;
- non-PyTorch allocations that a PyTorch snapshot cannot see;
- fragmentation and unused reserved capacity.

### 3.3 Timing and profiling path

| Resource | Exact target |
|---|---|
| [PyTorch Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) | warm-up, synchronization and blocked autorange |
| [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | operator CPU/CUDA time and shapes |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/) | request/runtime timeline and launch gaps |
| [Nsight Compute](https://docs.nvidia.com/nsight-compute/) | achieved occupancy, memory throughput and tensor-core behavior |
| [vLLM benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/) | serving and latency workload drivers |
| [TensorRT-LLM benchmarking](https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-benchmarking.html) | backend-specific throughput/latency procedure |

The minimum comparison is predicted bytes/time versus measured allocated/reserved memory and
measured kernel/runtime time. Explain the residual rather than silently treating the formula as the
measurement.

---

## 4. Model and Runtime Quantization

### 4.1 Concept-first external path

| Order | Resource | Exact target |
|---:|---|---|
| 1 | [Hugging Face — Quantization concepts](https://huggingface.co/docs/transformers/quantization/concept_guide) | affine/symmetric schemes, granularity, PTQ/QAT, INT and FP formats |
| 2 | [Hugging Face — Selecting a quantization method](https://huggingface.co/docs/transformers/main/quantization/selecting) | calibration, backend and hardware trade-offs |
| 3 | [TensorRT-LLM — Numerical Precision](https://nvidia.github.io/TensorRT-LLM/reference/precision.html) | Q/DQ, per-tensor/token/channel scaling and support matrix |
| 4 | [vLLM — Quantization](https://docs.vllm.ai/en/stable/features/quantization/index.html) | engine methods and hardware matrix |
| 5 | [LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/) | current deployment-oriented workflows |
| 6 | [TensorRT-LLM PyTorch backend quantization](https://nvidia.github.io/TensorRT-LLM/torch/features/quantization.html) | FP8/NVFP4 model preparation and loading |

### 4.2 Paper ladder

| Branch | Paper | Read for |
|---|---|---|
| outliers and mixed precision | [LLM.int8()](../QUANT/LLMINT8.pdf) | activation outliers and mixed precision |
| W+A PTQ | [ZeroQuant](../QUANT/ZEROQUANT.pdf) | hardware-aware W8A8 and backend co-design |
| weight-only PTQ | [GPTQ](../QUANT/GPTQ.pdf) | second-order layer-wise reconstruction |
| activation smoothing | [SmoothQuant](../QUANT/SMOOTHQUANT.pdf) | moving quantization difficulty from activations to weights |
| activation-aware weights | [AWQ](../QUANT/AWQ.pdf) | salient channels and W4A16 |
| serving co-design | [Atom](../QUANT/ATOM.pdf) | low-bit serving beyond checkpoint size |
| rotation | [QuaRot](../QUANT/QUAROT.pdf) → [SpinQuant](../QUANT/SPINQUANT.pdf) | random versus learned rotations and outliers |
| broad evaluation | [Evaluating Quantized LLMs](../QUANT/QLLM-EVAL.pdf) | weight, activation and KV quality across model families |
| KV precision | [KIVI](../CACHE/KIVI.pdf) → [KVQuant](../CACHE/KVQUANT.pdf) | KV distributions, granularity and decode kernels |

### 4.3 Coverage checklist

Do not call the quantization branch complete unless the external readings and code paths cover:

- PTQ versus QAT;
- weight-only, weight-activation, KV, attention, and MoE-expert quantization;
- storage dtype, compute dtype, accumulator dtype, scale dtype, and communication dtype;
- symmetric versus asymmetric mapping;
- static versus dynamic quantization;
- per-tensor, per-channel, per-token, per-group/block, and per-head granularity;
- INT8, INT4, FP8, FP4, NVFP4, MXFP4, and mixed-precision formats;
- calibration data, clipping, observers, outliers, rotations, and reconstruction;
- packing layout, scale/zero-point metadata, fused dequantization, and kernel dispatch;
- quality, memory, latency, throughput, batch-size, context-length, and hardware boundaries.

QLoRA belongs to the low-bit fine-tuning branch. It is not the default inference-quantization
mainline.

### 4.4 Deployment code path

#### LLM Compressor

Read the local shallow clone in this order:

```text
RESOURCES/repos/llm-compressor/README.md
RESOURCES/repos/llm-compressor/examples/quantization_w8a8_fp8/
RESOURCES/repos/llm-compressor/examples/quantization_w8a8_int8/
RESOURCES/repos/llm-compressor/examples/quantization_w4a16/
RESOURCES/repos/llm-compressor/examples/awq/
RESOURCES/repos/llm-compressor/examples/quantization_kv_cache/
RESOURCES/repos/llm-compressor/examples/quantization_attention/
RESOURCES/repos/llm-compressor/examples/quantizing_moe/
RESOURCES/repos/llm-compressor/examples/transform/
```

#### NVIDIA Model Optimizer

```text
RESOURCES/repos/model-optimizer/examples/llm_ptq/
RESOURCES/repos/model-optimizer/examples/llm_qat/
RESOURCES/repos/model-optimizer/examples/llm_eval/
```

#### vLLM

```text
RESOURCES/repos/vllm/vllm/model_executor/layers/quantization/
RESOURCES/repos/vllm/vllm/model_executor/layers/quantization/kv_cache.py
RESOURCES/repos/vllm/vllm/model_executor/layers/quantization/compressed_tensors/
RESOURCES/repos/vllm/vllm/model_executor/layers/quantization/online/
RESOURCES/repos/vllm/vllm/model_executor/layers/quantization/utils/
```

Trace:

```text
checkpoint quantization_config
→ engine config selection
→ QuantizationConfig / method
→ packed weight loader
→ linear or fused-MoE kernel
→ scale/dequant path
→ fallback or unsupported-shape path
```

### 4.5 Adjacent model-compression branches

Quantization is not the entire model-compression field. Continue with the curated
[`COMPRESSION`](../COMPRESSION/README.md) path for:

- unstructured and hardware-supported N:M sparsity;
- SparseGPT and Wanda post-training pruning;
- structured FFN, hidden-width, head, GQA-group, state-dimension, and depth pruning;
- teacher-student distillation;
- pruning/quantization composition.

Primary production-oriented source:

```text
RESOURCES/repos/model-optimizer/examples/llm_sparsity/
RESOURCES/repos/model-optimizer/examples/pruning/
RESOURCES/repos/model-optimizer/examples/llm_distill/
```

Require a sparse representation and supported sparse kernel before claiming latency improvement.
Dense kernels do not become faster merely because checkpoint values are zero.

---

## 5. KV Cache Compression, Selection, and Eviction

### 5.1 Method map

| Family | Representative sources | What changes |
|---|---|---|
| lower precision | [KIVI](../CACHE/KIVI.pdf), [KVQuant](../CACHE/KVQUANT.pdf) | representation |
| attention sinks/window | [StreamingLLM](../CACHE/STREAMINGLLM.pdf) | retained token positions |
| heavy-hitter eviction | [H2O](../CACHE/H2O.pdf) | retained token positions |
| observation-window selection | [SnapKV](../CACHE/SNAPKV.pdf) | prompt KV retained after prefill |
| layer-adaptive budgets | [PyramidKV](../CACHE/PYRAMIDKV.pdf) | per-layer retained-token budget |
| head-adaptive retention | [DuoAttention](../ATTENTION/DUOATTENTION.pdf) | full cache for retrieval heads, bounded cache for streaming heads |
| low-rank/key compression | [ShadowKV repository](https://github.com/ByteDance-Seed/ShadowKV) | stored representation and reconstruction |
| framework comparison | [NVIDIA KVPress](https://github.com/NVIDIA/kvpress) | common implementation/evaluation surface |
| physical paging | [vLLM](../SERVING/VLLM.pdf), [vAttention](../CACHE/VATTENTION.pdf) | allocation/layout, not logical attention |
| tiering/offload | [LMCache](https://github.com/LMCache/LMCache) | state location |
| disaggregated state | [Mooncake](../CACHE/MOONCAKE.pdf) | state placement and transfer |

### 5.2 External code-reading path: KVPress

The local shallow clone provides a unified research surface:

```text
RESOURCES/repos/kvpress/README.md
RESOURCES/repos/kvpress/kvpress/presses/
RESOURCES/repos/kvpress/notebooks/
RESOURCES/repos/kvpress/evaluation/
RESOURCES/repos/kvpress/tests/presses/
```

Use it to compare selection rules, per-layer/head budgets, prefill-only compression, decoding-time
compression, and composed policies. Keep its Hugging Face evaluation role separate from production
engine integration.

### 5.3 Quality and systems evidence

For every approximate KV method, record:

- selection time and whether it runs during prefill or every decode step;
- transient memory before compression;
- retained tokens per layer and head;
- contiguous/block-compatible layout versus gather-heavy access;
- attention backend compatibility;
- long-context retrieval, perplexity, reasoning and task quality;
- batch/context break-even point;
- interaction with GQA/MLA, quantization, prefix caching, and speculative decoding.

Compression ratio without kernel compatibility and workload quality is not sufficient evidence.

---

## 6. Attention-State Architecture

### 6.1 Sharing across heads

| Order | Source | Code/tutorial bridge |
|---:|---|---|
| 1 | [Multi-Query Attention](../ARCHITECTURE/MQA.pdf) | Transformers model configs and attention modules |
| 2 | [Grouped-Query Attention](../ARCHITECTURE/GQA.pdf) | [LLMs-from-scratch GQA](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/04_gqa) |
| 3 | [DeepSeek-V2 MLA](../ARCHITECTURE/DEEPSEEK-V2.pdf) | [LLMs-from-scratch MLA](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/05_mla) |
| 4 | [DeepSeek-V3](../ARCHITECTURE/DEEPSEEK-V3.pdf) | `RESOURCES/repos/deepseek-v3/inference/` |

Compare KV capacity, decode bytes, projection/reconstruction work, positional encoding interaction,
TP layout, and available attention backends.

### 6.2 Sharing across layers

These papers are related but not interchangeable:

| Source | Mechanism | Code path |
|---|---|---|
| [Cross-Layer Attention](../ARCHITECTURE/CLA.pdf) | adjacent layers reuse K/V heads | paper/config comparison |
| [MLKV](../ARCHITECTURE/MLKV.pdf) | multi-layer KV heads extend head sharing across depth | `RESOURCES/repos/pythia-mlkv/` |
| [YOCO](../ARCHITECTURE/YOCO.pdf) | self-decoder builds reusable global KV for a cross-decoder | `microsoft/unilm/YOCO`, FLA YOCO model |
| [LiSA](../ARCHITECTURE/LISA.pdf) | aligns and approximates attention maps across layers | paper implementation |
| [LLMs-from-scratch Cross-Layer KV Sharing](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/10_kv-sharing) | educational implementation and estimator | shared study checkout |

Exact local paths:

```text
RESOURCES/repos/pythia-mlkv/gpt_neox_mlkv/
RESOURCES/repos/pythia-mlkv/convert_to_mlkv.py
RESOURCES/repos/pythia-mlkv/measure_memory.py
RESOURCES/repos/flash-linear-attention/fla/models/yoco/
```

Distinguish KV sharing from attention-map sharing, and distinguish architectures trained with
sharing from post-hoc conversion/uptraining.

---

## 7. Efficient Attention Families

### 7.1 Exact softmax attention with better IO

| Order | Resource | Exact target |
|---:|---|---|
| 1 | [FlashAttention](../ATTENTION/FLASHATTN.pdf) | IO-awareness, tiling and online softmax |
| 2 | [FlashAttention-2](../ATTENTION/FLASHATTN2.pdf) | work partitioning and occupancy |
| 3 | [FlashAttention-3](../ATTENTION/FLASHATTN3.pdf) | Hopper asynchrony and low precision |
| 4 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | interface, `csrc/`, supported masks/shapes |
| 5 | [GPU Mode Lecture 12](https://github.com/gpu-mode/lectures/tree/main/lecture_012) | implementation and register pressure |
| 6 | [FlashInfer](../ATTENTION/FLASHINFER.pdf) | ragged/paged prefill and decode serving kernels |

Trace separate backends for prefill, decode, paged KV, sliding window, GQA, MLA, and speculative
verification.

### 7.2 Local, sparse, retrieval, and mixed-span attention

| Family | Resource | Read for |
|---|---|---|
| sliding window | [Mistral](../ARCHITECTURE/MISTRAL.pdf), [LLMs-from-scratch SWA](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/06_swa) | bounded local context and rolling cache |
| attention sinks | [StreamingLLM](../CACHE/STREAMINGLLM.pdf) | stable streaming with initial tokens |
| dynamic sparse prefill | [MInference](../ATTENTION/MINFERENCE.pdf) | pattern selection and long-context prefill |
| query-aware sparse decode | [Quest](https://github.com/mit-han-lab/Quest) | page selection and approximate KV access |
| retrieval/streaming heads | [DuoAttention](../ATTENTION/DUOATTENTION.pdf) | heterogeneous head retention |
| model-native sparsity | [Native Sparse Attention](../ATTENTION/NSA.pdf) | trainable hierarchical sparse attention |
| current code tutorial | [LLMs-from-scratch DSA](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa) | readable sparse-attention implementation |

For every method, identify whether sparsity is static, content-dependent, learned, or retrieved;
whether all KV remains stored; and whether selection overhead and memory layout permit real kernel
speedup.

### 7.3 Linear, recurrent, SSM, and hybrid attention

Start with an educational implementation, then follow the architecture/paper progression:

| Order | Source | Main question |
|---:|---|---|
| 1 | [LLMs-from-scratch Gated DeltaNet](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet) | how does recurrent linear attention replace growing KV? |
| 2 | [Transformers are RNNs / Linear Transformer](../ATTENTION/LINEAR-TRANSFORMER.pdf) | feature-map attention and recurrent form |
| 3 | [RetNet](../ARCHITECTURE/RETNET.pdf) | parallel, recurrent and chunkwise equivalence |
| 4 | [Mamba](../ARCHITECTURE/MAMBA.pdf) and [Mamba-2](../ARCHITECTURE/MAMBA2.pdf) | selective state and state-space duality |
| 5 | [Based](../ATTENTION/BASED.pdf) | recall-throughput trade-off and local+linear hybrid |
| 6 | [Gated Linear Attention](../ATTENTION/GLA.pdf) | data-dependent gates and hardware-efficient chunks |
| 7 | [DeltaNet](../ATTENTION/DELTANET.pdf) | delta-rule memory update and parallel training |
| 8 | [Kimi Linear](../ATTENTION/KIMI-LINEAR.pdf) | KDA+MLA hybrid and long-context serving |
| Branch | [Griffin](../ARCHITECTURE/GRIFFIN.pdf) | local attention plus gated recurrence |

Primary implementation:

```text
RESOURCES/repos/flash-linear-attention/fla/models/linear_attn/
RESOURCES/repos/flash-linear-attention/fla/models/gla/
RESOURCES/repos/flash-linear-attention/fla/models/delta_net/
RESOURCES/repos/flash-linear-attention/fla/models/gated_deltanet/
RESOURCES/repos/flash-linear-attention/fla/models/kda/
RESOURCES/repos/flash-linear-attention/fla/models/retnet/
RESOURCES/repos/flash-linear-attention/fla/models/mamba2/
RESOURCES/repos/flash-linear-attention/fla/models/samba/
RESOURCES/repos/flash-linear-attention/fla/ops/
```

Current model/deployment reference:

```text
RESOURCES/repos/kimi-linear/README.md
RESOURCES/repos/kimi-linear/tech_report.pdf
RESOURCES/repos/flash-linear-attention/fla/models/kda/
RESOURCES/repos/flash-linear-attention/fla/ops/kda/
```

Compare recurrent state size, chunkwise prefill, token-by-token update, recall quality, state
migration, batching, preemption, speculative rollback, and production engine support. A linear
asymptotic expression alone does not establish lower latency than optimized softmax attention.

---

## 8. GPU Execution and Kernel Path

### 8.1 CUDA foundations

| Priority | Resource | Exact target |
|---|---|---|
| Core | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | programming model, memory hierarchy, execution |
| Core | [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | memory access, occupancy, synchronization |
| Core | [GPU Mode Lectures](https://github.com/gpu-mode/lectures) | lectures 1–8, 10, 12, 14, 17, 18 |
| Reference | [CUDA Samples](https://github.com/NVIDIA/cuda-samples) | device query, bandwidth, reduction, matrix multiplication |

Local official PDFs:

```text
COURSE/CUDA-PROGRAMMING-GUIDE.pdf
COURSE/CUDA-BEST-PRACTICES.pdf
```

### 8.2 Triton progression

Use the [official Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
in this order:

1. vector addition;
2. fused softmax;
3. matrix multiplication;
4. low-memory dropout;
5. layer normalization;
6. fused attention;
7. grouped GEMM.

Then use:

- [Triton Puzzles](https://github.com/srush/Triton-Puzzles);
- [GPU Mode Triton lectures](https://github.com/gpu-mode/lectures);
- shared Triton checkout at `/home/junyao/code/study/tutorials/gpu-systems/triton/`
  ([workspace map](../RESOURCES/README.md#reused-workspace-instead-of-duplicate-clones));
- `RESOURCES/repos/flash-linear-attention/fla/ops/` for recurrent/chunkwise kernels.

### 8.3 CUTLASS, CuTe, and vendor paths

| Resource | Read for |
|---|---|
| [CUTLASS documentation](https://docs.nvidia.com/cutlass/) | GEMM hierarchy, grouped GEMM and architecture-specific kernels |
| [CuTe DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) | layouts and Python DSL |
| `RESOURCES/repos/cutlass/examples/` | minimal examples before production templates |
| `RESOURCES/repos/deepgemm/` | low-precision GEMM and MoE paths |
| `RESOURCES/repos/flashinfer/` | serving attention, sampling and GEMM/MoE kernels |

For low precision, confirm that the kernel consumes the advertised packed format and accumulator
dtype. A smaller checkpoint routed through an unpack/dequant fallback may save capacity without
improving latency.

---

## 9. Compiler and Runtime Optimization

### 9.1 PyTorch compiler path

| Order | Official resource | Exact target |
|---:|---|---|
| 1 | [torch.compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | capture, guards, recompilation and modes |
| 2 | [PyTorch compiler programming model](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) | graph breaks and dynamic behavior |
| 3 | [TorchInductor debugging](https://docs.pytorch.org/docs/stable/torch.compiler_inductor_profiling.html) | generated-kernel profiling |
| 4 | [CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) | static-address and replay constraints |

Inspect:

```text
torch/_dynamo/
torch/_inductor/
torch/_functorch/
torch/csrc/dynamo/
```

### 9.2 Engine/runtime mechanisms

Read these in the next stage's engine source, but classify their optimization layer here:

| Mechanism | Resource effect | Primary source |
|---|---|---|
| PagedAttention | allocation waste and KV placement | [vLLM paper](../SERVING/VLLM.pdf) |
| prefix caching | avoid repeated prefill | vLLM/SGLang cache docs and source |
| continuous batching | GPU utilization and queueing | [Orca](../SERVING/ORCA.pdf) |
| chunked prefill | prefill/decode interference | [Sarathi-Serve](../SERVING/SARATHI.pdf) |
| CUDA Graph | launch/CPU overhead | PyTorch and engine graph-runner source |
| attention backend selection | shape/layout-specific kernel | vLLM/SGLang/TensorRT-LLM |
| workspace/pool sizing | capacity and fragmentation | engine worker/model-runner source |

Keep algorithmic compression separate from physical memory management and online scheduling, even
when they interact in the same engine.

---

## 10. Production Source Trace

For any selected optimization, locate all four layers:

```text
model config and checkpoint metadata
→ engine feature/config selection
→ backend/kernel dispatch
→ benchmark and quality evaluation
```

### vLLM

```text
RESOURCES/repos/vllm/vllm/model_executor/models/
RESOURCES/repos/vllm/vllm/model_executor/layers/attention/
RESOURCES/repos/vllm/vllm/model_executor/layers/quantization/
RESOURCES/repos/vllm/vllm/v1/attention/
RESOURCES/repos/vllm/vllm/v1/worker/
```

### SGLang

```text
RESOURCES/repos/sglang/python/sglang/srt/models/
RESOURCES/repos/sglang/python/sglang/srt/layers/attention/
RESOURCES/repos/sglang/python/sglang/srt/layers/quantization/
RESOURCES/repos/sglang/python/sglang/srt/mem_cache/
```

### TensorRT-LLM

```text
RESOURCES/repos/tensorrt-llm/tensorrt_llm/_torch/models/
RESOURCES/repos/tensorrt-llm/tensorrt_llm/_torch/modules/
RESOURCES/repos/tensorrt-llm/tensorrt_llm/_torch/pyexecutor/
RESOURCES/repos/tensorrt-llm/examples/
```

Pin the engine commit. Feature support, hardware matrices, fallback behavior, and source layout
change faster than the foundational papers.

---

## 11. Selection Matrix

Use this only after reading the linked sources.

| Observed bottleneck | Candidate branches | Required counter-check |
|---|---|---|
| weights do not fit | weight-only/mixed quantization, TP, offload | kernel support and quality |
| decode reads dominate | GQA/MQA/MLA, KV quantization, linear/hybrid attention | model availability and decode backend |
| KV capacity dominates | cross-layer sharing, lower KV precision, retention, window, tiering | semantic/quality class |
| long prefill compute dominates | FlashAttention, sparse attention, chunking, CP | selection overhead and TTFT |
| allocator waste dominates | paged/virtual memory, pool sizing | metadata and gather cost |
| short kernels/CPU gaps dominate | fusion, compile, CUDA Graph, scheduler path | dynamic-shape/recompilation limits |
| low-bit checkpoint is not faster | packed kernel, shape support, fused dequant | fallback and conversion overhead |
| pruned model is not faster | structured/N:M format, sparse kernel, smaller dense architecture | zero weights still use dense kernels |
| linear attention underperforms | hybrid full-attention layers, state size, training recipe | recall and long-context quality |

---

## 12. Repository Index

| Priority | Repository | Role |
|---|---|---|
| A | [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | readable memory, GQA, MLA, SWA, DeltaNet, DSA and KV-sharing tutorials |
| A | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | linear/recurrent/sparse/hybrid models and kernels |
| A | [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) | deployment quantization for vLLM |
| A | [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | NVIDIA quantization, sparsity and deployment |
| A | [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) | KV compression implementations and evaluation |
| A | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | exact IO-aware attention |
| A | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | serving-oriented attention and inference kernels |
| A | [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) | current hybrid linear-attention model and deployment |
| B | [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention) | retrieval/streaming head split and KV reduction |
| B | [zaydzuhri/pythia-mlkv](https://github.com/zaydzuhri/pythia-mlkv) | cross-layer KV-sharing reference |
| B | [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest) | query-aware sparse attention |
| B | [SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant) | low-bit KV reference and kernels |
| B | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | GEMM/CuTe primitives |
| B | [gpu-mode/lectures](https://github.com/gpu-mode/lectures) | GPU and kernel course |

---

## What to Defer

Until a concrete bottleneck selects the branch, do not read linearly through:

- every quantization algorithm or checkpoint wrapper;
- every sparse-attention mask;
- every KV eviction paper;
- all FLA models and kernels;
- all vLLM quantization backends;
- all CUTLASS examples;
- training-only compression details that do not affect the target inference path.

Breadth is represented by the taxonomy and repository atlas. Depth should follow one measured
bottleneck.

---

## Exit Gate

This layer is complete when you can:

- [ ] derive a parameter/KV/workspace prediction from an external source and reconcile it with
      measured allocated and reserved memory;
- [ ] classify an optimization by semantics, resource, implementation layer, and workload effect;
- [ ] distinguish head sharing, latent KV, cross-layer sharing, retention, sparse attention, linear
      attention, exact IO-aware attention, and paging;
- [ ] compare weight-only, W+A, KV, attention, and MoE quantization with explicit dtype,
      granularity, calibration and kernel paths;
- [ ] trace one optimization from checkpoint/config through production engine dispatch to kernel;
- [ ] use profiler evidence to explain why a theoretical reduction does or does not improve TTFT,
      TPOT/ITL, throughput, capacity, or cost;
- [ ] state the quality contract and at least one regression/break-even region.

---

**Previous:** [`03-modern-llm-architecture.md`](03-modern-llm-architecture.md) ·
**Next:** [`05-single-node-inference-engine.md`](05-single-node-inference-engine.md) ·
**Evaluation:** [`COMPETENCY-GATES.md`](COMPETENCY-GATES.md)
