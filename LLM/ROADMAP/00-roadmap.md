# LLM and AI Systems Research Roadmap

> **Target:** database / systems researcher → LLM inference and AI systems researcher
>
> **Center of gravity:** efficient, correct and reliable inference
>
> **Coverage:** prerequisites, Transformer/LLM architecture, training and post-training systems,
> model/runtime optimization, decoding, single-node engines, online serving, distributed inference,
> MoE, production operations and bottleneck-driven research
>
> **Organization rule:** external English tutorials and official documentation → representative
> papers → reference repositories → production source paths. These files are maps, not rewritten
> tutorials. There is no calendar schedule.

[Roadmap index](README.md) ·
[Competency gates](COMPETENCY-GATES.md) ·
[Repository atlas](../RESOURCES/GITHUB-REPO-ATLAS.md) ·
[Paper provenance](../RESOURCES/SOURCES.md)

---

## 1. End Goal

Build enough model knowledge to reason correctly about execution, then go deep on the systems layers
that determine latency, throughput, capacity, reliability, quality and cost:

```text
AI/ML + PyTorch
→ Transformer and generation mechanics
→ modern LLM architecture
→ training and post-training systems
→ compression, efficient attention, GPU kernels and compilers
→ decoding, speculative execution and test-time compute
→ single-node inference engine
→ KV/state, scheduling and online serving
→ distributed inference and MoE
→ production platform and reliability
→ bottleneck-driven systems research
```

The target capability is not “know many model or paper names.” It is:

```text
identify the semantic contract
→ predict compute / memory / communication / state / queue cost
→ locate the real implementation
→ measure the limiting resource
→ preserve correctness and quality
→ state the workload and break-even boundary
```

---

## 2. Main Dependency Path

| Layer | Document | Scope | Representative external entry |
|---:|---|---|---|
| 01 | [`01-ai-ml-foundations.md`](01-ai-ml-foundations.md) | linear algebra, probability, optimization, neural networks, autograd and PyTorch | [Dive into Deep Learning](https://d2l.ai/) |
| 02 | [`02-transformer-foundations.md`](02-transformer-foundations.md) | tokenization, attention, decoder LM, prefill/decode, KV and Hugging Face | [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) |
| 03 | [`03-modern-llm-architecture.md`](03-modern-llm-architecture.md) | GPT/Llama/Qwen/DeepSeek, GQA/MQA/MLA, MoE, sparse/linear/recurrent/diffusion architectures | [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) |
| 04 | [`04-training-post-training-systems.md`](04-training-post-training-systems.md) | data, tokenizer, scaling, distributed pretraining, SFT/PEFT, preference learning, RL rollout systems | [Stanford CS336](https://cs336.stanford.edu/) |
| 05 | [`05-single-node-inference-optimization.md`](05-single-node-inference-optimization.md) | quantization, pruning, KV reduction, efficient attention, CUDA/Triton/CUTLASS, compiler/runtime | [Scaling Book — Inference](https://jax-ml.github.io/scaling-book/inference/) |
| 06 | [`06-decoding-test-time-compute.md`](06-decoding-test-time-compute.md) | sampling/search, structured outputs, speculative decoding, parallel/diffusion decoding, reasoning compute | [HF generation strategies](https://huggingface.co/docs/transformers/generation_strategies) |
| 07 | [`07-single-node-inference-engine.md`](07-single-node-inference-engine.md) | request state, scheduler, model runner, attention backend, sampler and streaming | [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) |
| 08 | [`08-kv-scheduling-serving.md`](08-kv-scheduling-serving.md) | KV/state, batching, admission, scheduling, prefix reuse, disaggregation and serving metrics | [vLLM](../SERVING/VLLM.pdf) and [Orca](../SERVING/ORCA.pdf) |
| 09 | [`09-distributed-inference-moe.md`](09-distributed-inference-moe.md) | TP/PP/DP/CP/EP, collectives, topology, placement, MoE kernels/routing/skew | [Megatron-LM](../PARALLEL/MEGATRON.pdf) |
| 10 | [`10-production-reliability.md`](10-production-reliability.md) | gateways, lifecycle, observability, overload, recovery, isolation, security, cost and energy | [Google SRE Book](https://sre.google/sre-book/table-of-contents/) |
| 11 | [`11-bottleneck-research.md`](11-bottleneck-research.md) | workload/bottleneck matrix, falsifiable questions and evidence standards | [MLSys proceedings](https://proceedings.mlsys.org/) |

The path is dependency-ordered, not time-boxed. Enter at the first layer whose exit gate cannot
already be defended.

---

## 3. Complete Topic Architecture

### Layer 01 — AI, ML, Mathematics, and PyTorch

Sub-aspects:

- tensors, shapes, dtypes, layouts and devices;
- matrix multiplication, vector spaces, projections, eigendecomposition, SVD and PCA;
- probability, likelihood, entropy, cross-entropy and KL divergence;
- calculus, gradients, SGD, Adam/AdamW and generalization;
- MLPs, activations, normalization, residual paths and computation graphs;
- PyTorch modules, autograd, optimizers, data loading and model state.

Read [`01-ai-ml-foundations.md`](01-ai-ml-foundations.md).

### Layer 02 — Transformer and Language-Model Foundations

Sub-aspects:

- normalization, pre-tokenization, BPE/WordPiece/Unigram and special tokens;
- embeddings, causal next-token likelihood and cross-entropy;
- Q/K/V, attention masks, multi-head attention and positional encoding;
- FFN/GLU, residual connections and normalization;
- encoder-only, encoder-decoder and decoder-only families;
- training, prefill, decode, autoregressive generation and KV cache;
- Hugging Face config/tokenizer/model/generation/cache interfaces.

Representative works: [Attention Is All You Need](../FOUNDATION/ATTENTION.pdf), GPT-family model
papers and executable external tutorials in
[`02-transformer-foundations.md`](02-transformer-foundations.md).

### Layer 03 — Modern LLM Architecture

Sub-aspects:

- GPT, Llama, Mistral, Qwen, DeepSeek and other decoder families;
- learned/sinusoidal positions, RoPE, ALiBi, YaRN and LongRoPE;
- Pre/Post-Norm, RMSNorm, LayerNorm, SwiGLU and residual variants;
- MHA, MQA, GQA and MLA;
- full, sliding-window, local, sparse, retrieval and hybrid attention;
- cross-layer KV/attention sharing: CLA, MLKV, YOCO and related designs;
- dense FFN, MoE, fine-grained/shared experts and conditional depth;
- linear/recurrent attention, RetNet, RWKV, Mamba/SSM and hybrid models;
- MTP/speculative heads and early-exit-compatible models;
- autoregressive, masked-diffusion and block-diffusion language models;
- multimodal encoders/projectors/token streams and VLM request shapes.

Read [`03-modern-llm-architecture.md`](03-modern-llm-architecture.md) and its linked primary papers.

### Layer 04 — Training and Post-Training Systems

Sub-aspects:

- data acquisition, extraction, filtering, deduplication, decontamination, mixture and streaming;
- tokenizer training, chat templates and training/serving parity;
- causal/masked/span/FIM/multimodal objectives and scaling laws;
- initialization, AdamW, schedules, mixed precision, checkpointing and profiling;
- DDP, FSDP/ZeRO, TP, PP, SP/CP, EP and distributed checkpoints;
- continued pretraining, SFT, LoRA/QLoRA and multi-adapter implications;
- preference data, reward modeling, RLAIF and process/outcome rewards;
- DPO/IPO/KTO/ORPO/SimPO-family offline preference optimization;
- PPO/GRPO-family online RL and reasoning post-training;
- rollout generation, weight synchronization, policy staleness and agent environments;
- training–inference co-design for tokenizer, KV, MoE, MTP, precision and checkpoints.

Read [`04-training-post-training-systems.md`](04-training-post-training-systems.md).

### Layer 05 — Model, State, Kernel, and Runtime Optimization

Sub-aspects:

- source-grounded parameter/FLOP/byte/memory models and profiler reconciliation;
- weight-only, weight-activation, KV, attention and MoE quantization;
- INT8/INT4, FP8/FP4/NVFP4/MXFP4 and mixed precision;
- GPTQ, AWQ, SmoothQuant, LLM.int8(), ZeroQuant, rotations and QAT;
- pruning, structural sparsity, distillation and low-rank/structural reduction;
- KV quantization, compression, eviction, selection, retrieval, tiering and offload;
- GQA/MQA/MLA and cross-layer state sharing;
- sliding-window, sparse, linear/recurrent/SSM and hybrid attention;
- exact IO-aware attention: FlashAttention family and FlashInfer;
- CUDA execution, Triton, CUTLASS/CuTe, fusion and CUDA Graphs;
- TorchInductor, TensorRT-LLM and compiler/IR/backend dispatch;
- hardware portability across NVIDIA, AMD, TPU, CPU and edge runtimes.

Read [`05-single-node-inference-optimization.md`](05-single-node-inference-optimization.md).

### Layer 06 — Decoding, Speculative Execution, and Test-Time Compute

Sub-aspects:

- greedy, temperature, top-k/top-p/min-p, typical/eta and contrastive decoding;
- beam, diverse beam, best-of-N and parallel sampling;
- JSON/regex/CFG/tool-call constrained generation;
- classic small-draft/target speculative decoding and rejection sampling;
- self-speculation, early exit and layer skipping;
- Medusa, Hydra, MTP, ReDrafter and parallel draft heads;
- EAGLE, EAGLE-2/3 and feature-level drafting;
- prompt lookup, N-gram, suffix and REST retrieval proposals;
- SpecInfer/Sequoia tree construction and block verification;
- Lookahead/Jacobi/parallel decoding;
- TriForce/MagicDec/LongSpec long-context speculation;
- block-diffusion/DFlash draft generation and diffusion-LM decoding;
- draft/target placement, verification kernels, provisional KV, rollback and compaction;
- dynamic speculation under batching and scheduler pressure;
- correctness/distribution preservation and reproducibility;
- self-consistency, verifier-guided search, Tree of Thoughts, budget forcing and adaptive
  test-time compute;
- agent/tool multi-call execution and branch-state management.

The pinned 30-paper speculative set, external intros, reference implementations and production
paths are all indexed by
[`06-decoding-test-time-compute.md`](06-decoding-test-time-compute.md).

### Layer 07 — Single-Node Inference Engine

Sub-aspects:

- API/request/sequence lifecycle;
- tokenizer/input processing and output streaming;
- request queues and continuous batching;
- prefill/decode mixing and token budgets;
- paged/block KV allocation and prefix reuse;
- preemption, swap, recompute and resume;
- worker/model runner and execution plans;
- attention/backend dispatch and CUDA Graph capture;
- logits, sampling, structured output and speculative hooks;
- CPU/GPU coordination and observability.

Read [`07-single-node-inference-engine.md`](07-single-node-inference-engine.md).

### Layer 08 — KV, Scheduling, and Online Serving

Sub-aspects:

- workload shape, arrival process, prefix structure and output uncertainty;
- TTFT, TPOT/ITL, E2E latency, throughput, goodput, fairness and cost;
- routing, admission and backpressure;
- static/continuous batching and chunked prefill;
- decode-first, priority, deadline and fair scheduling;
- prefix cache, paged/virtual memory and multi-tier state;
- KV quantization/compression/eviction and remote state;
- multi-LoRA, structured-output, multimodal and reasoning request scheduling;
- prefill/decode and encode/prefill/decode disaggregation;
- cache-aware routing, autoscaling, migration and fault recovery;
- simulation, trace replay and workload-valid evaluation.

Read [`08-kv-scheduling-serving.md`](08-kv-scheduling-serving.md).

### Layer 09 — Distributed Inference and MoE

Sub-aspects:

- accelerator/node/rack topology and collective cost;
- TP, PP, DP/replicas, CP/SP and EP;
- PCIe, NVLink/NVSwitch, InfiniBand/RoCE, RDMA and collective libraries;
- communication/compute overlap and topology-aware placement;
- heterogeneous and cross-accelerator execution;
- disaggregated prefill/decode, remote KV and expert services;
- MoE routing, capacity, auxiliary losses and load balance;
- grouped GEMM, sparse kernels, fused MoE and quantized experts;
- all-to-all dispatch/combine, expert placement, replication, caching and offload;
- online MoE batching, hot experts, failures and observability.

Read [`09-distributed-inference-moe.md`](09-distributed-inference-moe.md).

### Layer 10 — Production Platform, Lifecycle, and Reliability

Sub-aspects:

- API gateway, authentication, quota, routing and request normalization;
- model registry, artifact lineage, checkpoint conversion and supply-chain safety;
- model loading, cold start, warm pools, multi-model placement and adapter lifecycle;
- control plane/data plane separation;
- metrics, traces, logs, profiling and workload telemetry;
- overload, admission, backpressure, degradation and autoscaling;
- GPU/worker/node/network/cache failures and recovery;
- cancellation, retry, replay, idempotency and state consistency;
- multi-tenant isolation, privacy, security and abuse resistance;
- correctness canaries, output parity and safe rollout/rollback;
- capacity planning, cost, energy and carbon;
- Kubernetes/operator and platform integration.

Read [`10-production-reliability.md`](10-production-reliability.md).

### Layer 11 — Bottleneck-Driven Research

Read [`11-bottleneck-research.md`](11-bottleneck-research.md) for workload matrices, bottleneck
classes, open systems questions, representative work and evidence standards. It is a research
index, not a project list or schedule.

---

## 4. Cross-Cutting Coverage Matrix

This matrix prevents a one-line mention from being mistaken for complete coverage.

| Cross-cutting aspect | Architecture/model | Kernel/runtime | Engine/serving | Distributed/production |
|---|---|---|---|---|
| attention and persistent state | MHA/MQA/GQA/MLA/CLA/SSM | FlashAttention, sparse/linear kernels, KV format | paging, eviction, prefix cache | remote/tiered KV and transfer |
| quantization/compression | trained/post-training precision, pruning | low-bit GEMM/attention/MoE kernels | loader/backend/fallback | communication dtype and fleet compatibility |
| speculative decoding | MTP/Medusa/EAGLE/early exit | draft + verify kernels, tree masks | token budgets, rollback, dynamic depth | draft/target placement |
| test-time reasoning | reasoning checkpoint and verifier | long decode/branch kernels | branch scheduling, shared prefixes | verifier/tool placement and quality-cost SLO |
| MoE | routing and expert design | grouped GEMM and fused MoE | skew-aware batching | EP, all-to-all and expert placement |
| multimodal/VLM | encoder/projector/visual tokens | vision/audio kernels and preprocessing | encode/prefill/decode lifecycle | encoder disaggregation and modality-aware routing |
| embeddings/rerankers/encoders | encoder objectives and pooling | dense/batched encoder kernels | non-generative request path | multi-model pipelines |
| structured outputs/tools | tokenizer/schema semantics | grammar mask kernels | parser state and tool wait/resume | sandbox/service reliability |
| compiler/IR | graph and operator semantics | TorchInductor/Triton/TensorRT/MLIR | graph buckets and fallback | heterogeneous backend fleet |
| hardware | precision/layout constraints | CUDA/ROCm/TPU/CPU/edge | memory capacity and concurrency | NVLink/RDMA/CXL/fabric and placement |
| correctness | model/tokenizer contract | numerical parity | sampling/cache/state parity | rollout, recovery and version consistency |

---

## 5. Representative Architecture and Systems Lineage

The detailed documents contain the complete categorized lists. This table gives the canonical
lineage that should be recognizable before specializing.

| Branch | Representative works |
|---|---|
| Transformer and decoder LM | Attention Is All You Need, GPT family, Llama family |
| positions and decoder blocks | RoPE, ALiBi, RMSNorm, SwiGLU, YaRN/LongRoPE |
| KV-head/state reduction | MQA, GQA, MLA, CLA, MLKV, YOCO |
| efficient attention | FlashAttention 1/2/3, FlashInfer, StreamingLLM, MInference, Native Sparse Attention |
| linear/recurrent/SSM | Linear Transformer, RetNet, RWKV, Mamba/Mamba-2, GLA, DeltaNet, Kimi Linear |
| MoE | GShard, Switch, ST-MoE, Expert Choice, MegaBlocks, Mixtral, DeepSeekMoE/V2/V3 |
| quantization | LLM.int8(), ZeroQuant, GPTQ, SmoothQuant, AWQ, Atom, QuaRot, SpinQuant, KIVI/KVQuant |
| serving | Orca, vLLM/PagedAttention, SGLang, Sarathi-Serve, DistServe, Splitwise, Mooncake, Llumnix, Preble |
| speculative decoding | SpecDecode, Speculative Sampling, Medusa, EAGLE 1/2/3, SpecInfer, Sequoia, LayerSkip, REST, TriForce, MagicDec |
| training/post-training | Scaling Laws, Chinchilla, Megatron, ZeRO, LoRA/QLoRA, InstructGPT, DPO, GRPO, DeepSeek-R1 |
| test-time compute | Self-Consistency, Tree of Thoughts, process reward models, compute-optimal test-time scaling, s1 |

Local PDFs and provenance are indexed in [`../RESOURCES/SOURCES.md`](../RESOURCES/SOURCES.md).

---

## 6. Source-Grounded Cost Model

The roadmap does not invent a new cost-model tutorial. Use:

| Topic | External source | Roadmap use |
|---|---|---|
| compute, bandwidth and capacity | [Scaling Book — Rooflines](https://jax-ml.github.io/scaling-book/roofline/) | lower bounds and arithmetic intensity |
| Transformer shapes/FLOPs | [Scaling Book — Transformer Math](https://jax-ml.github.io/scaling-book/transformers/) | operator and parameter ledger |
| training | [Scaling Book — Training](https://jax-ml.github.io/scaling-book/training/) | activations, optimizer state and distributed cost |
| inference | [Scaling Book — Inference](https://jax-ml.github.io/scaling-book/inference/) | prefill/decode/KV and batching |
| GPU/framework overhead | [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | launches, fusion, memory and compiler boundary |
| memory measurement | [PyTorch CUDA memory](https://docs.pytorch.org/docs/main/torch_cuda_memory) | allocated/reserved/external memory |
| GPU kernel evidence | [Nsight Systems](https://docs.nvidia.com/nsight-systems/) and [Nsight Compute](https://docs.nvidia.com/nsight-compute/) | timeline and per-kernel evidence |
| network collectives | [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | payload and synchronization |
| serving metrics | Orca, DistServe and Sarathi-Serve | TTFT/TPOT/E2E/goodput |
| production objectives | [Google SRE — SLOs](https://sre.google/sre-book/service-level-objectives/) | SLI/SLO/error budget |

Required record for any performance claim:

```text
model + tokenizer + revision
workload and arrival distribution
prefill / decode / verification / encode / tool phase
tensor shapes, layouts and dtypes
weights, activations, KV/state, workspaces and allocator pools
FLOPs, bytes, launches, barriers, collectives and transfers
parallelism and topology
predicted bottleneck
measured profiler + service evidence
correctness / quality / SLO contract
negative cases and break-even boundary
```

---

## 7. External English Learning Spine

| Source | Primary use |
|---|---|
| [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en) | AI/ML, mathematics, neural-network and PyTorch prerequisites |
| [PyTorch Tutorials](https://github.com/pytorch/tutorials) | official framework mechanics |
| [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer) | executable original Transformer |
| [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) | decoder basics plus GQA/MLA/KV/efficient-attention intros |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/) | tokenizer, models, fine-tuning and generation |
| [Stanford CS336](https://cs336.stanford.edu/) | end-to-end LM data/training/systems/evaluation |
| [GPU MODE Lectures](https://github.com/gpu-mode/lectures) | GPU, Triton, CUDA, kernels and communication |
| [Scaling Book](https://jax-ml.github.io/scaling-book/) | quantitative training/inference/distributed systems |
| [Efficient Deep Learning Systems](https://github.com/mryab/efficient-dl-systems) | profiling, compilation and deployment |
| [MLSysBook](https://github.com/harvard-edge/cs249r_book) | deployment, reliability, security and broader ML systems |
| [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | production inference source |

The full categorized GitHub map is
[`../RESOURCES/GITHUB-REPO-ATLAS.md`](../RESOURCES/GITHUB-REPO-ATLAS.md).

---

## 8. Database-Researcher Entry Points

| Database/systems concept | AI-systems analogue |
|---|---|
| buffer pool and paging | KV/state allocation, reuse, eviction, tiering and migration |
| query scheduling and admission | continuous batching, token budgets, preemption, fairness and SLOs |
| materialized views/cache | prefix cache, prompt reuse, shared branch state |
| transactions/recovery | request state commit/rollback, cancellation, replay and recovery |
| partitioning/skew | model/expert placement, MoE hot experts and load balance |
| disaggregation | prefill/decode, remote KV, draft/target and expert services |
| query optimization | kernel/backend/parallelism/placement selection |
| adaptive execution | dynamic batching, speculative depth and test-time compute allocation |
| provenance and governance | data/checkpoint lineage, model registry and rollout safety |

---

## 9. Competency and Completion Standard

[`COMPETENCY-GATES.md`](COMPETENCY-GATES.md) tests whether a topic can be derived, traced and
measured. The roadmap is complete when you can:

- explain the AI/ML and Transformer foundations without treating framework calls as magic;
- map a modern checkpoint/configuration to compute, memory, state, kernel and communication cost;
- understand how data, tokenizer, training and post-training choices affect inference;
- distinguish quantization, KV reduction, sparse/linear attention and exact IO-aware kernels;
- trace ordinary, structured, speculative, diffusion and test-time reasoning generation paths;
- trace a request through frontend, scheduler, model runner, attention backend, state manager,
  sampler and stream;
- choose parallelism and placement from workload, topology and SLO;
- analyze dense and MoE inference, including routing, grouped GEMM, communication and skew;
- reason about multimodal, embedding/reranking, multi-model and agent workloads rather than assuming
  every request is text generation;
- preserve tokenizer, numerical, sampling, cache and recovery correctness;
- reason about overload, failures, isolation, lifecycle, security, cost and energy;
- convert a measured bottleneck into a falsifiable and reproducible systems research claim.

---

**Next:** [`01-ai-ml-foundations.md`](01-ai-ml-foundations.md) ·
**Evaluation:** [`COMPETENCY-GATES.md`](COMPETENCY-GATES.md)
