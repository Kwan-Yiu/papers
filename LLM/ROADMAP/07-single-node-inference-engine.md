# Single-Node LLM Inference Engine — Curated Reading Map

> **Role:** transition from framework generation to a serving engine
> **Target:** trace request state, scheduling, paged KV allocation, model execution, attention, sampling, and streaming in readable and production codebases
> **Format:** external English tutorials, design documents, blogs, papers, and exact GitHub source paths
> **Not included:** an original inference-engine tutorial or a calendar

[Roadmap](00-roadmap.md) ·
[Decoding and test-time compute](06-decoding-test-time-compute.md) ·
[Serving systems](08-kv-scheduling-serving.md) ·
[Competency gates](COMPETENCY-GATES.md)

---

## Recommended Traversal

```text
Hugging Face generate()
        |
        v
tiny readable engine
        |
        +--> request / sequence state
        +--> scheduler
        +--> KV block manager
        +--> model runner
        +--> attention backend
        +--> sampler
        |
        v
production engine: vLLM or SGLang
        |
        v
second production engine for comparison
```

Use:

- **Core** — required;
- **Branch** — follow when it matches the research direction;
- **Reference** — consult on demand;
- **Local clone** — present under `../RESOURCES/repos/`;
- **Link** — keep remote unless executing or modifying.

---

## Coverage Checklist

### Engine boundary

- [ ] offline versus online inference;
- [ ] API server, tokenizer process, engine core, worker, model runner, and output processor;
- [ ] request ID, prompt tokens, generated tokens, sampling parameters, status, and finish reason;
- [ ] asynchronous frontend versus synchronous model step.

### Scheduling and state

- [ ] waiting, running, preempted, swapped, and finished states;
- [ ] static batching, continuous batching, iteration-level scheduling, and chunked prefill;
- [ ] token budget, sequence budget, KV-block budget, and preemption policy;
- [ ] prefill-only, decode-only, and mixed batches;
- [ ] fairness, head-of-line blocking, and starvation.

### Memory and execution

- [ ] paged KV layout, block table, allocation, free, copy-on-write, and prefix reuse;
- [ ] model loading, weight layout, input preparation, CUDA Graphs, and distributed executor boundary;
- [ ] prefill and decode attention backends;
- [ ] logits processing, sampling, stop criteria, and streaming;
- [ ] CPU launch path and synchronization points.

---

## 1. Framework Baseline

Start from the framework path established in [02-transformer-foundations.md](02-transformer-foundations.md).

| Priority | Resource | Exact target | Purpose |
|---|---|---|---|
| Core | [Hugging Face — Optimizing LLMs for speed and memory](https://huggingface.co/docs/transformers/llm_tutorial_optimization) | autoregressive loop and KV-cache sections | establish the non-serving baseline |
| Core | [Hugging Face generation source](https://github.com/huggingface/transformers/tree/main/src/transformers/generation) | `utils.py`, logits processors, stopping criteria | identify what the engine must replace or reorganize |
| Core | [Orca](../SERVING/ORCA.pdf) | iteration-level scheduling and selective batching | conceptual bridge to continuous batching |
| Core | [vLLM paper](../SERVING/VLLM.pdf) | PagedAttention, block management, scheduler, evaluation | memory-management bridge |
| Core | [vLLM launch blog](https://blog.vllm.ai/2023/06/20/vllm.html) | design overview and benchmark setup | concise system overview |

Required baseline trace:

1. tokenize one request;
2. locate the framework generation loop;
3. identify model forward, cache update, logits processing, sampling, and stopping;
4. list which steps are per request and which can be batched;
5. list which state must persist between decode iterations.

---

## 2. First Readable Engine: Nano-vLLM

Repository: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

Nano-vLLM is the shortest Python path from an API resembling vLLM to an engine with a scheduler, paged cache, model runner, tensor parallelism, `torch.compile`, and CUDA Graphs.

### 2.1 Exact source order

| Order | Source path | Read for |
|---:|---|---|
| 1 | `example.py` | public API and output shape |
| 2 | `nanovllm/llm.py` | user-facing wrapper and engine construction |
| 3 | `nanovllm/config.py` | model/runtime/KV capacity configuration |
| 4 | `nanovllm/engine/sequence.py` | request state and lifecycle |
| 5 | `nanovllm/engine/scheduler.py` | waiting/running queues and batch selection |
| 6 | `nanovllm/engine/block_manager.py` | physical blocks, allocation, free, prefix/hash behavior |
| 7 | `nanovllm/engine/llm_engine.py` | step loop and component coordination |
| 8 | `nanovllm/engine/model_runner.py` | input preparation, eager/graph execution, cache tensors |
| 9 | `nanovllm/layers/attention.py` | attention call and KV write/read |
| 10 | `nanovllm/layers/sampler.py` | logits-to-token path |
| 11 | `nanovllm/models/qwen3.py` | architecture integration |
| 12 | `nanovllm/utils/loader.py` | checkpoint loading |
| 13 | `bench.py` | supplied benchmark workload and its limits |

### 2.2 Required trace

For one two-request batch, annotate:

- request creation;
- queue insertion;
- scheduler decision;
- logical-to-physical KV mapping;
- model-runner inputs;
- prefill/decode distinction;
- cache write;
- sampling;
- sequence update;
- block release.

Do not accept benchmark numbers from the README without reproducing the workload and controlling hardware, model revision, input/output lengths, and decoding configuration.

---

## 3. C++/CUDA Path: Tiny-vLLM

Repository: [jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm)

The repository README is itself an external course. It covers model-file loading, CPU/GPU memory, Llama forward, CUDA kernels, KV cache, static and continuous batching, online softmax, PagedAttention, and paged KV.

### 3.1 README course sections

Follow the README table of contents in this grouped order:

| Group | README sections | Source anchor |
|---|---|---|
| model representation | Safetensors; floating point; GPU/CPU memory | `src/main.cpp` |
| decoder operators | embedding; RMSNorm; RoPE; GEMM; attention; GQA; SiLU; softmax; FFN | `src/kernels.cu`, `src/kernels.cuh` |
| inference phases | single-token inference; prefill vs decode; KV cache | `src/main.cpp` |
| request batching | buffer reuse; static batching; continuous batching | `src/main.cpp` |
| attention optimization | online softmax; PagedAttention; paged KV; paged-attention CUDA kernel | `src/kernels.cu` |
| validation | tokenizer and Python reference | `python/reference.py`, `python/decode_test.py`, `python/tokenizer.py` |
| profiling | supplied scripts | `ncu.sh`, `nsys.sh` |

### 3.2 Use this branch when

- explicit C++ ownership and memory allocation matter;
- PyTorch abstractions hide the control path;
- a CUDA kernel should be tied directly to an engine feature;
- safetensors loading and weight layout need to be traced;
- CPU/GPU overlap is a target.

The repository evolves rapidly. Pin a commit when citing its behavior.

---

## 4. Intermediate Modern Engine: Mini-SGLang

Repository: [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)
Design blog: [Mini-SGLang: Efficient Inference Engine in a Nutshell](https://www.lmsys.org/blog/2025-12-17-minisgl/)

Mini-SGLang is a compact bridge to modern features including RadixAttention, chunked prefill, overlap scheduling, tensor parallelism, JIT CUDA kernels, and MoE.

### 4.1 Documentation first

| Order | Path | Read for |
|---:|---|---|
| 1 | `README.md` | scope, supported features, run path |
| 2 | `docs/structures.md` | processes, messages, request and batch structures |
| 3 | `docs/features.md` | cache, scheduling, graph, and parallel features |
| 4 | `benchmark/offline/` and `benchmark/online/` | supplied workloads and metrics |

### 4.2 Source order

| Order | Source path | Read for |
|---:|---|---|
| 1 | `python/minisgl/llm/llm.py` | public LLM/server boundary |
| 2 | `python/minisgl/core.py` | runtime composition |
| 3 | `python/minisgl/engine/engine.py` | step loop |
| 4 | `python/minisgl/scheduler/` | scheduling and batch state; confirm current tree |
| 5 | `python/minisgl/kvcache/base.py` | cache interface |
| 6 | `python/minisgl/kvcache/naive_cache.py` | baseline cache |
| 7 | `python/minisgl/kvcache/radix_cache.py` | prefix tree, matching, reuse, eviction |
| 8 | `python/minisgl/engine/graph.py` | CUDA Graph path |
| 9 | `python/minisgl/attention/` | backend interface and implementations |
| 10 | `python/minisgl/engine/sample.py` | sampling |
| 11 | `python/minisgl/distributed/` | tensor-parallel communication |
| 12 | `python/minisgl/kernel/` | JIT, radix, NCCL, Triton MoE |

If a listed directory moves, use `docs/structures.md` and symbol search from `engine.py`; do not infer current behavior from an older blog post.

---

## 5. Production Engine Spine: vLLM

Repository: [vllm-project/vllm](https://github.com/vllm-project/vllm)
Local clone: `../RESOURCES/repos/vllm/`
Official overview: [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)

### 5.1 Documentation path

| Priority | Resource | Read for |
|---|---|---|
| Core | [Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/) | processes, engine core, worker, model runner, model |
| Core | [PagedAttention design](https://docs.vllm.ai/en/latest/design/paged_attention/) | current implementation notes and kernel layout |
| Core | [Optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) | chunked prefill, batching, memory knobs |
| Branch | [Parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) | executor and parallel deployment boundary |
| Reference | [vLLM developer documentation](https://docs.vllm.ai/en/latest/contributing/model/) | model integration and contribution conventions |

Use unversioned `latest` pages for orientation and pin the repository commit for source claims.

### 5.2 V1 source path

| Order | Source path in local clone | Question |
|---:|---|---|
| 1 | `vllm/entrypoints/llm.py` | how does offline inference enter the engine? |
| 2 | `vllm/entrypoints/openai/api_server.py` | how does online serving differ? |
| 3 | `vllm/v1/engine/core.py` | what owns the central step loop? |
| 4 | `vllm/v1/core/sched/scheduler.py` | how are requests and token budgets selected? |
| 5 | `vllm/v1/core/sched/output.py` | what schedule is sent to workers? |
| 6 | `vllm/v1/core/kv_cache_manager.py` | how is capacity allocated and released? |
| 7 | `vllm/v1/core/single_type_kv_cache_manager.py` | how does the common cache type work? |
| 8 | `vllm/v1/kv_cache_interface.py` | how are cache layouts described across models? |
| 9 | `vllm/v1/worker/gpu_worker.py` | how does a worker own device initialization and memory? |
| 10 | `vllm/v1/worker/gpu/model_runner.py` | how are model inputs, cache, graphs, and outputs handled? |
| 11 | `vllm/v1/worker/gpu/input_batch.py` | how is mutable batch state represented? |
| 12 | `vllm/v1/worker/gpu/block_table.py` | how does the runner consume logical block mappings? |
| 13 | `vllm/model_executor/layers/attention/` | how is an attention backend selected? |
| 14 | `vllm/v1/sample/` or current sampler path | how are logits transformed and sampled? |
| 15 | `tests/v1/` | which invariants are tested? |

### 5.3 Trace method

Trace one feature at a time:

1. plain greedy request;
2. two concurrent requests with unequal lengths;
3. chunked prefill;
4. prefix-cache hit;
5. preemption or KV-pressure case;
6. CUDA Graph hit and miss.

For each, record the same request state, scheduler output, block-table change, runner input, and output update.

---

## 6. Production Engine Comparison: SGLang

Repository: [sgl-project/sglang](https://github.com/sgl-project/sglang)
Local clone: `../RESOURCES/repos/sglang/`

### 6.1 External design reading

| Priority | Resource | Read for |
|---|---|---|
| Core | [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) | radix-tree prefix reuse and cache-aware scheduling |
| Core | [SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) | scheduler and load-balancer evolution |
| Core | [SGLang paper](../SERVING/SGLANG.pdf) | runtime/programming model and evaluation |
| Branch | [SGLang Runtime performance overview](https://www.lmsys.org/blog/2024-07-25-sglang-llama3/) | engine components and reproducible configurations |

### 6.2 Current source path

| Order | Source path in local clone | Read for |
|---:|---|---|
| 1 | `python/sglang/srt/entrypoints/` | server/runtime entry |
| 2 | `python/sglang/srt/managers/scheduler.py` | scheduler event loop |
| 3 | `python/sglang/srt/managers/scheduler_components/` | decomposed scheduler responsibilities |
| 4 | `python/sglang/srt/mem_cache/radix_cache.py` | prefix-cache policy |
| 5 | `python/sglang/srt/mem_cache/memory_pool.py` | request/token/KV pools |
| 6 | `python/sglang/srt/mem_cache/allocator/` | paged and specialized allocators |
| 7 | `python/sglang/srt/model_executor/` | forward batch, runner, graph path |
| 8 | `python/sglang/srt/layers/attention/attention_registry.py` | backend selection |
| 9 | `python/sglang/srt/layers/attention/` | prefill/decode/MLA backend implementations |
| 10 | benchmark and test directories | workload definitions and invariants |

Compare SGLang to vLLM only after pinning both commits and using the same model, attention backend, precision, request trace, and metric definitions.

---

## 7. TensorRT-LLM Reference Branch

Repository: [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
Local clone: `../RESOURCES/repos/tensorrt-llm/`
Official guide: [TensorRT-LLM Architecture Overview](https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html)

### 7.1 Documentation

| Priority | Resource | Read for |
|---|---|---|
| Core | [Architecture Overview](https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html) | `LLM`, `PyExecutor`, scheduler, KV manager, model engine, sampler |
| Core | [Performance Analysis](https://nvidia.github.io/TensorRT-LLM/performance/perf-analysis.html) | framework metrics and bottleneck workflow |
| Branch | [Executor API](https://nvidia.github.io/TensorRT-LLM/advanced/executor.html) | asynchronous request API and in-flight batching |
| Branch | [KV Cache Connector](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-connector.html) | external/offloaded/disaggregated cache interface |

### 7.2 Current PyTorch-backend source

| Source path in local clone | Read for |
|---|---|
| `tensorrt_llm/_torch/pyexecutor/py_executor.py` | central loop |
| `tensorrt_llm/_torch/pyexecutor/scheduler/` | capacity and microbatch scheduling |
| `tensorrt_llm/_torch/pyexecutor/resource_manager.py` | managed runtime resources |
| `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` | cache integration |
| `tensorrt_llm/_torch/pyexecutor/model_engine.py` | forward execution |
| `tensorrt_llm/_torch/pyexecutor/sampler/` | sampling path |
| `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py` | graph capture/replay |
| `tensorrt_llm/runtime/kv_cache_manager_v2/` | lower-level cache implementation |

Use TensorRT-LLM as a comparative architecture after completing one vLLM or SGLang trace.

---

## 8. Paper-to-Code Map

| Topic | Paper / blog | Code anchor |
|---|---|---|
| iteration-level scheduling | [Orca](../SERVING/ORCA.pdf) | engine scheduler step loop |
| paged KV | [vLLM](../SERVING/VLLM.pdf) | vLLM KV manager and block table |
| chunked prefill | [Sarathi-Serve](../SERVING/SARATHI.pdf) | vLLM/SGLang chunked-prefill scheduler path |
| prefix trees | [SGLang](../SERVING/SGLANG.pdf) and [blog](https://www.lmsys.org/blog/2024-01-17-sglang/) | SGLang/Mini-SGLang radix cache |
| virtual-memory KV | [vAttention](../CACHE/VATTENTION.pdf) | framework virtual-memory allocator branch |
| serving attention | [FlashInfer](../ATTENTION/FLASHINFER.pdf) | selected attention backend |
| disaggregated prefill/decode | [DistServe](../SERVING/DISTSERVE.pdf), [Splitwise](../SERVING/SPLITWISE.pdf) | later distributed layer |

For every paper mechanism, record:

- claimed bottleneck;
- state/data structure introduced;
- scheduler or memory invariant;
- code location;
- workload assumption;
- metric and baseline;
- conditions where the mechanism should not help.

---

## 9. Required Evidence

Produce:

1. a component diagram for one readable engine and one production engine;
2. one end-to-end request trace with exact functions/files;
3. a request-state transition table;
4. a scheduler-input/output table for prefill, decode, mixed, and KV-pressure cases;
5. a KV lifecycle trace from allocation to release;
6. a block-table example for at least two requests;
7. a model-runner input-shape table;
8. an attention-backend selection trace;
9. a CPU/GPU timeline for one engine step;
10. a correctness comparison against Hugging Face greedy generation.

The comparison record must pin:

- engine commit;
- model and tokenizer revision;
- dtype and quantization state;
- attention backend;
- eager/compiled/CUDA Graph mode;
- request trace;
- deterministic decoding configuration;
- hardware and software versions.

---

## 10. Repository Index

| Repository | Role | Starting path | Status |
|---|---|---|---|
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | shortest Python engine path | `nanovllm/engine/` | Link |
| [jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm) | C++/CUDA engine course | README, `src/main.cpp`, `src/kernels.cu` | Link |
| [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) | compact modern serving engine | `docs/`, `python/minisgl/` | Link |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | production engine spine | `vllm/v1/` | Local clone |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | production engine comparison | `python/sglang/srt/` | Local clone |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | optimized runtime comparison | `tensorrt_llm/_torch/pyexecutor/` | Local clone |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | serving attention kernels | docs, Python API, `csrc/` | Local clone |
| [pytorch-labs/gpt-fast](https://github.com/pytorch-labs/gpt-fast) | framework-to-engine bridge | `generate.py`, `model.py`, `tp.py` | Link |

---

## 11. What to Defer

Until the single-node path is traceable, defer:

- multi-node orchestration;
- disaggregated prefill/decode;
- external KV tiers;
- expert-parallel MoE;
- autoscaling and fleet routing;
- every model and attention backend;
- multimodal serving.

---

## Exit Gate

Continue to [08-kv-scheduling-serving.md](08-kv-scheduling-serving.md) when:

- [ ] a request can be traced through API, scheduler, KV manager, runner, attention, sampler, and output;
- [ ] prefill, decode, and mixed batches can be located in scheduler code;
- [ ] a paged KV block is followed from allocation through a block table to attention and release;
- [ ] continuous batching and chunked prefill can be tied to concrete state transitions;
- [ ] eager and CUDA Graph execution paths can be distinguished;
- [ ] one readable engine and one production engine have been compared;
- [ ] deterministic output matches a Hugging Face baseline;
- [ ] performance claims include a pinned commit, controlled workload, and profiler evidence.

The gate is a source-grounded request trace, not familiarity with every engine feature.

---

**Previous:** [`06-decoding-test-time-compute.md`](06-decoding-test-time-compute.md) ·
**Next:** [`08-kv-scheduling-serving.md`](08-kv-scheduling-serving.md)
