# GitHub Repository Atlas for LLM Inference and AI Systems

> Deep-search snapshot: 2026-07-30
>
> Selection rule: official implementations and widely used infrastructure first.
> Exact local commits: [`SOURCES.md`](SOURCES.md)

## Legend

- **A** — core: read code or run experiments;
- **B** — directional: read when selecting that subfield;
- **C** — reference: know what it is; do not clone/read linearly;
- **L** — shallow clone exists under `RESOURCES/repos/`;
- **R** — reused from `/home/junyao/code/study`;
- **Link** — catalogued, not cloned;
- **Archived** — upstream is archived; use for history/baseline, not as default current stack;
- **Moved** — development moved or the old repo became a mirror.

This atlas deliberately avoids a live star ranking. Popularity does not tell whether a repository is a production
engine, paper artifact, kernel library, teaching implementation or archived wrapper.

---

## 1. Learning and model semantics

| Pri | Local | Repository | Type | Read for |
|---|---|---|---|---|
| A | L | [huggingface/transformers](https://github.com/huggingface/transformers) | model-definition framework | canonical configs and readable model semantics across dense/MoE/SSM families |
| A | R | [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | tutorial/reference | build decoder, tokenizer and training loop from first principles |
| A | R | [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) | tutorial/reference | line-by-line original Transformer |
| A | L | [stanford-cs336/lectures](https://github.com/stanford-cs336/lectures) | course | language modeling, scaling, systems and inference |
| A | L | [gpu-mode/lectures](https://github.com/gpu-mode/lectures) | course | GPU profiling, kernels, communication |
| B | Link | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | minimal training code | compact GPT training baseline |
| B | Link | [karpathy/llm.c](https://github.com/karpathy/llm.c) | C/CUDA teaching system | raw training kernels and runtime mechanics |
| B | Link | [meta-pytorch/gpt-fast](https://github.com/meta-pytorch/gpt-fast) | minimal inference | compact PyTorch-native generation and compilation |
| B | Link | [huggingface/nanotron](https://github.com/huggingface/nanotron) | distributed training reference | minimal 3D-parallel training |
| B | Link | [allenai/OLMo-core](https://github.com/allenai/OLMo-core) | open model building blocks | inspect modern model/training components |
| C | Link | [meta-llama/llama-models](https://github.com/meta-llama/llama-models) | official utilities | Llama model utilities and reference material |
| C | Link | [google-deepmind/gemma](https://github.com/google-deepmind/gemma) | official model reference | Gemma-family implementation/material |

---

## 2. Architecture families

| Pri | Local | Repository | Architecture signal | Systems use |
|---|---|---|---|---|
| A | L | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | MLA, fine-grained/shared-expert MoE, MTP | small reference inference code and configs |
| A | L | [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) | dense + MoE, thinking/non-thinking family | controlled dense/MoE study and official model info |
| A | L | [state-spaces/mamba](https://github.com/state-spaces/mamba) | selective SSM, Mamba/Mamba-2 | fixed-state/scan alternative to KV attention |
| A | L | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | linear attention, SSM and hybrid kernels | emerging architecture implementations and efficient ops |
| B | Link | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | diffusion language model | non-autoregressive iterative generation and cache/scheduler study |
| B | Link | [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) | masked diffusion language model | denoising schedule and parallel token-update reference |
| B | Link | [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) | recurrent LLM | constant-state autoregressive execution |
| B | Link | [mistralai/mistral-inference](https://github.com/mistralai/mistral-inference) | Mistral reference inference | model semantics/history; **Archived** in 2026 snapshot |
| B | Link | [openai/gpt-oss](https://github.com/openai/gpt-oss) | open-weight MoE models | reference implementations and backend interoperability |
| B | Link | [xai-org/grok-1](https://github.com/xai-org/grok-1) | open MoE architecture release | architecture study, not a full serving stack |
| C | Link | [microsoft/unilm](https://github.com/microsoft/unilm) | RetNet and other model research | recurrent/retention reference |

---

## 3. Single-node and model-serving engines

| Pri | Local | Repository | Layer | Best use / warning |
|---|---|---|---|---|
| A | L | [vllm-project/vllm](https://github.com/vllm-project/vllm) | engine + API server | scheduler, paged KV, model runner, distributed inference |
| A | L | [sgl-project/sglang](https://github.com/sgl-project/sglang) | engine + language/runtime | radix cache, structured programs, optimized serving |
| A | L | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA engine/runtime | vendor-optimized precision, kernels, multi-GPU runtime |
| B | Link | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | C/C++ local inference | CPU/consumer hardware, GGUF, edge/server design |
| B | Link | [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) | deployment/serving | compression + TurboMind serving |
| B | Link | [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) | compiled deployment engine | compiler-driven multi-platform inference |
| B | Link | [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) | runtime extension | ONNX Runtime generative inference |
| B | Link | [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | Apple Silicon inference | MLX model loading, generation, quantization |
| B | Link | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | consumer-GPU engine | low-bit local inference |
| C | Link | [triton-inference-server/server](https://github.com/triton-inference-server/server) | general inference server | production server layer around model backends |
| C | Link | [OpenPPL/ppl.llm.kernel.cuda](https://github.com/OpenPPL/ppl.llm.kernel.cuda) | CUDA kernels | PPL LLM kernel reference |
| C | Link | [OpenPPL/ppl.llm.serving](https://github.com/OpenPPL/ppl.llm.serving) | serving | PPL serving reference |
| C | Link | [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) | serving engine | historical production reference; **Archived** in 2026 snapshot |
| C | Link | [ModelTC/LightLLM](https://github.com/ModelTC/LightLLM) | serving engine | alternative implementation and historical comparison |

---

## 4. Distributed serving and orchestration

| Pri | Local | Repository | Scope | Read for |
|---|---|---|---|---|
| A | L | [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) | datacenter inference orchestration | disaggregation, routing, KV transfer/cache, scaling above engines |
| A | L | [llm-d/llm-d](https://github.com/llm-d/llm-d) | Kubernetes distributed inference | KV-aware routing, wide EP, production workflows |
| A | Link | [vllm-project/aibrix](https://github.com/vllm-project/aibrix) | pluggable GenAI infrastructure | routing, autoscaling, cache and research components |
| A | Link | [kubernetes-sigs/gateway-api-inference-extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) | Kubernetes inference routing API | standard inference pools and endpoint selection |
| B | Link | [kserve/kserve](https://github.com/kserve/kserve) | Kubernetes model serving platform | deployment, autoscaling and multi-framework operations |
| B | Link | [ray-project/ray](https://github.com/ray-project/ray) | distributed runtime | replicas, Serve, scheduling and RL/inference integration |
| B | Link | [skypilot-org/skypilot](https://github.com/skypilot-org/skypilot) | cloud orchestration | placement, multi-cloud jobs and service deployment |
| B | L | [microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve) | research engine | chunked prefill and stall-free batching |
| B | L | [LLMServe/DistServe](https://github.com/LLMServe/DistServe) | research system | prefill/decode disaggregation |
| B | L | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) | distributed serving/state | KV-centric disaggregation and transfer |
| C | Link | [triton-inference-server/server](https://github.com/triton-inference-server/server) | server platform | ensembles, metrics, backend integration |

---

## 5. KV cache, state and long context

| Pri | Local | Repository | Focus | Research use |
|---|---|---|---|---|
| A | L | [LMCache/LMCache](https://github.com/LMCache/LMCache) | KV/state layer | sharing, offload, remote/tiered cache, engine integration |
| A | L | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) | distributed KV store/serving | remote KV and transfer paths |
| B | Link | [bytedance/InfiniStore](https://github.com/bytedance/InfiniStore) | distributed KV store | cross-node KV storage |
| B | Link | [alibaba/tair-kvcache](https://github.com/alibaba/tair-kvcache) | KV cache system + simulation | global cache management |
| B | Link | [taco-project/FlexKV](https://github.com/taco-project/FlexKV) | multi-level distributed KV | Dynamo-integrated offload direction |
| B | Link | [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) | KV compression framework | unified evaluation and integration of compression methods |
| B | Link | [Zefan-Cai/KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory) | KV compression methods | algorithm comparison |
| B | Link | [microsoft/MInference](https://github.com/microsoft/MInference) | sparse long-context prefill | dynamic sparse attention |
| B | Link | [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) | attention sinks/streaming | bounded-window streaming behavior |
| B | Link | [SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant) | KV quantization | long-context low-bit KV |
| B | Link | [ByteDance-Seed/ShadowKV](https://github.com/ByteDance-Seed/ShadowKV) | long-context KV optimization | low-rank/key cache direction |
| B | Link | [UChi-JCL/CacheGen](https://github.com/UChi-JCL/CacheGen) | KV compression/streaming | remote KV transfer compression |
| C | Link | [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua) | prompt compression | input-token reduction; not the same as runtime KV management |

---

## 6. Attention, GEMM, compiler and kernel infrastructure

| Pri | Local | Repository | Focus | Notes |
|---|---|---|---|---|
| A | L | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | exact attention kernels | prefill IO-aware attention baseline |
| A | L | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | LLM serving kernels | paged/decode attention, sampling, GEMM/MoE paths |
| A | R | [triton-lang/triton](https://github.com/triton-lang/triton) | GPU DSL/compiler | implement and autotune custom kernels |
| A | L | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | CUDA templates/CuTe DSL | GEMM/grouped-GEMM and architecture-specific kernels |
| A | L | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | LLM tensor-core kernels | low-precision GEMM and fused MoE work |
| B | Link | [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | optimized Transformer blocks | composable attention/building blocks |
| B | Link | [NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend) | cuDNN frontend/open kernels | graph and kernel APIs |
| B | Link | [ROCm/aiter](https://github.com/ROCm/aiter) | AMD tensor engine | ROCm attention/MoE/operator paths |
| B | Link | [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) | AMD compute libraries | current home for CK-style development |
| B | L | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | linear attention/SSM ops | emerging architecture kernels |
| B | Link | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | kernel DSL | alternative high-performance kernel authoring |
| B | Link | [pytorch/FBGEMM](https://github.com/pytorch/FBGEMM) | GEMM/quant/MoE primitives | production low-precision/grouped operations |
| B | Link | [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | Transformer low precision | FP8 and distributed model components |
| C | Link | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | CUDA examples | API and optimization exercises |
| C | Link | [microsoft/BitBLAS](https://github.com/microsoft/BitBLAS) | mixed-precision matmul | useful paper/code history; **Archived** |
| C | Link | [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) | AMD kernel templates | **Moved/deprecated mirror**; use `rocm-libraries` |

---

## 7. Quantization and compression

| Pri | Local | Repository | Focus | Status/use |
|---|---|---|---|---|
| A | L | [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | deployment optimization | quantization, pruning, distillation, speculation |
| A | L | [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) | Transformers→vLLM compression | current deployment-oriented quant workflows |
| A | R | [artidoro/qlora](https://github.com/artidoro/qlora) | low-bit fine-tuning | QLoRA reference, not a serving engine |
| B | Link | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | k-bit PyTorch ops | training/loading and low-bit components |
| B | Link | [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) | GPTQ paper code | algorithm reference |
| B | Link | [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | AWQ | algorithm + kernels/integration history |
| B | Link | [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant) | SmoothQuant | activation smoothing reference |
| B | Link | [pytorch/ao](https://github.com/pytorch/ao) | native quantization/sparsity | current PyTorch quant stack |
| B | Link | [huggingface/optimum-quanto](https://github.com/huggingface/optimum-quanto) | PyTorch quant backend | application-level quantization |
| B | Link | [spcl/QuaRot](https://github.com/spcl/QuaRot) | rotation-based 4-bit | end-to-end research implementation |
| B | Link | [efeslab/Atom](https://github.com/efeslab/Atom) | low-bit serving | systems-oriented quantization |
| B | Link | [intel/auto-round](https://github.com/intel/auto-round) | low-bit quantization | multi-hardware and engine integrations |
| C | Link | [AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | GPTQ wrapper | **Archived**; prefer current engine/compressor paths |
| C | Link | [casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | AWQ wrapper | **Archived**; prefer maintained integrations |

---

## 8. MoE architecture, kernels and communication

| Pri | Local | Repository | Layer | Read for |
|---|---|---|---|---|
| A | L | [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | EP communication | low-latency/high-throughput dispatch and combine |
| A | L | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | expert compute | grouped/fused low-precision MoE kernels |
| A | L | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | full framework | router, token dispatcher, grouped GEMM and parallel groups |
| A | L | [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | full framework | distributed/MoE training and inference history |
| A | L | [microsoft/Tutel](https://github.com/microsoft/Tutel) | MoE runtime | routing, dispatch, modern model/precision support |
| A | L | [databricks/megablocks](https://github.com/databricks/megablocks) | sparse expert compute | dropless block-sparse MoE |
| A | L | [bytedance/flux](https://github.com/bytedance/flux) | communication overlap | TP/EP communication-compute overlap |
| B | L | [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe) | production engine | fused MoE backends and expert parallel execution |
| B | L | [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/moe) | production engine | MoE runner, token dispatch and wide EP |
| B | L | [huggingface/transformers](https://github.com/huggingface/transformers) | semantics + expert backends | readable model definitions and evolving expert interface |
| B | Link | [pytorch/FBGEMM](https://github.com/pytorch/FBGEMM) | primitives | grouped GEMM/quantized expert building blocks |
| B | Link | [laekov/fastmoe](https://github.com/laekov/fastmoe) | research framework | FastMoE historical distributed PyTorch implementation |
| B | Link | [ranggihwang/Pregated_MoE](https://github.com/ranggihwang/Pregated_MoE) | paper artifact | expert prefetch/offload via pre-gating |
| B | Link | [MoE-Inf/awesome-moe-inference](https://github.com/MoE-Inf/awesome-moe-inference) | curated bibliography | MoE inference paper discovery, verify papers independently |
| C | Link | [deepspeedai/DeepSpeed-MII](https://github.com/deepspeedai/DeepSpeed-MII) | inference wrapper | DeepSpeed-powered serving reference |

---

## 9. Speculative and alternative decoding

| Pri | Local | Repository | Method | Read for |
|---|---|---|---|---|
| A | L | [vllm-project/speculators](https://github.com/vllm-project/speculators) | unified speculation library | building/evaluating speculators for vLLM |
| A | L | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | EAGLE 1/2/3 | feature-level drafting |
| A | Link | [hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench) | benchmark | workload/acceptance comparison |
| B | Link | [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) | multiple heads | Medusa training/inference artifact |
| B | Link | [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) | lookahead | draft-model-free decoding |
| B | Link | [Infini-AI-Lab/TriForce](https://github.com/Infini-AI-Lab/TriForce) | hierarchical long-context speculation | sparse KV/draft hierarchy |
| B | Link | [Infini-AI-Lab/Sequoia](https://github.com/Infini-AI-Lab/Sequoia) | tree-based speculation | robust tree construction/verification |
| B | Link | [feifeibear/LLMSpeculativeSampling](https://github.com/feifeibear/LLMSpeculativeSampling) | minimal speculative sampling | compact correctness reference |
| B | Link | [meta-pytorch/gpt-fast](https://github.com/meta-pytorch/gpt-fast) | small PyTorch inference | simple compilation/spec paths |
| C | L | [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | optimization framework | production-oriented speculative/model optimization integration |

Thirty local papers are indexed in [`../SPEC/README.md`](../SPEC/README.md).

---

## 10. Communication and transfer

| Pri | Local | Repository | Primitive | Use |
|---|---|---|---|---|
| A | Link | [NVIDIA/nccl](https://github.com/NVIDIA/nccl) | GPU collectives | production communication layer |
| A | L | [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests) | collective benchmark | topology/bandwidth/latency baseline |
| A | L | [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | EP dispatch/combine | MoE-specific low-latency communication |
| A | Link | [ai-dynamo/nixl](https://github.com/ai-dynamo/nixl) | inference transfer library | KV/state transfer abstraction |
| B | Link | [NVIDIA/nvshmem](https://github.com/NVIDIA/nvshmem) | GPU one-sided communication | device-initiated transfer and overlap |
| B | Link | [openucx/ucx](https://github.com/openucx/ucx) | communication framework | RDMA/transport layer |
| B | Link | [pytorch/gloo](https://github.com/pytorch/gloo) | collective library | CPU/multi-machine communication reference |
| B | L | [bytedance/flux](https://github.com/bytedance/flux) | overlapped TP/EP | fused communication-compute |

---

## 11. Benchmarking, traces, simulation and profiling

| Pri | Local | Repository | Type | Use / warning |
|---|---|---|---|---|
| A | L | [microsoft/vidur](https://github.com/microsoft/vidur) | trace-driven simulator | config/workload exploration, then calibrate on hardware |
| A | L | [alibaba/ServeGen](https://github.com/alibaba/ServeGen) | workload generation | realistic serving workload synthesis/traces |
| A | Link | [casys-kaist/LLMServingSim](https://github.com/casys-kaist/LLMServingSim) | simulator | heterogeneous/disaggregated serving simulation |
| A | Link | [llm-d/llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark) | cluster benchmark workflow | deployment→run→collect→teardown |
| A | Link | [mlcommons/inference](https://github.com/mlcommons/inference) | standardized benchmark | MLPerf reference implementations |
| A | L | [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/main/benchmarks) | engine benchmarks | latency/throughput/serving harnesses |
| A | L | [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/main/benchmark) | engine benchmarks | serving and model-specific benchmarks |
| B | Link | [triton-inference-server/perf_analyzer](https://github.com/triton-inference-server/perf_analyzer) | performance analyzer | GenAI-Perf and Triton measurement tooling |
| B | Link | [pytorch/kineto](https://github.com/pytorch/kineto) | profiler backend | CPU+GPU traces and counters |
| B | Link | [THUDM/LongBench](https://github.com/THUDM/LongBench) | quality benchmark | long-context task quality |
| B | Link | [gkamradt/needle-in-a-haystack](https://github.com/gkamradt/needle-in-a-haystack) | long-context probe | simple retrieval sanity test, not sufficient alone |
| B | Link | [hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench) | speculation benchmark | acceptance/speed comparison |
| C | Link | [ray-project/llmperf](https://github.com/ray-project/llmperf) | serving benchmark | **Archived**; useful historical harness |

---

## 12. Distributed training/post-training adjacent to inference

These are not prerequisites for the inference core, but matter for rollout engines, architecture co-design and
training-serving convergence.

| Pri | Local | Repository | Use |
|---|---|---|---|
| B | L | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | large-scale parallelism and MoE |
| B | L | [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | distributed optimization and inference |
| B | Link | [verl-project/verl](https://github.com/verl-project/verl) | RL/post-training rollout architecture |
| B | Link | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | scalable agentic RL and vLLM/Ray integration |
| B | Link | [huggingface/trl](https://github.com/huggingface/trl) | post-training algorithms |
| B | Link | [hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI) | distributed model scaling |
| C | Link | [pytorch/pytorch](https://github.com/pytorch/pytorch) | compiler, distributed and runtime source of truth |

---

## 13. How to use this atlas

### If your topic is KV/state

Read/run:

```text
vLLM or SGLang
→ LMCache
→ Mooncake
→ llm-d or Dynamo
→ Vidur/ServeGen
```

### If your topic is MoE

Read/run:

```text
Transformers/DeepSeek-V3 semantics
→ Megatron router/dispatcher
→ MegaBlocks/DeepGEMM expert compute
→ DeepEP/Flux communication
→ vLLM/SGLang end-to-end serving
```

### If your topic is kernels

Read/run:

```text
GPU Mode + Triton
→ FlashAttention/FlashInfer
→ CUTLASS/DeepGEMM
→ engine integration
→ online workload
```

### If your topic is distributed serving

Read/run:

```text
vLLM/SGLang execution engine
→ LMCache/Mooncake/NIXL state transfer
→ DistServe/Sarathi paper baselines
→ Dynamo/llm-d/AIBrix orchestration
→ cluster benchmark + failure tests
```

### What not to do

- do not read a giant repo from top to bottom;
- do not compare README throughput numbers across different hardware/workloads;
- do not adopt an archived wrapper as the default 2026 stack;
- do not equate paper artifact with production engine;
- do not clone model weights when source/config is enough;
- do not use a simulator without calibration;
- do not use GitHub popularity as research evidence.
