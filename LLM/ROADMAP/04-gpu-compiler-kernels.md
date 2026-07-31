# GPU, Compiler, and Kernels — Curated Reading Map

> **Role:** hardware/software foundation for LLM inference performance
> **Target:** move from PyTorch operators to traces, generated kernels, GPU memory traffic, and compiler decisions
> **Format:** external English courses, official documentation, blogs, assignments, papers, and repository source paths
> **Not included:** an original CUDA/Triton tutorial or a calendar

---

## Dependency Map

```text
roofline and GPU model
        |
        +--> profiling and benchmark discipline
        |
        +--> CUDA execution and memory
        |        |
        |        +--> Triton kernels
        |        +--> CUTLASS / CuTe
        |        +--> attention kernels
        |
        +--> PyTorch dispatcher / compile stack
                 |
                 +--> graph capture and breaks
                 +--> Inductor / Triton code generation
                 +--> CUDA Graphs
```

Resource labels:

- **Core** — required;
- **Branch** — follow when the topic is directly relevant;
- **Reference** — use on demand;
- **Local PDF** — already stored in this repository;
- **Link** — do not clone unless executing or modifying the source.

---

## Coverage Checklist

### Hardware and cost

- [ ] SM, warp, thread block, grid, tensor core, register, shared memory, L1/L2, HBM;
- [ ] coalescing, bank conflicts, divergence, occupancy, synchronization, and launch overhead;
- [ ] FLOPs, bytes moved, arithmetic intensity, memory capacity, latency, and bandwidth;
- [ ] compute-bound, memory-bound, launch-bound, synchronization-bound, and communication-bound;
- [ ] prefill versus decode GEMM and attention shapes.

### Profiling

- [ ] CPU wall time versus asynchronous GPU time;
- [ ] warmup, synchronization, event timing, repeated samples, and shape control;
- [ ] PyTorch profiler, Nsight Systems, Nsight Compute, and generated-code inspection;
- [ ] kernel timeline, launch gaps, utilization, memory traffic, occupancy, and achieved throughput.

### Kernels and compilers

- [ ] vector add, reduction, softmax, normalization, GEMM, and attention;
- [ ] tiling, fusion, persistent kernels, grouped GEMM, and autotuning;
- [ ] CUDA, Triton, CUTLASS, CuTe, cuBLAS, and attention backends;
- [ ] dispatcher, Dynamo, FX, AOTAutograd, Inductor, Triton, guards, and graph breaks;
- [ ] CUDA Graph capture constraints and replay.

---

## 1. Performance Model Before Kernel Code

### 1.1 Primary reading spine

| Priority | Resource | Format | Exact reading target | Extract |
|---|---|---|---|---|
| Core | [How To Scale Your Model — All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) | systems book | full chapter | compute, bandwidth, memory-capacity lower bounds |
| Core | [How To Scale Your Model — How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) | systems book | GPU hierarchy, networking, collectives, LLM rooflines | chip-to-cluster cost model |
| Core | [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) | systems blog | Parts 1–3 and the case study | eager overhead, fusion, memory traffic, compilation |
| Core | [Efficiently Scaling Transformer Inference](../PERF/TRANSFORMERINFER.pdf) | Local PDF | analytical model and prefill/decode comparison | inference-specific arithmetic intensity |
| Reference | [Roofline: An Insightful Visual Performance Model](../PERF/ROOFLINE.pdf) | Local PDF | model and limitations | canonical roofline formulation |

### 1.2 Optional deeper courses

| Priority | Resource | Format | Read / inspect |
|---|---|---|---|
| Branch | [mryab/efficient-dl-systems](https://github.com/mryab/efficient-dl-systems) | university course repo | GPU/CUDA, benchmarking, profiling, compiler, and inference lectures |
| Branch | [MLSysBook Volume 2](../COURSE/MLSYS-VOL2.pdf) | Local PDF | hardware acceleration and efficient inference chapters |
| Reference | [MLSysBook online](https://mlsysbook.ai/) | open systems book | use the current web table of contents to match the Local PDF |

Required evidence:

- one operator cost sheet with shape, dtype, FLOPs, bytes, arithmetic intensity, and predicted bound;
- separate rows for prefill and single-token decode;
- hardware assumptions with source links.

Do not optimize a kernel before producing this cost sheet.

---

## 2. CUDA Execution and Memory

### 2.1 Official NVIDIA path

| Priority | Resource | Format | Exact reading target |
|---|---|---|---|
| Core | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) | official guide | programming model; execution model; memory hierarchy; asynchronous execution; streams; graphs |
| Core | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | official guide | performance metrics; memory optimization; execution configuration; instruction optimization |
| Local | [CUDA Programming Guide](../COURSE/CUDA-PROGRAMMING-GUIDE.pdf) | Local PDF | offline copy; use web guide for current behavior |
| Local | [CUDA Best Practices](../COURSE/CUDA-BEST-PRACTICES.pdf) | Local PDF | offline copy; use web guide for current behavior |
| Reference | [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html) | official API reference | streams, events, memory, graphs, error handling |

### 2.2 GPU Mode course path

Repository: [gpu-mode/lectures](https://github.com/gpu-mode/lectures)
Videos: [GPU MODE YouTube channel](https://www.youtube.com/@GPUMODE)

| Order | Repository path | Topic |
|---:|---|---|
| 1 | `lecture_001/` | profiling and integrating CUDA kernels with PyTorch |
| 2 | `lecture_003/pmpp.ipynb` | CUDA start for Python programmers |
| 3 | `lecture_004/` | compute and memory architecture |
| 4 | `lecture_005/matmul_l5.ipynb` | CUDA optimization progression |
| 5 | `lecture_008/` | coalescing, tiling, divergence, occupancy, coarsening |
| 6 | `lecture_009/` | reductions, determinism, accuracy |
| 7 | `lecture_012/` | FlashAttention implementation |
| 8 | `lecture_014/A_Practitioners_Guide_to_Triton.ipynb` | Triton programming |
| 9 | `lecture_017/` | NCCL and collectives |
| 10 | `lecture_018/` | fused kernels |
| Branch | `lecture_023/`, `lecture_037/`, `lecture_041/`, `lecture_057/` | tensor cores, SASS, CUDA docs, CuTe |

The lecture repository changes over time. Use its root README as the source of truth for current lecture numbers and media links.

### 2.3 CUDA sample source map

Repository: [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)

| Topic | Source path |
|---|---|
| device and runtime basics | `Samples/1_Utilities/deviceQuery/`, `Samples/0_Introduction/` |
| streams and overlap | search under `Samples/0_Introduction/` for asynchronous copy and streams |
| reductions | `Samples/2_Concepts_and_Techniques/reduction/` |
| matrix multiplication | `Samples/0_Introduction/matrixMul/` |
| CUDA Graphs | graph samples under `Samples/3_CUDA_Features/` |
| peer-to-peer | P2P samples under `Samples/0_Introduction/` and `Samples/5_Domain_Specific/` |

Use the repository tree for the current exact directory names; samples can move between releases.

---

## 3. Benchmarking and Profiling

### 3.1 Timing and profiler tutorials

| Priority | Resource | Format | Exact reading target | Output |
|---|---|---|---|---|
| Core | [PyTorch Benchmark Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) | official recipe | `torch.utils.benchmark` workflow | reliable CPU/operator microbenchmarks |
| Core | [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | official recipe | activities, schedules, traces, memory | operator and kernel timeline |
| Core | [PyTorch CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) | official note | asynchronous execution, streams, events, allocator | correct GPU timing |
| Core | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | official guide | CUDA trace, NVTX, CLI capture, statistics | end-to-end CPU/GPU timeline |
| Core | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | official guide | metrics, sections, roofline, source correlation | per-kernel diagnosis |
| Branch | [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | official recipe | inference-relevant sections | checklist of framework-level issues |

### 3.2 Required benchmark record

Every performance record must include:

- hardware and software versions;
- exact tensor shapes, dtypes, strides, and device;
- warmup policy and measured repetitions;
- synchronization/timing mechanism;
- median and tail or dispersion, not only the best sample;
- correctness tolerance and reference implementation;
- profiler evidence when the claimed cause is kernel-level;
- generated code or compiler logs when the claimed cause is compiler-level.

Required profiler views:

1. PyTorch operator table;
2. CPU/GPU timeline;
3. one Nsight Compute kernel report;
4. one source or generated-code correlation.

---

## 4. Triton Kernel Path

### 4.1 Official tutorial sequence

Use [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) in the published order:

| Order | Tutorial | Required extraction |
|---:|---|---|
| 1 | [Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) | program IDs, offsets, masks |
| 2 | [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | fusion, row mapping, occupancy |
| 3 | [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | blocking, pointer arithmetic, ordering, autotuning |
| 4 | Low-Memory Dropout | seed/counter and masks |
| 5 | Layer Normalization | reductions and backward path |
| 6 | Fused Attention | tiled attention and online softmax |
| Branch | Group GEMM | irregular/grouped workloads and MoE |
| Branch | Persistent Matmul | persistent scheduling |
| Branch | Block-Scaled Matmul | low-precision scaling formats |

Repository: [triton-lang/triton](https://github.com/triton-lang/triton)

Source path:

1. `python/tutorials/`;
2. `python/triton/language/` for the language surface;
3. `python/triton/compiler/` for compilation entry points;
4. `lib/` for compiler dialects and lowering;
5. `third_party/` only for backend-specific work.

### 4.2 Supplementary Triton sources

| Priority | Resource | Exact target |
|---|---|---|
| Core | [GPU Mode Lecture 14](https://github.com/gpu-mode/lectures/tree/main/lecture_014) | `A_Practitioners_Guide_to_Triton.ipynb` |
| Branch | [Triton Puzzles](https://github.com/srush/Triton-Puzzles) | puzzles in order; compare solutions after attempting each |
| Branch | [stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems) | assignment handout attention/kernel optimization section and tests |
| Reference | [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream) | Triton compiler and kernel-resource indices |

Required kernel evidence:

- PyTorch reference output;
- correctness across several shapes, including non-multiples of block size;
- benchmark against the relevant PyTorch/vendor baseline;
- profiler evidence for the limiting resource;
- explanation tied to the roofline estimate.

---

## 5. CUTLASS, CuTe, and Vendor Libraries

| Priority | Resource | Format | Read / inspect |
|---|---|---|---|
| Core | [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/) | official docs | quickstart, GEMM model, CuTe layouts, examples |
| Core | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | source repo | `README.md`, `media/docs/`, `examples/`, `include/cutlass/`, `include/cute/` |
| Branch | [CuTe DSL documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) | official docs | layout algebra and Python DSL when targeting recent NVIDIA GPUs |
| Branch | [GPU Mode CUTLASS lectures](https://github.com/gpu-mode/lectures) | lectures | Lectures 15, 36, 57, and later CuTe material |
| Reference | [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html) | official API | GEMM APIs, data types, algorithms, reproducibility |

Read this branch when:

- Triton lacks the required tensor-core feature;
- a serving framework calls CUTLASS/CuTe kernels;
- grouped GEMM or quantized GEMM needs vendor-specific control;
- generated SASS or architecture-specific scheduling is part of the research question.

---

## 6. Attention Kernel Reading

| Order | Resource | Format | Exact target |
|---:|---|---|---|
| 1 | [FlashAttention](../ATTENTION/FLASHATTN.pdf) | Local PDF | IO-awareness, tiling, online softmax, backward |
| 2 | [FlashAttention-2](../ATTENTION/FLASHATTN2.pdf) | Local PDF | work partitioning and occupancy changes |
| 3 | [FlashAttention-3](../ATTENTION/FLASHATTN3.pdf) | Local PDF | Hopper-specific asynchrony and low precision |
| 4 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | source repo | README → Python interface → `csrc/flash_attn/` |
| 5 | [GPU Mode Lecture 12](https://github.com/gpu-mode/lectures/tree/main/lecture_012) | code + slides | `flash_attention.cu`, notebook, register-spilling example |
| Branch | [FlashInfer paper](../ATTENTION/FLASHINFER.pdf) | Local PDF | serving-oriented attention abstraction |
| Branch | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | source repo | docs, Python API, `csrc/`, JIT and serving integrations |

When tracing a serving engine, record which attention backend is selected for:

- prefill;
- decode;
- paged KV;
- sliding-window or sparse attention;
- MHA/GQA/MLA;
- speculative verification.

---

## 7. PyTorch Compiler and Runtime

### 7.1 Official documentation path

| Priority | Resource | Exact reading target | Extract |
|---|---|---|---|
| Core | [Introduction to `torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | basic use, speedup, graph breaks, troubleshooting | public compilation workflow |
| Core | [`torch.compile` programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html) | guards, graph breaks, dynamic shapes, recompilation | correctness and specialization constraints |
| Core | [Profiling `torch.compile`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html) | compilation time, graph breaks, generated kernels | evidence collection |
| Core | [PyTorch compiler troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html) | logs, minification, recompilation | debugging workflow |
| Branch | [End-to-end `torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html) | model-level compilation | whole-model boundary |
| Branch | [CUDA Graphs in PyTorch](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) | capture/replay model and constraints | launch-overhead reduction |

### 7.2 Repository path

Repository: [pytorch/pytorch](https://github.com/pytorch/pytorch)

| Layer | Starting path |
|---|---|
| Python API | `torch/__init__.py`, `torch/_dynamo/` |
| graph capture | `torch/_dynamo/` |
| FX graph representation | `torch/fx/` |
| AOTAutograd | `torch/_functorch/aot_autograd.py` and adjacent modules |
| Inductor | `torch/_inductor/` |
| generated Triton kernels | compiler debug output, then `torch/_inductor/codegen/` |
| dispatcher/operator registration | `aten/`, `torch/library.py`, C++ dispatcher docs |
| CUDA Graph integration | search `CUDAGraph` in Inductor/runtime paths |

Do not read the PyTorch repository linearly. Start from one compiled model, enable logs, preserve the FX/Inductor artifacts, and follow only the path responsible for a concrete graph or kernel.

### 7.3 Required compiler trace

For one decoder block:

1. capture eager operator trace;
2. compile it with stable shapes;
3. save graph-break and recompilation logs;
4. save FX/Inductor graphs;
5. inspect at least one generated Triton kernel;
6. compare eager and compiled correctness;
7. compare latency after excluding compilation;
8. identify the remaining unfused kernels and why they remain separate.

---

## 8. Repository Index

| Repository | Role | Exact starting path | Status |
|---|---|---|---|
| [gpu-mode/lectures](https://github.com/gpu-mode/lectures) | GPU course code and slides | root README; lectures 1, 3–5, 8–9, 12, 14, 17–18 | Link |
| [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream) | curated GPU resource index | topic directories and README | Link |
| [mryab/efficient-dl-systems](https://github.com/mryab/efficient-dl-systems) | full efficient-DL systems course | current course outline | Link |
| [jax-ml/scaling-book](https://github.com/jax-ml/scaling-book) | roofline and scaling source | roofline, GPU, inference, profiling chapters | Link |
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | CUDA reference samples | `Samples/` | Link |
| [triton-lang/triton](https://github.com/triton-lang/triton) | language/compiler/tutorials | `python/tutorials/`, `python/triton/`, `lib/` | Link |
| [srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles) | small Triton exercises | numbered puzzles | Link |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | GEMM and CuTe infrastructure | `media/docs/`, `examples/`, `include/` | Link |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | attention kernels | Python interface, `csrc/flash_attn/` | Link |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | serving attention kernels | docs, Python API, `csrc/` | Link |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | framework/compiler/runtime | `torch/_dynamo/`, `torch/_inductor/`, `aten/` | Link |
| [stanford-cs336/assignment2-systems](https://github.com/stanford-cs336/assignment2-systems) | systems assignment and tests | assignment PDF and `tests/` | Link |

---

## 9. What to Defer

Defer unless demanded by a concrete bottleneck:

- hand-written SASS;
- every CUDA memory space and instruction;
- full MLIR/LLVM compiler implementation;
- backward-only training kernels;
- exotic accelerators unrelated to the target hardware;
- architecture-specific tensor-core instructions before profiler evidence points there.

---

## Exit Gate

Continue to [05-single-node-inference-engine.md](05-single-node-inference-engine.md) when:

- [ ] prefill and decode operator shapes can be placed on a roofline;
- [ ] GPU timing is correct under asynchronous execution;
- [ ] a PyTorch profiler trace, Nsight Systems trace, and Nsight Compute report can be interpreted;
- [ ] one CUDA or Triton kernel is correct across boundary shapes and benchmarked against a baseline;
- [ ] tiling, fusion, occupancy, and memory coalescing can be connected to profiler evidence;
- [ ] a `torch.compile` path can be traced through graph capture, Inductor, and generated Triton;
- [ ] graph breaks, recompilation, and CUDA Graph constraints can be identified;
- [ ] an attention backend can be located from a serving framework call site to its kernel repository.

The gate is evidence-based performance reasoning, not completion of every GPU resource.
