# Local Resources

## What is local

- Paper PDFs: theme folders one level above;
- Course/reference PDFs: [`../COURSE/`](../COURSE/README.md);
- Shallow source clones: `repos/`;
- Exact URLs and pinned commits: [`SOURCES.md`](SOURCES.md);
- Deep-search categorized GitHub map: [`GITHUB-REPO-ATLAS.md`](GITHUB-REPO-ATLAS.md).

## Recommended code-reading order

Do not read vLLM, SGLang, or TensorRT-LLM linearly. Enter each codebase through a concrete question:

1. linked D2L, PyTorch, `micrograd`, and Zero-to-Hero resources — AI/ML prerequisites;
2. reused `LLMs-from-scratch` and Annotated Transformer — decoder and attention mechanics;
3. linked Hugging Face course plus local `transformers` — framework model/cache/generation paths;
4. `cs336-lectures` — resource accounting, GPU systems, parallelism, and inference;
5. `gpu-mode-lectures` — profiling, CUDA/Triton, and kernel optimization;
6. linked Nano-vLLM/Mini-SGLang/Tiny-vLLM — readable engine paths;
7. `transformers` / `deepseek-v3` — model semantics and architecture deltas;
8. `mamba` / `flash-linear-attention` / `kimi-linear` — SSM, recurrence, linear and hybrid architectures;
9. `pythia-mlkv` — cross-head and cross-layer KV-sharing implementation;
10. `kvpress` / `duo-attention` — KV retention/compression and head-adaptive attention;
11. `flash-attention` / `flashinfer` — attention algorithm → serving kernel;
12. `model-optimizer` / `llm-compressor` — deployable weight/activation/KV/attention quantization;
13. `vllm` — scheduler, KV block manager, model runner, quantization and attention/MoE backends;
14. `sglang` — radix/prefix cache, scheduler, runtime, and wide EP;
15. `sarathi-serve` / `distserve` — chunked prefill and PD disaggregation;
16. `lmcache` / `mooncake` — tiered, remote, and disaggregated KV;
17. `megatron-lm` / `tutel` / `deepspeed` — MoE router, dispatcher, and parallelism;
18. `megablocks` / `deepgemm` — sparse/grouped expert compute;
19. `deepep` / `flux` — EP dispatch/combine and communication overlap;
20. `dynamo` / `llm-d` — cluster routing, KV-aware orchestration, and production deployment;
21. `vidur` / `servegen` — simulation and realistic workload generation;
22. `tensorrt-llm` — vendor-optimized runtime and low-precision paths;
23. `speculators` / `eagle` — speculative decoding implementation.

## Reused workspace instead of duplicate clones

These resources already exist under `/home/junyao/code/study`, so this library references them
instead of creating duplicate clones:

| Resource | Local path |
|---|---|
| LLMs from Scratch | `/home/junyao/code/study/tutorials/llm/LLMs-from-scratch` |
| Annotated Transformer | `/home/junyao/code/study/tutorials/transformer/annotated-transformer` |
| CS336 assignment 1 | `/home/junyao/code/study/tutorials/llm/cs336/assignment1-basics` |
| CS336 assignment 2 | `/home/junyao/code/study/tutorials/llm/cs336/assignment2-systems` |
| CS336 assignment 3 | `/home/junyao/code/study/tutorials/llm/cs336/assignment3-scaling` |
| CS336 assignment 4 | `/home/junyao/code/study/tutorials/llm/cs336/assignment4-data` |
| CS336 assignment 5 | `/home/junyao/code/study/tutorials/llm/cs336/assignment5-alignment` |
| Triton | `/home/junyao/code/study/tutorials/gpu-systems/triton` |
| QLoRA | `/home/junyao/code/study/tutorials/llm-quantization/qlora` |

## Storage note

The clone directory is intentionally ignored by the parent paper repository. Some upstreams include
large reference/test assets even in shallow clones; no model checkpoints or Python environments were
downloaded. See `SOURCES.md` for exact HEADs and the current disk-size snapshot.
