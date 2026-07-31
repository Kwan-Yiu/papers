# CACHE

External sources for KV capacity, bandwidth, retention, representation, physical layout, tiering,
and disaggregation.

Companion map:
[`../ROADMAP/04-single-node-inference-optimization.md`](../ROADMAP/04-single-node-inference-optimization.md).

## 1. Beginner bridge

1. [LLMs-from-scratch — KV Cache](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/03_kv-cache);
2. [Hugging Face — KV cache strategies](https://huggingface.co/docs/transformers/kv_cache);
3. [How To Scale Your Model — Inference](https://jax-ml.github.io/scaling-book/inference/).

## 2. Retain fewer tokens

1. [`STREAMINGLLM.pdf`](STREAMINGLLM.pdf) — attention sinks plus a recent window;
2. [`H2O.pdf`](H2O.pdf) — heavy-hitter eviction;
3. [`SNAPKV.pdf`](SNAPKV.pdf) — observation-window prompt selection;
4. [`PYRAMIDKV.pdf`](PYRAMIDKV.pdf) — layer-adaptive retention budgets.

Unified implementation/evaluation:

- [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress);
- local clone: `../RESOURCES/repos/kvpress/`.

## 3. Represent KV with fewer bits

1. [`KIVI.pdf`](KIVI.pdf) — asymmetric 2-bit KV quantization;
2. [`KVQUANT.pdf`](KVQUANT.pdf) — per-channel keys, pre-RoPE keys, non-uniform formats and kernels;
3. [LLM Compressor KV example](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache);
4. [vLLM quantized KV cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html).

## 4. Change physical layout or location

- [`VATTENTION.pdf`](VATTENTION.pdf) — virtual-memory-backed KV;
- [vLLM / PagedAttention](../SERVING/VLLM.pdf) — block allocation;
- [LMCache](https://github.com/LMCache/LMCache) — local/remote tiering and reuse;
- [`MOONCAKE.pdf`](MOONCAKE.pdf) — disaggregated KV transfer/store.

Always distinguish logical retention, physical allocation, cache hit, bytes loaded, transfer
latency, recomputation cost, and quality loss.
