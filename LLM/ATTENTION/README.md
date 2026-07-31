# ATTENTION

External sources for exact IO-aware attention, sparse/local attention, and linear/recurrent/hybrid
alternatives.

Companion map:
[`../ROADMAP/04-single-node-inference-optimization.md`](../ROADMAP/04-single-node-inference-optimization.md).

## 1. Exact softmax attention kernels

1. [`FLASHATTN.pdf`](FLASHATTN.pdf) — IO-aware exact attention;
2. [`FLASHATTN2.pdf`](FLASHATTN2.pdf) — work partitioning and occupancy;
3. [`FLASHATTN3.pdf`](FLASHATTN3.pdf) — Hopper-specific execution;
4. [`FLASHINFER.pdf`](FLASHINFER.pdf) — ragged/paged serving kernels.

Code:

- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention);
- [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer);
- [GPU Mode Lecture 12](https://github.com/gpu-mode/lectures/tree/main/lecture_012).

## 2. Local, sparse, and mixed-span attention

1. [`MINFERENCE.pdf`](MINFERENCE.pdf) — dynamic sparse long-context prefill;
2. [`DUOATTENTION.pdf`](DUOATTENTION.pdf) — retrieval heads plus streaming heads;
3. [`NSA.pdf`](NSA.pdf) — model-native hierarchical sparse attention.

External tutorials/code:

- [LLMs-from-scratch — Sliding Window Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/06_swa);
- [LLMs-from-scratch — DeepSeek Sparse Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa);
- [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention);
- [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest).

## 3. Linear, recurrent, and hybrid attention

Use the readable
[LLMs-from-scratch Gated DeltaNet tutorial](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet)
before the primary papers:

1. [`LINEAR-TRANSFORMER.pdf`](LINEAR-TRANSFORMER.pdf) — feature-map/recurrent formulation;
2. [`BASED.pdf`](BASED.pdf) — local plus linear attention and recall-throughput trade-off;
3. [`GLA.pdf`](GLA.pdf) — gated linear attention and hardware-efficient chunks;
4. [`DELTANET.pdf`](DELTANET.pdf) — delta-rule recurrent memory;
5. [`KIMI-LINEAR.pdf`](KIMI-LINEAR.pdf) — KDA plus MLA hybrid.

Primary code:

- [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention);
- [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear);
- local paths under `../RESOURCES/repos/flash-linear-attention/fla/models/` and `fla/ops/`.

FlashAttention is an exact softmax-attention kernel optimization. Linear attention changes the model
formulation and persistent state. Sparse attention changes the set of attended positions. Keep
these branches separate when comparing memory, quality, and latency.
