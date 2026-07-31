# ARCHITECTURE

Model-component papers selected for inference-systems reasoning. Read the architecture delta and immediately ask how
it changes persistent state, memory traffic, kernel shapes, parallelism and scheduling.

Companion taxonomy: [`../ROADMAP/03-modern-llm-architecture.md`](../ROADMAP/03-modern-llm-architecture.md).

## 1. Modern decoder components

1. [`RMSNORM.pdf`](RMSNORM.pdf) — Root Mean Square Layer Normalization;
2. [`GLU-VARIANTS.pdf`](GLU-VARIANTS.pdf) — GLU variants / SwiGLU;
3. [`ROPE.pdf`](ROPE.pdf) — Rotary Position Embedding;
4. [`ALIBI.pdf`](ALIBI.pdf) — Attention with Linear Biases;
5. [`MQA.pdf`](MQA.pdf) — Multi-Query Attention;
6. [`GQA.pdf`](GQA.pdf) — Grouped-Query Attention.

## 2. Cross-layer attention and KV sharing

Read the external tutorial first:

- [LLMs-from-scratch — Cross-Layer KV Sharing](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/10_kv-sharing).

Then distinguish the primary papers:

1. [`CLA.pdf`](CLA.pdf) — adjacent layers share K/V heads;
2. [`MLKV.pdf`](MLKV.pdf) — multi-layer KV heads extend sharing across depth;
3. [`YOCO.pdf`](YOCO.pdf) — a self-decoder creates global KV reused by a cross-decoder;
4. [`LISA.pdf`](LISA.pdf) — align and approximate attention maps across layers.

Reference implementations:

- `../RESOURCES/repos/pythia-mlkv/`;
- `../RESOURCES/repos/flash-linear-attention/fla/models/yoco/`;
- [microsoft/unilm/YOCO](https://github.com/microsoft/unilm/tree/master/YOCO).

Do not conflate cross-head KV sharing, cross-layer KV sharing, and cross-layer attention-map
sharing.

## 3. Model-family deltas

1. [`LLAMA3.pdf`](LLAMA3.pdf) — Llama 3 model family;
2. [`MISTRAL.pdf`](MISTRAL.pdf) — GQA + sliding-window attention;
3. [`QWEN3.pdf`](QWEN3.pdf) — dense and MoE series, thinking modes;
4. [`DEEPSEEK-V2.pdf`](DEEPSEEK-V2.pdf) — MLA + DeepSeekMoE;
5. [`DEEPSEEK-V3.pdf`](DEEPSEEK-V3.pdf) — aux-loss-free balance + MTP and systems co-design.

## 4. Long-context position extension

1. [`YARN.pdf`](YARN.pdf) — RoPE context extension;
2. [`LONGROPE.pdf`](LONGROPE.pdf) — non-uniform RoPE extension.

Position extension is not a KV-capacity or attention-complexity solution by itself.

## 5. Recurrent, SSM and hybrid alternatives

1. [`MAMBA.pdf`](MAMBA.pdf) — selective state spaces;
2. [`MAMBA2.pdf`](MAMBA2.pdf) — state-space duality;
3. [`RETNET.pdf`](RETNET.pdf) — retention with parallel/recurrent/chunkwise forms;
4. [`RWKV.pdf`](RWKV.pdf) — Transformer-like training with recurrent inference;
5. [`GRIFFIN.pdf`](GRIFFIN.pdf) — local attention + gated linear recurrence.

## 6. Multi-token output

- [`MULTI-TOKEN-PREDICTION.pdf`](MULTI-TOKEN-PREDICTION.pdf) — multi-token prediction as training objective and
  speculative-friendly architecture.

## 7. Dynamic depth and conditional compute

- [`MIXTURE-OF-DEPTHS.pdf`](MIXTURE-OF-DEPTHS.pdf) — route tokens through only a subset of block computations.

This branch changes per-token work, batching regularity and intermediate-state management. It should be studied as
layer-level conditional execution, separately from expert-level MoE routing.

## 8. Diffusion language models

1. [`MDLM.pdf`](MDLM.pdf) — masked diffusion language modeling;
2. [`LLADA.pdf`](LLADA.pdf) — scaled diffusion language model.

Diffusion generation is not left-to-right decode. Its system path repeatedly refines masked positions, exposing a
different balance among parallel token updates, refinement steps, cache reuse, scheduler design and quality/latency
control.

## Required exercise

For each paper, fill one row:

```text
feature
semantic change
persistent state
bytes/token
FLOPs/token
kernel requirement
parallelism effect
prefill effect
decode effect
quality constraint
```
