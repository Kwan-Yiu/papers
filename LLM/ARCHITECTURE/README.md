# ARCHITECTURE

Model-component papers selected for inference-systems reasoning. Read the architecture delta and immediately ask how
it changes persistent state, memory traffic, kernel shapes, parallelism and scheduling.

Companion taxonomy: [`../ROADMAP/03-architecture-taxonomy.md`](../ROADMAP/03-architecture-taxonomy.md).

## 1. Modern decoder components

1. [`RMSNORM.pdf`](RMSNORM.pdf) — Root Mean Square Layer Normalization;
2. [`GLU-VARIANTS.pdf`](GLU-VARIANTS.pdf) — GLU variants / SwiGLU;
3. [`ROPE.pdf`](ROPE.pdf) — Rotary Position Embedding;
4. [`ALIBI.pdf`](ALIBI.pdf) — Attention with Linear Biases;
5. [`MQA.pdf`](MQA.pdf) — Multi-Query Attention;
6. [`GQA.pdf`](GQA.pdf) — Grouped-Query Attention.

## 2. Model-family deltas

1. [`LLAMA3.pdf`](LLAMA3.pdf) — Llama 3 model family;
2. [`MISTRAL.pdf`](MISTRAL.pdf) — GQA + sliding-window attention;
3. [`QWEN3.pdf`](QWEN3.pdf) — dense and MoE series, thinking modes;
4. [`DEEPSEEK-V2.pdf`](DEEPSEEK-V2.pdf) — MLA + DeepSeekMoE;
5. [`DEEPSEEK-V3.pdf`](DEEPSEEK-V3.pdf) — aux-loss-free balance + MTP and systems co-design.

## 3. Long-context position extension

1. [`YARN.pdf`](YARN.pdf) — RoPE context extension;
2. [`LONGROPE.pdf`](LONGROPE.pdf) — non-uniform RoPE extension.

Position extension is not a KV-capacity or attention-complexity solution by itself.

## 4. Recurrent, SSM and hybrid alternatives

1. [`MAMBA.pdf`](MAMBA.pdf) — selective state spaces;
2. [`MAMBA2.pdf`](MAMBA2.pdf) — state-space duality;
3. [`RETNET.pdf`](RETNET.pdf) — retention with parallel/recurrent/chunkwise forms;
4. [`RWKV.pdf`](RWKV.pdf) — Transformer-like training with recurrent inference;
5. [`GRIFFIN.pdf`](GRIFFIN.pdf) — local attention + gated linear recurrence.

## 5. Multi-token output

- [`MULTI-TOKEN-PREDICTION.pdf`](MULTI-TOKEN-PREDICTION.pdf) — multi-token prediction as training objective and
  speculative-friendly architecture.

## 6. Dynamic depth and conditional compute

- [`MIXTURE-OF-DEPTHS.pdf`](MIXTURE-OF-DEPTHS.pdf) — route tokens through only a subset of block computations.

This branch changes per-token work, batching regularity and intermediate-state management. It should be studied as
layer-level conditional execution, separately from expert-level MoE routing.

## 7. Diffusion language models

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
