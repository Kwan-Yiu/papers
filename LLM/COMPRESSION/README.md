# COMPRESSION

External sources for model pruning, weight sparsity, structural reduction, and distillation as
inference optimizations.

Companion map:
[`../ROADMAP/04-single-node-inference-optimization.md`](../ROADMAP/04-single-node-inference-optimization.md).

## 1. Concepts and deployment documentation

1. [NVIDIA Model Optimizer overview](https://nvidia.github.io/Model-Optimizer/getting_started/1_overview.html);
2. [Model Optimizer — Sparsity](https://nvidia.github.io/Model-Optimizer/guides/6_sparsity.html);
3. [Model Optimizer — Pruning](https://nvidia.github.io/Model-Optimizer/guides/3_pruning.html);
4. [Model Optimizer — Distillation](https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html).

## 2. Paper progression

1. [`SPARSEGPT.pdf`](SPARSEGPT.pdf) — one-shot unstructured and semi-structured pruning;
2. [`WANDA.pdf`](WANDA.pdf) — weight-and-activation pruning without retraining.

Primary repositories:

- [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt);
- [locuslab/wanda](https://github.com/locuslab/wanda);
- [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer).

## 3. Local production-oriented path

```text
../RESOURCES/repos/model-optimizer/examples/llm_sparsity/
../RESOURCES/repos/model-optimizer/examples/llm_sparsity/weight_sparsity/
../RESOURCES/repos/model-optimizer/examples/llm_sparsity/attention_sparsity/
../RESOURCES/repos/model-optimizer/examples/pruning/minitron/
../RESOURCES/repos/model-optimizer/examples/pruning/puzzletron/
../RESOURCES/repos/model-optimizer/examples/llm_distill/
```

Keep these branches distinct:

- unstructured weight sparsity;
- hardware-supported N:M sparsity;
- structured width, FFN, head, GQA-group, state-dimension, and depth pruning;
- teacher-student distillation;
- sparse checkpoint storage versus sparse compute;
- model-size reduction versus measured latency/throughput improvement.

A zero weight is not automatically skipped by a dense kernel. Report the deployed sparse
representation, kernel, hardware support, quality-recovery procedure, and end-to-end workload
effect.
