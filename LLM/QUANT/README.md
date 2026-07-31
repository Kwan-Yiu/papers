# QUANT

Curated external path for LLM inference quantization. QLoRA is a low-bit fine-tuning method and is
not treated as the inference mainline.

Companion map:
[`../ROADMAP/04-single-node-inference-optimization.md`](../ROADMAP/04-single-node-inference-optimization.md).

## 1. Concepts and deployment documentation

1. [Hugging Face — Quantization concepts](https://huggingface.co/docs/transformers/quantization/concept_guide);
2. [Hugging Face — Selecting a quantization method](https://huggingface.co/docs/transformers/main/quantization/selecting);
3. [TensorRT-LLM — Numerical Precision](https://nvidia.github.io/TensorRT-LLM/reference/precision.html);
4. [vLLM — Quantization](https://docs.vllm.ai/en/stable/features/quantization/index.html);
5. [LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/);
6. [TensorRT-LLM PyTorch backend quantization](https://nvidia.github.io/TensorRT-LLM/torch/features/quantization.html).

## 2. Paper progression

1. [`LLMINT8.pdf`](LLMINT8.pdf) — outliers and mixed precision;
2. [`ZEROQUANT.pdf`](ZEROQUANT.pdf) — hardware-aware W+A PTQ and backend co-design;
3. [`GPTQ.pdf`](GPTQ.pdf) — weight-only PTQ;
4. [`SMOOTHQUANT.pdf`](SMOOTHQUANT.pdf) — move activation difficulty to weights;
5. [`AWQ.pdf`](AWQ.pdf) — activation-aware weight quantization;
6. [`ATOM.pdf`](ATOM.pdf) — serving co-design;
7. [`QUAROT.pdf`](QUAROT.pdf) → [`SPINQUANT.pdf`](SPINQUANT.pdf) — random and learned rotations;
8. [`QLLM-EVAL.pdf`](QLLM-EVAL.pdf) — evaluation across weights, activations and KV.

KV-specific continuation:

- [`../CACHE/KIVI.pdf`](../CACHE/KIVI.pdf);
- [`../CACHE/KVQUANT.pdf`](../CACHE/KVQUANT.pdf).

## 3. Production source path

Read:

```text
../RESOURCES/repos/llm-compressor/examples/
../RESOURCES/repos/model-optimizer/examples/llm_ptq/
../RESOURCES/repos/model-optimizer/examples/llm_qat/
../RESOURCES/repos/vllm/vllm/model_executor/layers/quantization/
../RESOURCES/repos/tensorrt-llm/tensorrt_llm/_torch/
```

A complete comparison covers PTQ/QAT; weight-only, W+A, KV, attention and MoE quantization;
integer/floating/microscale formats; scale granularity; calibration and outliers; packed layout and
dequantization; kernel/hardware support; memory, latency, throughput and task quality.

A smaller checkpoint alone is not a systems result.
