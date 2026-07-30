# QUANT

LLM inference quantization. QLoRA is a fine-tuning method and is referenced from the shared study
workspace rather than treated as the inference mainline.

## Reading order

1. [`LLMINT8.pdf`](LLMINT8.pdf) — outliers and mixed precision;
2. [`GPTQ.pdf`](GPTQ.pdf) — weight-only PTQ;
3. [`SMOOTHQUANT.pdf`](SMOOTHQUANT.pdf) — migrate activation difficulty to weights;
4. [`AWQ.pdf`](AWQ.pdf) — activation-aware weight quantization;
5. [`ATOM.pdf`](ATOM.pdf) — serving co-design;
6. [`QUAROT.pdf`](QUAROT.pdf) — rotation and outlier removal.

A complete evaluation needs memory, kernel availability, latency/throughput, calibration cost, and
task quality. A smaller checkpoint alone is not a systems result.
