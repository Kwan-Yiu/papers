# FOUNDATION

Architecture papers needed to understand inference workloads.

## Reading order

1. [`ATTENTION.pdf`](ATTENTION.pdf) — original Transformer; focus on attention shapes and causal execution;
2. [`LLAMA2.pdf`](LLAMA2.pdf) — representative decoder-only architecture;
3. [`DEEPSEEKR1.pdf`](DEEPSEEKR1.pdf) — reasoning model and long-output workload context.

DeepSeek-V3 is already stored in [`../SPEC/DEEPSEEKV3.pdf`](../SPEC/DEEPSEEKV3.pdf) and is not
duplicated here.

Do not read these as model-history papers. Extract tensor shapes, active parameters, KV representation,
precision, parallelism, and prompt/output behavior.
