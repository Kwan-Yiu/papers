# SERVING

Scheduling, memory management, disaggregation, routing, migration, and simulation.

## Core lineage

1. [`ORCA.pdf`](ORCA.pdf) — iteration-level scheduling;
2. [`VLLM.pdf`](VLLM.pdf) — paged KV memory;
3. [`SGLANG.pdf`](SGLANG.pdf) — prefix reuse / RadixAttention;
4. [`SARATHI.pdf`](SARATHI.pdf) — chunked prefill and stall-free scheduling;
5. [`DISTSERVE.pdf`](DISTSERVE.pdf) and [`SPLITWISE.pdf`](SPLITWISE.pdf) — prefill/decode split;
6. [`PREBLE.pdf`](PREBLE.pdf) — distributed prefix-aware scheduling;
7. [`VIDUR.pdf`](VIDUR.pdf) — serving simulation.

## Directional papers

- [`CACHEBLEND.pdf`](CACHEBLEND.pdf) — RAG knowledge-prefix reuse;
- [`LLUMNIX.pdf`](LLUMNIX.pdf) — live migration and dynamic scheduling.

Use TTFT, ITL/TPOT, E2E, P99, throughput, and SLO goodput. Report prompt/output distributions and
arrival process.
