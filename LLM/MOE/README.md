# MOE

Architecture, routing, expert compute, communication, placement and online serving.

Complete tutorial and research map:
[`../ROADMAP/04-moe-deep-dive.md`](../ROADMAP/04-moe-deep-dive.md).

## 1. Architecture and routing foundations

1. [`GSHARD.pdf`](GSHARD.pdf) — top-2 routing, capacity and automatic sharding;
2. [`SWITCH.pdf`](SWITCH.pdf) — simplified top-1 routing;
3. [`ST-MOE.pdf`](ST-MOE.pdf) — stable/transferable sparse expert training;
4. [`EXPERT-CHOICE.pdf`](EXPERT-CHOICE.pdf) — experts choose fixed-capacity token sets;
5. [`DEEPSEEK-MOE.pdf`](DEEPSEEK-MOE.pdf) — fine-grained + shared experts;
6. [`MIXTRAL.pdf`](MIXTRAL.pdf) — modern top-2 decoder MoE;
7. [`AUX-LOSS-FREE.pdf`](AUX-LOSS-FREE.pdf) — load balance without a strong auxiliary objective;
8. [`PEER.pdf`](PEER.pdf) — extreme fine-grained scaling toward a million tiny experts.

## 2. Runtime, expert compute and communication

1. [`DEEPSPEEDMOE.pdf`](DEEPSPEEDMOE.pdf) — end-to-end MoE inference/training;
2. [`TUTEL.pdf`](TUTEL.pdf) — adaptive MoE runtime;
3. [`MEGABLOCKS.pdf`](MEGABLOCKS.pdf) — dropless block-sparse computation;
4. [`FASTMOE.pdf`](FASTMOE.pdf) — distributed PyTorch MoE foundation;
5. [`MOESYS.pdf`](MOESYS.pdf) — heterogeneous memory and distributed MoE;
6. [`UCCL-EP.pdf`](UCCL-EP.pdf) — portable expert-parallel communication.

## 3. Inference-specific directions

1. [`PRE-GATED-MOE.pdf`](PRE-GATED-MOE.pdf) — predict expert needs early for offload;
2. [`FIDDLER.pdf`](FIDDLER.pdf) — CPU-GPU expert orchestration;
3. [`QMOE.pdf`](QMOE.pdf) — sub-1-bit compression of very large MoE;
4. [`MOESHARD.pdf`](MOESHARD.pdf) — expert sharding for balance.

## Local code order

```text
Transformers / DeepSeek-V3 semantics
→ Megatron router and dispatcher
→ MegaBlocks / DeepGEMM compute
→ DeepEP / Flux communication
→ vLLM / SGLang end-to-end paths
```

Always track total and active parameters, tokens per expert, grouped-GEMM shapes, dispatch/combine bytes,
max-rank tail, expert hotness, prefill/decode split and communication-compute overlap.
