# Sources and Pinned Repositories

Downloaded and verified on 2026-07-31. "Core" means required by the prerequisite-driven competency
path; "directional" means read when selecting that research direction; "frontier" is a current
problem snapshot.

## Foundation

| File | Title | Source | Use |
|---|---|---|---|
| [`FOUNDATION/ATTENTION.pdf`](../FOUNDATION/ATTENTION.pdf) | Attention Is All You Need | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) | Core |
| [`FOUNDATION/LLAMA2.pdf`](../FOUNDATION/LLAMA2.pdf) | Llama 2: Open Foundation and Fine-Tuned Chat Models | [arXiv 2307.09288](https://arxiv.org/abs/2307.09288) | Core architecture |
| [`FOUNDATION/DEEPSEEKR1.pdf`](../FOUNDATION/DEEPSEEKR1.pdf) | DeepSeek-R1 | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) | Reasoning workload |

## Training and Post-Training

| File | Title | Source | Use |
|---|---|---|---|
| [`TRAINING/SCALING-LAWS.pdf`](../TRAINING/SCALING-LAWS.pdf) | Scaling Laws for Neural Language Models | [arXiv 2001.08361](https://arxiv.org/abs/2001.08361) | Scaling foundation |
| [`TRAINING/DEDUP.pdf`](../TRAINING/DEDUP.pdf) | Deduplicating Training Data Makes Language Models Better | [arXiv 2107.06499](https://arxiv.org/abs/2107.06499) | Data quality |
| [`TRAINING/CHINCHILLA.pdf`](../TRAINING/CHINCHILLA.pdf) | Training Compute-Optimal Large Language Models | [arXiv 2203.15556](https://arxiv.org/abs/2203.15556) | Compute-optimal scaling |
| [`TRAINING/DATA-CONSTRAINED.pdf`](../TRAINING/DATA-CONSTRAINED.pdf) | Scaling Data-Constrained Language Models | [arXiv 2305.16264](https://arxiv.org/abs/2305.16264) | Finite-data regimes |
| [`TRAINING/DOREMI.pdf`](../TRAINING/DOREMI.pdf) | DoReMi | [arXiv 2305.10429](https://arxiv.org/abs/2305.10429) | Data-mixture optimization |
| [`TRAINING/DATACOMP-LM.pdf`](../TRAINING/DATACOMP-LM.pdf) | DataComp-LM | [arXiv 2406.11794](https://arxiv.org/abs/2406.11794) | Data benchmark |
| [`TRAINING/FINEWEB.pdf`](../TRAINING/FINEWEB.pdf) | The FineWeb Datasets | [arXiv 2406.17557](https://arxiv.org/abs/2406.17557) | Web-data pipeline |
| [`TRAINING/GPIPE.pdf`](../TRAINING/GPIPE.pdf) | GPipe | [arXiv 1811.06965](https://arxiv.org/abs/1811.06965) | Pipeline parallelism |
| [`TRAINING/PIPEDREAM.pdf`](../TRAINING/PIPEDREAM.pdf) | PipeDream | [arXiv 1806.03377](https://arxiv.org/abs/1806.03377) | Pipeline scheduling |
| [`TRAINING/ZERO.pdf`](../TRAINING/ZERO.pdf) | ZeRO | [arXiv 1910.02054](https://arxiv.org/abs/1910.02054) | Sharded training state |
| [`TRAINING/ZERO-OFFLOAD.pdf`](../TRAINING/ZERO-OFFLOAD.pdf) | ZeRO-Offload | [arXiv 2101.06840](https://arxiv.org/abs/2101.06840) | Optimizer offload |
| [`TRAINING/MEGATRON-3D.pdf`](../TRAINING/MEGATRON-3D.pdf) | Efficient Large-Scale LM Training with Megatron-LM | [arXiv 2104.04473](https://arxiv.org/abs/2104.04473) | 3D parallelism |
| [`TRAINING/FLAN.pdf`](../TRAINING/FLAN.pdf) | Finetuned Language Models Are Zero-Shot Learners | [arXiv 2109.01652](https://arxiv.org/abs/2109.01652) | Instruction tuning |
| [`TRAINING/LORA.pdf`](../TRAINING/LORA.pdf) | LoRA | [arXiv 2106.09685](https://arxiv.org/abs/2106.09685) | Parameter-efficient adaptation |
| [`TRAINING/SELF-INSTRUCT.pdf`](../TRAINING/SELF-INSTRUCT.pdf) | Self-Instruct | [arXiv 2212.10560](https://arxiv.org/abs/2212.10560) | Synthetic instruction data |
| [`TRAINING/LIMA.pdf`](../TRAINING/LIMA.pdf) | LIMA | [arXiv 2305.11206](https://arxiv.org/abs/2305.11206) | Supervised alignment |
| [`TRAINING/INSTRUCTGPT.pdf`](../TRAINING/INSTRUCTGPT.pdf) | Training Language Models to Follow Instructions with Human Feedback | [arXiv 2203.02155](https://arxiv.org/abs/2203.02155) | RLHF foundation |
| [`TRAINING/CONSTITUTIONAL-AI.pdf`](../TRAINING/CONSTITUTIONAL-AI.pdf) | Constitutional AI | [arXiv 2212.08073](https://arxiv.org/abs/2212.08073) | AI feedback |
| [`TRAINING/DPO.pdf`](../TRAINING/DPO.pdf) | Direct Preference Optimization | [arXiv 2305.18290](https://arxiv.org/abs/2305.18290) | Offline preference optimization |
| [`TRAINING/VERIFY-STEP.pdf`](../TRAINING/VERIFY-STEP.pdf) | Let's Verify Step by Step | [arXiv 2305.20050](https://arxiv.org/abs/2305.20050) | Process reward models |
| [`TRAINING/KTO.pdf`](../TRAINING/KTO.pdf) | KTO | [arXiv 2402.01306](https://arxiv.org/abs/2402.01306) | Preference optimization |
| [`TRAINING/DEEPSEEKMATH.pdf`](../TRAINING/DEEPSEEKMATH.pdf) | DeepSeekMath / GRPO | [arXiv 2402.03300](https://arxiv.org/abs/2402.03300) | Reasoning RL |
| [`TRAINING/ORPO.pdf`](../TRAINING/ORPO.pdf) | ORPO | [arXiv 2403.07691](https://arxiv.org/abs/2403.07691) | Preference optimization |
| [`TRAINING/SIMPO.pdf`](../TRAINING/SIMPO.pdf) | SimPO | [arXiv 2405.14734](https://arxiv.org/abs/2405.14734) | Preference optimization |
| [`TRAINING/DEEPSEEK-R1.pdf`](../TRAINING/DEEPSEEK-R1.pdf) | DeepSeek-R1 | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) | Reasoning RL and workload |

Topic grouping and tutorial/repository links are in
[`TRAINING/README.md`](../TRAINING/README.md).

## Performance foundations

| File | Title | Source | Use |
|---|---|---|---|
| [`PERF/ROOFLINE.pdf`](../PERF/ROOFLINE.pdf) | Roofline: An Insightful Visual Performance Model | [UC Berkeley tech report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) | Core |
| [`PERF/TRANSFORMERINFER.pdf`](../PERF/TRANSFORMERINFER.pdf) | Efficiently Scaling Transformer Inference | [arXiv 2211.05102](https://arxiv.org/abs/2211.05102) | Core |
| [`PERF/DEEPSPEEDINFER.pdf`](../PERF/DEEPSPEEDINFER.pdf) | DeepSpeed Inference | [arXiv 2207.00032](https://arxiv.org/abs/2207.00032) | Reference |
| [`PERF/FLEXGEN.pdf`](../PERF/FLEXGEN.pdf) | FlexGen | [arXiv 2303.06865](https://arxiv.org/abs/2303.06865) | Offload direction |

## Architecture

| File | Title | Source | Use |
|---|---|---|---|
| [`ARCHITECTURE/MQA.pdf`](../ARCHITECTURE/MQA.pdf) | Fast Transformer Decoding: One Write-Head is All You Need | [arXiv 1911.02150](https://arxiv.org/abs/1911.02150) | KV/head-sharing foundation |
| [`ARCHITECTURE/GQA.pdf`](../ARCHITECTURE/GQA.pdf) | GQA: Training Generalized Multi-Query Transformer Models | [arXiv 2305.13245](https://arxiv.org/abs/2305.13245) | Core architecture |
| [`ARCHITECTURE/ROPE.pdf`](../ARCHITECTURE/ROPE.pdf) | RoFormer: Rotary Position Embedding | [arXiv 2104.09864](https://arxiv.org/abs/2104.09864) | Position foundation |
| [`ARCHITECTURE/RMSNORM.pdf`](../ARCHITECTURE/RMSNORM.pdf) | Root Mean Square Layer Normalization | [arXiv 1910.07467](https://arxiv.org/abs/1910.07467) | Component foundation |
| [`ARCHITECTURE/GLU-VARIANTS.pdf`](../ARCHITECTURE/GLU-VARIANTS.pdf) | GLU Variants Improve Transformer | [arXiv 2002.05202](https://arxiv.org/abs/2002.05202) | FFN foundation |
| [`ARCHITECTURE/ALIBI.pdf`](../ARCHITECTURE/ALIBI.pdf) | Train Short, Test Long: Attention with Linear Biases | [arXiv 2108.12409](https://arxiv.org/abs/2108.12409) | Position alternative |
| [`ARCHITECTURE/MISTRAL.pdf`](../ARCHITECTURE/MISTRAL.pdf) | Mistral 7B | [arXiv 2310.06825](https://arxiv.org/abs/2310.06825) | GQA + sliding window |
| [`ARCHITECTURE/LLAMA3.pdf`](../ARCHITECTURE/LLAMA3.pdf) | The Llama 3 Herd of Models | [arXiv 2407.21783](https://arxiv.org/abs/2407.21783) | Modern dense family |
| [`ARCHITECTURE/QWEN3.pdf`](../ARCHITECTURE/QWEN3.pdf) | Qwen3 Technical Report | [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) | Dense/MoE + reasoning modes |
| [`ARCHITECTURE/DEEPSEEK-V2.pdf`](../ARCHITECTURE/DEEPSEEK-V2.pdf) | DeepSeek-V2 | [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) | MLA + DeepSeekMoE |
| [`ARCHITECTURE/DEEPSEEK-V3.pdf`](../ARCHITECTURE/DEEPSEEK-V3.pdf) | DeepSeek-V3 Technical Report | [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) | MLA/MoE/MTP co-design |
| [`ARCHITECTURE/CLA.pdf`](../ARCHITECTURE/CLA.pdf) | Reducing Transformer Key-Value Cache Size with Cross-Layer Attention | [arXiv 2405.12981](https://arxiv.org/abs/2405.12981) | Cross-layer KV sharing |
| [`ARCHITECTURE/MLKV.pdf`](../ARCHITECTURE/MLKV.pdf) | Multi-Layer Key-Value Heads | [arXiv 2406.09297](https://arxiv.org/abs/2406.09297) | Cross-head/cross-layer KV sharing |
| [`ARCHITECTURE/YOCO.pdf`](../ARCHITECTURE/YOCO.pdf) | You Only Cache Once | [arXiv 2405.05254](https://arxiv.org/abs/2405.05254) | Decoder-decoder shared KV |
| [`ARCHITECTURE/LISA.pdf`](../ARCHITECTURE/LISA.pdf) | Cross-layer Attention Sharing for Large Language Models | [arXiv 2408.01890](https://arxiv.org/abs/2408.01890) | Cross-layer attention-map sharing |
| [`ARCHITECTURE/YARN.pdf`](../ARCHITECTURE/YARN.pdf) | YaRN | [arXiv 2309.00071](https://arxiv.org/abs/2309.00071) | Context extension |
| [`ARCHITECTURE/LONGROPE.pdf`](../ARCHITECTURE/LONGROPE.pdf) | LongRoPE | [arXiv 2402.13753](https://arxiv.org/abs/2402.13753) | Context extension |
| [`ARCHITECTURE/MAMBA.pdf`](../ARCHITECTURE/MAMBA.pdf) | Mamba | [arXiv 2312.00752](https://arxiv.org/abs/2312.00752) | Selective SSM |
| [`ARCHITECTURE/MAMBA2.pdf`](../ARCHITECTURE/MAMBA2.pdf) | Transformers are SSMs / Mamba-2 | [arXiv 2405.21060](https://arxiv.org/abs/2405.21060) | State-space duality |
| [`ARCHITECTURE/RETNET.pdf`](../ARCHITECTURE/RETNET.pdf) | Retentive Network | [arXiv 2307.08621](https://arxiv.org/abs/2307.08621) | Retention/recurrent alternative |
| [`ARCHITECTURE/RWKV.pdf`](../ARCHITECTURE/RWKV.pdf) | RWKV | [arXiv 2305.13048](https://arxiv.org/abs/2305.13048) | Recurrent LLM |
| [`ARCHITECTURE/GRIFFIN.pdf`](../ARCHITECTURE/GRIFFIN.pdf) | Griffin | [arXiv 2402.19427](https://arxiv.org/abs/2402.19427) | Local-attention/recurrent hybrid |
| [`ARCHITECTURE/MULTI-TOKEN-PREDICTION.pdf`](../ARCHITECTURE/MULTI-TOKEN-PREDICTION.pdf) | Better & Faster LLMs via Multi-token Prediction | [arXiv 2404.19737](https://arxiv.org/abs/2404.19737) | MTP/speculation |
| [`ARCHITECTURE/MIXTURE-OF-DEPTHS.pdf`](../ARCHITECTURE/MIXTURE-OF-DEPTHS.pdf) | Mixture-of-Depths | [arXiv 2404.02258](https://arxiv.org/abs/2404.02258) | Dynamic depth / conditional compute |
| [`ARCHITECTURE/MDLM.pdf`](../ARCHITECTURE/MDLM.pdf) | Simple and Effective Masked Diffusion Language Models | [arXiv 2406.07524](https://arxiv.org/abs/2406.07524) | Masked diffusion language modeling |
| [`ARCHITECTURE/LLADA.pdf`](../ARCHITECTURE/LLADA.pdf) | Large Language Diffusion Models | [arXiv 2502.09992](https://arxiv.org/abs/2502.09992) | Large-scale diffusion language model |

## Attention and kernels

| File | Title | Source | Use |
|---|---|---|---|
| [`ATTENTION/FLASHATTN.pdf`](../ATTENTION/FLASHATTN.pdf) | FlashAttention | [arXiv 2205.14135](https://arxiv.org/abs/2205.14135) | Core |
| [`ATTENTION/FLASHATTN2.pdf`](../ATTENTION/FLASHATTN2.pdf) | FlashAttention-2 | [arXiv 2307.08691](https://arxiv.org/abs/2307.08691) | Core |
| [`ATTENTION/FLASHATTN3.pdf`](../ATTENTION/FLASHATTN3.pdf) | FlashAttention-3 | [arXiv 2407.08608](https://arxiv.org/abs/2407.08608) | Kernel direction |
| [`ATTENTION/FLASHINFER.pdf`](../ATTENTION/FLASHINFER.pdf) | FlashInfer | [arXiv 2501.01005](https://arxiv.org/abs/2501.01005) | Core serving kernels |
| [`ATTENTION/MINFERENCE.pdf`](../ATTENTION/MINFERENCE.pdf) | MInference 1.0 | [arXiv 2407.02490](https://arxiv.org/abs/2407.02490) | Long-context direction |
| [`ATTENTION/DUOATTENTION.pdf`](../ATTENTION/DUOATTENTION.pdf) | DuoAttention | [arXiv 2410.10819](https://arxiv.org/abs/2410.10819) | Retrieval/streaming head split |
| [`ATTENTION/NSA.pdf`](../ATTENTION/NSA.pdf) | Native Sparse Attention | [arXiv 2502.11089](https://arxiv.org/abs/2502.11089) | Long-context direction |
| [`ATTENTION/LINEAR-TRANSFORMER.pdf`](../ATTENTION/LINEAR-TRANSFORMER.pdf) | Transformers are RNNs | [arXiv 2006.16236](https://arxiv.org/abs/2006.16236) | Linear-attention foundation |
| [`ATTENTION/BASED.pdf`](../ATTENTION/BASED.pdf) | Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff | [arXiv 2402.18668](https://arxiv.org/abs/2402.18668) | Local+linear hybrid |
| [`ATTENTION/GLA.pdf`](../ATTENTION/GLA.pdf) | Gated Linear Attention Transformers | [arXiv 2312.06635](https://arxiv.org/abs/2312.06635) | Gated linear attention |
| [`ATTENTION/DELTANET.pdf`](../ATTENTION/DELTANET.pdf) | Parallelizing Linear Transformers with the Delta Rule | [arXiv 2406.06484](https://arxiv.org/abs/2406.06484) | Delta-rule recurrent memory |
| [`ATTENTION/KIMI-LINEAR.pdf`](../ATTENTION/KIMI-LINEAR.pdf) | Kimi Linear | [arXiv 2510.26692](https://arxiv.org/abs/2510.26692) | KDA/MLA hybrid |

## KV cache

| File | Title | Source | Use |
|---|---|---|---|
| [`CACHE/H2O.pdf`](../CACHE/H2O.pdf) | H2O: Heavy-Hitter Oracle | [arXiv 2306.14048](https://arxiv.org/abs/2306.14048) | Eviction direction |
| [`CACHE/STREAMINGLLM.pdf`](../CACHE/STREAMINGLLM.pdf) | Efficient Streaming Language Models with Attention Sinks | [arXiv 2309.17453](https://arxiv.org/abs/2309.17453) | Streaming direction |
| [`CACHE/KIVI.pdf`](../CACHE/KIVI.pdf) | KIVI | [arXiv 2402.02750](https://arxiv.org/abs/2402.02750) | Core KV quantization |
| [`CACHE/KVQUANT.pdf`](../CACHE/KVQUANT.pdf) | KVQuant | [arXiv 2401.18079](https://arxiv.org/abs/2401.18079) | Low-bit KV and kernels |
| [`CACHE/SNAPKV.pdf`](../CACHE/SNAPKV.pdf) | SnapKV | [arXiv 2404.14469](https://arxiv.org/abs/2404.14469) | Selection direction |
| [`CACHE/PYRAMIDKV.pdf`](../CACHE/PYRAMIDKV.pdf) | PyramidKV | [arXiv 2406.02069](https://arxiv.org/abs/2406.02069) | Layer-adaptive KV budgets |
| [`CACHE/VATTENTION.pdf`](../CACHE/VATTENTION.pdf) | vAttention | [arXiv 2405.04437](https://arxiv.org/abs/2405.04437) | Core virtual memory |
| [`CACHE/MOONCAKE.pdf`](../CACHE/MOONCAKE.pdf) | Mooncake | [arXiv 2407.00079](https://arxiv.org/abs/2407.00079) | Core disaggregated KV |

## Serving systems

| File | Title | Source | Use |
|---|---|---|---|
| [`SERVING/ORCA.pdf`](../SERVING/ORCA.pdf) | Orca | [USENIX OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | Core |
| [`SERVING/VLLM.pdf`](../SERVING/VLLM.pdf) | PagedAttention / vLLM | [arXiv 2309.06180](https://arxiv.org/abs/2309.06180) | Core |
| [`SERVING/SGLANG.pdf`](../SERVING/SGLANG.pdf) | SGLang | [arXiv 2312.07104](https://arxiv.org/abs/2312.07104) | Core |
| [`SERVING/SARATHI.pdf`](../SERVING/SARATHI.pdf) | Sarathi-Serve | [arXiv 2403.02310](https://arxiv.org/abs/2403.02310) | Core |
| [`SERVING/DISTSERVE.pdf`](../SERVING/DISTSERVE.pdf) | DistServe | [arXiv 2401.09670](https://arxiv.org/abs/2401.09670) | Core |
| [`SERVING/SPLITWISE.pdf`](../SERVING/SPLITWISE.pdf) | Splitwise | [arXiv 2311.18677](https://arxiv.org/abs/2311.18677) | Disaggregation |
| [`SERVING/CACHEBLEND.pdf`](../SERVING/CACHEBLEND.pdf) | CacheBlend | [arXiv 2405.16444](https://arxiv.org/abs/2405.16444) | RAG/cache direction |
| [`SERVING/LLUMNIX.pdf`](../SERVING/LLUMNIX.pdf) | Llumnix | [arXiv 2406.03243](https://arxiv.org/abs/2406.03243) | Migration/scheduling |
| [`SERVING/PREBLE.pdf`](../SERVING/PREBLE.pdf) | Preble | [arXiv 2407.00023](https://arxiv.org/abs/2407.00023) | Core prefix routing |
| [`SERVING/VIDUR.pdf`](../SERVING/VIDUR.pdf) | Vidur | [arXiv 2405.05465](https://arxiv.org/abs/2405.05465) | Core simulation |

## Quantization

| File | Title | Source | Use |
|---|---|---|---|
| [`QUANT/LLMINT8.pdf`](../QUANT/LLMINT8.pdf) | LLM.int8() | [arXiv 2208.07339](https://arxiv.org/abs/2208.07339) | Foundation |
| [`QUANT/ZEROQUANT.pdf`](../QUANT/ZEROQUANT.pdf) | ZeroQuant | [arXiv 2206.01861](https://arxiv.org/abs/2206.01861) | W+A PTQ and system co-design |
| [`QUANT/GPTQ.pdf`](../QUANT/GPTQ.pdf) | GPTQ | [arXiv 2210.17323](https://arxiv.org/abs/2210.17323) | Foundation |
| [`QUANT/SMOOTHQUANT.pdf`](../QUANT/SMOOTHQUANT.pdf) | SmoothQuant | [arXiv 2211.10438](https://arxiv.org/abs/2211.10438) | Core |
| [`QUANT/AWQ.pdf`](../QUANT/AWQ.pdf) | AWQ | [arXiv 2306.00978](https://arxiv.org/abs/2306.00978) | Core |
| [`QUANT/ATOM.pdf`](../QUANT/ATOM.pdf) | Atom | [arXiv 2310.19102](https://arxiv.org/abs/2310.19102) | Serving direction |
| [`QUANT/QUAROT.pdf`](../QUANT/QUAROT.pdf) | QuaRot | [arXiv 2404.00456](https://arxiv.org/abs/2404.00456) | Rotation direction |
| [`QUANT/SPINQUANT.pdf`](../QUANT/SPINQUANT.pdf) | SpinQuant | [arXiv 2405.16406](https://arxiv.org/abs/2405.16406) | Learned-rotation direction |
| [`QUANT/QLLM-EVAL.pdf`](../QUANT/QLLM-EVAL.pdf) | Evaluating Quantized Large Language Models | [arXiv 2402.18158](https://arxiv.org/abs/2402.18158) | Broad W/A/KV evaluation |
| [`QUANT/QLORA.pdf`](../QUANT/QLORA.pdf) | QLoRA | [arXiv 2305.14314](https://arxiv.org/abs/2305.14314) | Low-bit adaptation boundary |

## Pruning and model compression

| File | Title | Source | Use |
|---|---|---|---|
| [`COMPRESSION/SPARSEGPT.pdf`](../COMPRESSION/SPARSEGPT.pdf) | SparseGPT | [arXiv 2301.00774](https://arxiv.org/abs/2301.00774) | One-shot weight sparsity |
| [`COMPRESSION/WANDA.pdf`](../COMPRESSION/WANDA.pdf) | A Simple and Effective Pruning Approach for Large Language Models | [arXiv 2306.11695](https://arxiv.org/abs/2306.11695) | Activation-aware pruning |

## Parallelism and MoE

| File | Title | Source | Use |
|---|---|---|---|
| [`PARALLEL/MEGATRON.pdf`](../PARALLEL/MEGATRON.pdf) | Megatron-LM | [arXiv 1909.08053](https://arxiv.org/abs/1909.08053) | Core parallelism |
| [`MOE/SWITCH.pdf`](../MOE/SWITCH.pdf) | Switch Transformers | [arXiv 2101.03961](https://arxiv.org/abs/2101.03961) | MoE foundation |
| [`MOE/DEEPSPEEDMOE.pdf`](../MOE/DEEPSPEEDMOE.pdf) | DeepSpeed-MoE | [arXiv 2201.05596](https://arxiv.org/abs/2201.05596) | Core MoE system |
| [`MOE/TUTEL.pdf`](../MOE/TUTEL.pdf) | Tutel | [arXiv 2206.03382](https://arxiv.org/abs/2206.03382) | MoE runtime |
| [`MOE/MEGABLOCKS.pdf`](../MOE/MEGABLOCKS.pdf) | MegaBlocks | [arXiv 2211.15841](https://arxiv.org/abs/2211.15841) | Sparse kernel |
| [`MOE/GSHARD.pdf`](../MOE/GSHARD.pdf) | GShard | [arXiv 2006.16668](https://arxiv.org/abs/2006.16668) | Routing/sharding foundation |
| [`MOE/ST-MOE.pdf`](../MOE/ST-MOE.pdf) | ST-MoE | [arXiv 2202.08906](https://arxiv.org/abs/2202.08906) | Routing stability |
| [`MOE/EXPERT-CHOICE.pdf`](../MOE/EXPERT-CHOICE.pdf) | MoE with Expert Choice Routing | [arXiv 2202.09368](https://arxiv.org/abs/2202.09368) | Routing alternative |
| [`MOE/DEEPSEEK-MOE.pdf`](../MOE/DEEPSEEK-MOE.pdf) | DeepSeekMoE | [arXiv 2401.06066](https://arxiv.org/abs/2401.06066) | Shared/fine-grained experts |
| [`MOE/MIXTRAL.pdf`](../MOE/MIXTRAL.pdf) | Mixtral of Experts | [arXiv 2401.04088](https://arxiv.org/abs/2401.04088) | Modern decoder MoE |
| [`MOE/AUX-LOSS-FREE.pdf`](../MOE/AUX-LOSS-FREE.pdf) | Auxiliary-Loss-Free Load Balancing | [arXiv 2408.15664](https://arxiv.org/abs/2408.15664) | Load balancing |
| [`MOE/FASTMOE.pdf`](../MOE/FASTMOE.pdf) | FastMoE | [arXiv 2103.13262](https://arxiv.org/abs/2103.13262) | Distributed runtime foundation |
| [`MOE/MOESYS.pdf`](../MOE/MOESYS.pdf) | MoESys | [arXiv 2205.10034](https://arxiv.org/abs/2205.10034) | Heterogeneous memory/system |
| [`MOE/PRE-GATED-MOE.pdf`](../MOE/PRE-GATED-MOE.pdf) | Pre-gated MoE | [arXiv 2308.12066](https://arxiv.org/abs/2308.12066) | Expert offload/prefetch |
| [`MOE/MOESHARD.pdf`](../MOE/MOESHARD.pdf) | Accelerating MoE Inference with Expert Sharding | [arXiv 2503.08467](https://arxiv.org/abs/2503.08467) | Balance/sharding |
| [`MOE/QMOE.pdf`](../MOE/QMOE.pdf) | QMoE | [arXiv 2310.16795](https://arxiv.org/abs/2310.16795) | Extreme MoE compression |
| [`MOE/FIDDLER.pdf`](../MOE/FIDDLER.pdf) | Fiddler | [arXiv 2402.07033](https://arxiv.org/abs/2402.07033) | CPU-GPU orchestration |
| [`MOE/UCCL-EP.pdf`](../MOE/UCCL-EP.pdf) | UCCL-EP | [arXiv 2512.19849](https://arxiv.org/abs/2512.19849) | Portable EP communication |
| [`MOE/PEER.pdf`](../MOE/PEER.pdf) | Mixture of A Million Experts | [arXiv 2407.04153](https://arxiv.org/abs/2407.04153) | Extreme-scale fine-grained/retrieval experts |

## 2025–2026 frontier snapshot

| File | Title | Source |
|---|---|---|
| [`FRONTIER/SERVEGEN.pdf`](../FRONTIER/SERVEGEN.pdf) | ServeGen | [USENIX NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen) |
| [`FRONTIER/SYMPHONY.pdf`](../FRONTIER/SYMPHONY.pdf) | Symphony | [USENIX NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/agarwal) |
| [`FRONTIER/LIBRA.pdf`](../FRONTIER/LIBRA.pdf) | Libra | [USENIX NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/ruan-libra) |
| [`FRONTIER/STRATA.pdf`](../FRONTIER/STRATA.pdf) | Strata | [USENIX OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang) |
| [`FRONTIER/ECHO.pdf`](../FRONTIER/ECHO.pdf) | ECHO | [USENIX OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/liu-guangda) |
| [`FRONTIER/ECOSERVE.pdf`](../FRONTIER/ECOSERVE.pdf) | EcoServe | [USENIX OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/du) |
| [`FRONTIER/BATCHGEN.pdf`](../FRONTIER/BATCHGEN.pdf) | BatchGen | [USENIX OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/xu-tairan) |
| [`FRONTIER/DROIDSPEAK.pdf`](../FRONTIER/DROIDSPEAK.pdf) | DroidSpeak | [USENIX NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan) |
| [`FRONTIER/CONTINUUM.pdf`](../FRONTIER/CONTINUUM.pdf) | Continuum | [arXiv 2511.02230](https://arxiv.org/abs/2511.02230) |
| [`FRONTIER/KVFLOW.pdf`](../FRONTIER/KVFLOW.pdf) | KVFlow | [arXiv 2507.07400](https://arxiv.org/abs/2507.07400) |

## Course and reference PDFs

| File | Source |
|---|---|
| [`COURSE/CS229-NOTES.pdf`](../COURSE/CS229-NOTES.pdf) | [Stanford CS229 notes](https://cs229.stanford.edu/main_notes.pdf) |
| [`COURSE/BOYD-CVXBOOK.pdf`](../COURSE/BOYD-CVXBOOK.pdf) | [Boyd & Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) |
| [`COURSE/MLSYS-VOL2.pdf`](../COURSE/MLSYS-VOL2.pdf) | [Machine Learning Systems at Scale](https://mlsysbook.ai/vol2/inference/inference.html) |
| [`COURSE/CUDA-PROGRAMMING-GUIDE.pdf`](../COURSE/CUDA-PROGRAMMING-GUIDE.pdf) | [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) |
| [`COURSE/CUDA-BEST-PRACTICES.pdf`](../COURSE/CUDA-BEST-PRACTICES.pdf) | [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) |

The 30 speculative-decoding sources are recorded separately in
[`SPEC/README.md`](../SPEC/README.md).

## Shallow clones

| Local directory | Upstream | Pinned HEAD |
|---|---|---|
| `repos/cs336-lectures` | [stanford-cs336/lectures](https://github.com/stanford-cs336/lectures) | `8b59b5073076` |
| `repos/gpu-mode-lectures` | [gpu-mode/lectures](https://github.com/gpu-mode/lectures) | `b4df16e2b951` |
| `repos/transformers` | [huggingface/transformers](https://github.com/huggingface/transformers) | `b88388660761` |
| `repos/deepseek-v3` | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | `9b4e9788e4a3` |
| `repos/qwen3` | [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) | `7a2f61ffc7a2` |
| `repos/mamba` | [state-spaces/mamba](https://github.com/state-spaces/mamba) | `e9594ce1c732` |
| `repos/flash-linear-attention` | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | `3adcb3c50a9e` |
| `repos/vllm` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | `1a20d23dab6e` |
| `repos/sglang` | [sgl-project/sglang](https://github.com/sgl-project/sglang) | `b78d3999b54b` |
| `repos/tensorrt-llm` | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | `01a04f076223` |
| `repos/flash-attention` | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | `c75d019dea9d` |
| `repos/flashinfer` | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | `4b964ec4e147` |
| `repos/cutlass` | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | `f94ec46f4f63` |
| `repos/sarathi-serve` | [microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve) | `96f9911790ec` |
| `repos/distserve` | [LLMServe/DistServe](https://github.com/LLMServe/DistServe) | `82831f1604cc` |
| `repos/vidur` | [microsoft/vidur](https://github.com/microsoft/vidur) | `8383d2935bc6` |
| `repos/mooncake` | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) | `f19faaeac5d2` |
| `repos/lmcache` | [LMCache/LMCache](https://github.com/LMCache/LMCache) | `6b9b82737895` |
| `repos/deepep` | [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | `dd758caf4518` |
| `repos/deepgemm` | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | `559d79fb6994` |
| `repos/megatron-lm` | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | `ff6b92a7c371` |
| `repos/deepspeed` | [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | `d6fbd4768ae4` |
| `repos/tutel` | [microsoft/Tutel](https://github.com/microsoft/Tutel) | `ac0e51b943bb` |
| `repos/megablocks` | [databricks/megablocks](https://github.com/databricks/megablocks) | `952db33d6eac` |
| `repos/flux` | [bytedance/flux](https://github.com/bytedance/flux) | `19831ca2d820` |
| `repos/nccl-tests` | [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests) | `a0b82b2260cf` |
| `repos/dynamo` | [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) | `4bf6e438d4f4` |
| `repos/llm-d` | [llm-d/llm-d](https://github.com/llm-d/llm-d) | `611752d2f7a5` |
| `repos/servegen` | [alibaba/ServeGen](https://github.com/alibaba/ServeGen) | `7b04e997e065` |
| `repos/speculators` | [vllm-project/speculators](https://github.com/vllm-project/speculators) | `274fd59e5072` |
| `repos/eagle` | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | `cb7e0841fe0c` |
| `repos/model-optimizer` | [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | `4c3d364750a6` |
| `repos/llm-compressor` | [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) | `8cec0acc1931` |
| `repos/kimi-linear` | [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) | `8c1d85eb6b5f` |
| `repos/duo-attention` | [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention) | `fe93c314ae87` |
| `repos/kvpress` | [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) | `fcd17d7013be` |
| `repos/pythia-mlkv` | [zaydzuhri/pythia-mlkv](https://github.com/zaydzuhri/pythia-mlkv) | `c2ad06518e36` |
| `repos/trl` | [huggingface/trl](https://github.com/huggingface/trl) | `922dc584664d` |
| `repos/verl` | [verl-project/verl](https://github.com/verl-project/verl) | `aebd1f8a27d5` |
| `repos/datatrove` | [huggingface/datatrove](https://github.com/huggingface/datatrove) | `5987417f9560` |
| `repos/olmo-core` | [allenai/OLMo-core](https://github.com/allenai/OLMo-core) | `064b172e5169` |
| `repos/open-instruct` | [allenai/open-instruct](https://github.com/allenai/open-instruct) | `b18cfc57c07c` |
| `repos/medusa` | [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) | `e2a5d20c048a` |
| `repos/layerskip` | [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip) | `494752e5fbb0` |
| `repos/recurrent-drafter` | [apple/ml-recurrent-drafter](https://github.com/apple/ml-recurrent-drafter) | `bd8586bb9bbf` |
| `repos/lookahead-decoding` | [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) | `eed010da9c7b` |
| `repos/sequoia` | [Infini-AI-Lab/Sequoia](https://github.com/Infini-AI-Lab/Sequoia) | `0ff3ff71475c` |
| `repos/rest` | [FasterDecoding/REST](https://github.com/FasterDecoding/REST) | `154d0f0ae45d` |
| `repos/magicdec` | [Infini-AI-Lab/MagicDec](https://github.com/Infini-AI-Lab/MagicDec) | `09cd671c01bc` |
| `repos/xgrammar` | [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) | `3fb48bfd422a` |

All 50 repositories were verified shallow and clean after clone. They occupy approximately 5.4 GiB total;
no model checkpoints or Python environments were downloaded.
