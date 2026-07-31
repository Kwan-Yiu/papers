# LLM Training and Post-Training Systems — Curated Reading Map

> **Purpose:** connect data, objectives, optimization, distributed training, supervised
> adaptation, preference learning, reasoning RL, and rollout infrastructure to the checkpoints
> later served by an inference system.
>
> **Method:** use external English courses, official documentation, primary papers, and maintained
> repositories. This file is an index and dependency map, not a replacement tutorial.
>
> **Systems boundary:** training is included because data layout, precision, parallelism,
> checkpoint format, post-training objectives, and rollout generation directly constrain
> inference architecture and serving behavior.

[Roadmap](00-roadmap.md) ·
[Modern architecture](03-modern-llm-architecture.md) ·
[Efficient inference](05-single-node-inference-optimization.md) ·
[Decoding and test-time compute](06-decoding-test-time-compute.md) ·
[Repository atlas](../RESOURCES/GITHUB-REPO-ATLAS.md)

---

## 1. Field Map

| Branch | Small aspects that must be distinguished | Representative external entry |
|---|---|---|
| data system | acquisition, extraction, filtering, quality scoring, deduplication, decontamination, mixture, sharding, streaming | [Stanford CS336](https://cs336.stanford.edu/) and [DataTrove](https://github.com/huggingface/datatrove) |
| tokenizer | normalization, pre-tokenization, BPE, WordPiece, Unigram, byte fallback, special/chat/control tokens | [Hugging Face LLM Course — Tokenizers](https://huggingface.co/learn/llm-course/en/chapter6/1) |
| objective | causal LM, masked LM, span corruption, FIM, multilingual/code mixtures, multimodal objectives | [CS336 lectures](https://github.com/stanford-cs336/lectures) |
| scaling | parameter/data/compute laws, compute-optimal allocation, data-constrained regimes, transfer and emergence caveats | [Scaling Book](https://jax-ml.github.io/scaling-book/) |
| optimization | initialization, AdamW, schedules, clipping, regularization, mixed precision, loss scaling, optimizer state | [PyTorch optimization tutorials](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) |
| distributed training | DDP, FSDP/ZeRO, TP, PP, SP/CP, EP, activation checkpointing, offload, checkpoint/resume | [PyTorch Distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) |
| supervised adaptation | continued pretraining, instruction tuning, SFT, packing, loss masks, LoRA/QLoRA, adapters | [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) |
| preference learning | preference data, reward models, PPO, DPO-family objectives, RLAIF, process supervision | [TRL documentation](https://huggingface.co/docs/trl/) |
| reasoning/agent RL | outcome/process reward, rejection sampling, GRPO-family methods, verifiable rewards, tool environments | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) |
| rollout system | actor/reference/reward models, generation engines, placement, weight synchronization, stale policies, async pipelines | [verl](https://github.com/verl-project/verl) |
| evaluation/lifecycle | contamination, capability/safety evaluation, checkpoint conversion, lineage, reproducibility, promotion | [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |

The branches are related but not interchangeable. For example:

- a tokenizer change alters sequence lengths and therefore training and inference cost;
- a data-mixture result is not a distributed-training result;
- QLoRA reduces adaptation memory but does not make the resulting serving backend automatically
  support the same low-bit format;
- DPO is an offline preference objective; PPO/GRPO-style training includes online generation;
- reasoning RL is often an inference-systems workload because rollouts dominate wall time;
- training precision, checkpoint precision, communication precision, and serving precision may
  all differ.

---

## 2. External Learning Spine

Read external explanations in this order of dependency, without imposing a calendar:

| Priority | Resource | Exact target |
|---:|---|---|
| 1 | [Stanford CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/) | tokenization, data, Transformer construction, systems, scaling, alignment, evaluation |
| 2 | [CS336 lecture source](https://github.com/stanford-cs336/lectures) | executable slides and linked primary readings |
| 3 | [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/) | tokenizer, datasets, fine-tuning, sharing, and generation interfaces |
| 4 | [Google DeepMind Scaling Book](https://jax-ml.github.io/scaling-book/) | accelerator model, roofline, sharding, collectives, training and inference math |
| 5 | [PyTorch Distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) | DDP/FSDP/TP/PP decision boundary |
| 6 | [Megatron Core developer guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/) | production TP/PP/CP/EP and distributed checkpointing |
| 7 | [OLMo-core](https://github.com/allenai/OLMo-core) | open training scripts, configs, checkpointing, metrics, and model release lineage |
| 8 | [Hugging Face PEFT](https://huggingface.co/docs/peft/) | adapter and LoRA-family adaptation |
| 9 | [TRL](https://huggingface.co/docs/trl/) | SFT, reward modeling, DPO-family methods, PPO/GRPO and trainers |
| 10 | [verl documentation](https://verl.readthedocs.io/) | RL dataflow, worker placement, training/inference integration, agentic rollouts |
| 11 | [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray + vLLM + DeepSpeed post-training system |
| 12 | [Open Instruct](https://github.com/allenai/open-instruct) | reproducible SFT, preference, reward-model and RL recipes |

Use the primary papers below to understand claims and assumptions; use the repositories to
understand actual dataflow and failure modes.

---

## 3. Data Engineering and Tokenization

### 3.1 Data-pipeline taxonomy

| Sub-aspect | External intro or documentation | Representative work | Reference repository |
|---|---|---|---|
| web acquisition and extraction | [DataTrove processing guide](https://github.com/huggingface/datatrove#readme) | [FineWeb](https://arxiv.org/abs/2406.17557) | [DataTrove](https://github.com/huggingface/datatrove) |
| format normalization | [Hugging Face Datasets processing](https://huggingface.co/docs/datasets/process) | dataset-specific documentation | [datasets](https://github.com/huggingface/datasets) |
| rule/model quality filtering | [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | [FineWeb-Edu](https://arxiv.org/abs/2406.17557) | `DataTrove/examples/fineweb.py` |
| exact/near deduplication | [DataTrove dedup examples](https://github.com/huggingface/datatrove/tree/main/examples) | [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) | `DataTrove/examples/minhash_deduplication.py` |
| benchmark decontamination | [lm-evaluation-harness decontamination](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/decontamination) | contamination analyses attached to model reports | [Open Instruct decontamination](https://github.com/allenai/open-instruct/tree/main/decontamination) |
| data mixture | [DoReMi](https://arxiv.org/abs/2305.10429) | DoReMi; domain up/down-sampling in model reports | [DataComp-LM](https://github.com/mlfoundations/dclm) |
| corpus benchmark | [DataComp-LM](https://arxiv.org/abs/2406.11794) | standardized data filtering and model-based evaluation | [mlfoundations/dclm](https://github.com/mlfoundations/dclm) |
| sharding and streaming | [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html) | streaming corpora in OLMo/Megatron | [OLMo-core data modules](https://github.com/allenai/OLMo-core) |
| packing and document boundaries | [TRL SFT data formats](https://huggingface.co/docs/trl/sft_trainer#expected-dataset-type-and-format) | sequence packing and loss masking | [TRL](https://github.com/huggingface/trl) |
| synthetic data | [DataTrove synthetic-data guide](https://github.com/huggingface/datatrove#synthetic-data-generation) | Self-Instruct and model-specific recipes | [Open Instruct](https://github.com/allenai/open-instruct) |

Track provenance, licenses, hashes, filtering decisions, mixture weights, tokenizer revision,
document boundaries, and evaluation contamination. A final token count alone is not a data-system
description.

### 3.2 Tokenizer taxonomy

| Sub-aspect | External intro | Representative work or implementation |
|---|---|---|
| byte-pair encoding | [Hugging Face BPE chapter](https://huggingface.co/learn/llm-course/en/chapter6/5) | [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) |
| WordPiece | [Hugging Face WordPiece chapter](https://huggingface.co/learn/llm-course/en/chapter6/6) | BERT tokenizer implementations |
| Unigram / SentencePiece | [Hugging Face Unigram chapter](https://huggingface.co/learn/llm-course/en/chapter6/7) | [SentencePiece](https://arxiv.org/abs/1808.06226) |
| byte-level and fallback | [Hugging Face Tokenizers components](https://huggingface.co/docs/tokenizers/main/components) | GPT-2 byte-level BPE; SentencePiece byte fallback |
| readable implementation | [Andrej Karpathy — minbpe](https://github.com/karpathy/minbpe) | compact BPE training and encoding |
| production implementation | [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) | Rust core and bindings |
| special/chat/control tokens | [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating) | model tokenizer configs and templates |
| training-serving parity | [Open Instruct tokenization verification](https://github.com/allenai/open-instruct/blob/main/docs/verify-tokenization.md) | exact rendered prompt and loss-mask checks |

Systems consequences to keep visible:

- vocabulary size changes embedding/LM-head parameters and logits cost;
- fertility changes sequence length, KV growth, attention work and batching;
- padding side, BOS/EOS, chat templates and truncation alter both correctness and cost;
- speculative decoding may require tokenizer compatibility or explicit re-encoding;
- structured generation works over tokenizer vocabulary, not raw characters.

---

## 4. Objectives, Scaling Laws, and Data Allocation

### 4.1 Objective branches

| Objective | Representative source | Why it matters downstream |
|---|---|---|
| autoregressive causal LM | [GPT-3](https://arxiv.org/abs/2005.14165) | one-token-at-a-time generation and KV reuse |
| masked language modeling | [BERT](https://arxiv.org/abs/1810.04805) | bidirectional encoder workloads |
| span corruption / encoder-decoder | [T5](https://arxiv.org/abs/1910.10683) | encode/decode lifecycle and cross-attention state |
| fill-in-the-middle | [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255) | code-generation prompt formatting and suffix context |
| multi-token prediction | [Better & Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737) | auxiliary heads usable for speculative inference |
| masked diffusion LM | [MDLM](https://arxiv.org/abs/2406.07524) and [LLaDA](https://arxiv.org/abs/2502.09992) | iterative/block decoding rather than causal KV semantics |
| multimodal next-token/interleaved objectives | model technical reports and [LLaVA](https://arxiv.org/abs/2304.08485) | encoder cost, variable visual tokens, multimodal batching |

### 4.2 Scaling-law ladder

| Read | Extract |
|---|---|
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | empirical loss scaling with model, data and compute |
| [Training Compute-Optimal Large Language Models / Chinchilla](https://arxiv.org/abs/2203.15556) | compute-optimal parameter/token allocation |
| [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) | repeated data and finite-corpus regimes |
| [DoReMi](https://arxiv.org/abs/2305.10429) | data-mixture optimization |
| [DataComp-LM](https://arxiv.org/abs/2406.11794) | controlled comparison of data curation strategies |
| [Scaling Book — Training](https://jax-ml.github.io/scaling-book/training/) | hardware-aware training accounting |

Do not turn a fitted power law into a universal theorem. Record model family, data regime, loss or
downstream metric, optimizer, tokenizer, compute definition, and extrapolation range.

---

## 5. Optimization and Numerical Systems

### 5.1 External documentation

| Aspect | Primary external resource |
|---|---|
| optimizer loop | [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) |
| Adam/AdamW | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) and [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) |
| schedules | [Transformers optimization schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) |
| automatic mixed precision | [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) |
| FP8 training | [Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/) |
| activation checkpointing | [PyTorch checkpoint documentation](https://docs.pytorch.org/docs/stable/checkpoint.html) |
| profiler | [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) and [Nsight Systems](https://docs.nvidia.com/nsight-systems/) |

### 5.2 Coverage checklist

- initialization and residual scaling;
- batch size, microbatch size, sequence length and tokens per update;
- learning-rate warmup, decay, weight decay and gradient clipping;
- parameter, gradient, master-weight, optimizer-state and accumulator dtypes;
- FP32, TF32, BF16, FP16, FP8 and loss scaling;
- fused optimizer and fused cross-entropy paths;
- activation recomputation granularity;
- gradient accumulation and communication overlap;
- NaN/Inf detection, overflow recovery and reproducible restart;
- tokens/s, model FLOP utilization, HBM utilization and straggler time.

---

## 6. Distributed Training Systems

### 6.1 Parallelism map

| Mechanism | External intro | Representative work | Production repository |
|---|---|---|---|
| replicated data parallel | [PyTorch DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) | DDP design notes | [PyTorch](https://github.com/pytorch/pytorch) |
| parameter/gradient/state sharding | [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | [ZeRO](https://arxiv.org/abs/1910.02054) | PyTorch FSDP; [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) |
| tensor parallel | [Megatron Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html#tensor-parallelism-tp) | [Megatron-LM](https://arxiv.org/abs/1909.08053) | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| pipeline parallel | [Megatron Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html#pipeline-parallelism-pp) | [GPipe](https://arxiv.org/abs/1811.06965), [PipeDream](https://arxiv.org/abs/1806.03377) | Megatron Core; DeepSpeed |
| sequence/context parallel | Megatron Core context-parallel documentation | sequence parallelism and ring/context attention papers | Megatron Core |
| expert parallel | [Megatron Core MoE guide](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/moe.html) | GShard, Switch, MegaBlocks, DeepSeekMoE | Megatron Core; DeepSpeed; MegaBlocks |
| optimizer/activation offload | [DeepSpeed ZeRO documentation](https://www.deepspeed.ai/tutorials/zero/) | ZeRO-Offload / ZeRO-Infinity | DeepSpeed |
| distributed checkpoint | [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) | framework checkpoint formats | PyTorch; Megatron Core; OLMo-core |

### 6.2 Systems aspects

- topology discovery and rank mapping;
- collective choice, message size, latency/bandwidth regime and overlap;
- global/microbatch schedule and pipeline bubbles;
- sequence-length imbalance and variable-shape batches;
- memory fragmentation and temporary collective buffers;
- checkpoint bandwidth, consistency, resharding and resume;
- elastic/preemptible training, failure detection and partial progress;
- data-loader stalls, storage throughput and host memory;
- deterministic replay and seed management;
- power, cooling, utilization and cost accounting.

The inference parallelism stage revisits many primitives, but decode uses smaller, more frequent
operations and a different latency objective.

---

## 7. Continued Pretraining, SFT, and Parameter-Efficient Adaptation

### 7.1 External path

| Branch | Intro/tutorial | Representative work | Repository |
|---|---|---|---|
| domain continued pretraining | [Transformers causal LM task guide](https://huggingface.co/docs/transformers/tasks/language_modeling) | domain-adaptation model reports | Transformers |
| instruction tuning | [Open Instruct documentation](https://allenai.github.io/open-instruct/) | [FLAN](https://arxiv.org/abs/2109.01652), [Self-Instruct](https://arxiv.org/abs/2212.10560), [LIMA](https://arxiv.org/abs/2305.11206) | Open Instruct |
| SFT trainer | [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) | established instruction-tuning recipes | TRL |
| LoRA | [PEFT LoRA guide](https://huggingface.co/docs/peft/conceptual_guides/lora) | [LoRA](https://arxiv.org/abs/2106.09685) | [PEFT](https://github.com/huggingface/peft) |
| quantized adaptation | [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) | [QLoRA](https://arxiv.org/abs/2305.14314) | PEFT; bitsandbytes |
| multi-adapter serving boundary | [vLLM LoRA documentation](https://docs.vllm.ai/en/latest/features/lora/) | Punica/S-LoRA serving work | vLLM; SGLang |

Required distinctions:

- full fine-tuning versus adapters;
- base-model quantization during adaptation versus final deployment format;
- example-level versus packed-sequence loss masking;
- prompt, assistant-only and completion-only loss;
- merged versus runtime adapters;
- one adapter per deployment versus multi-LoRA scheduling.

---

## 8. Preference Data, Reward Models, and Feedback

| Sub-aspect | External intro | Representative work | Code entry |
|---|---|---|---|
| preference-data schema | [TRL preference dataset format](https://huggingface.co/docs/trl/dataset_formats) | chosen/rejected and conversational preference sets | TRL data utilities |
| reward modeling | [TRL RewardTrainer](https://huggingface.co/docs/trl/reward_trainer) | [InstructGPT](https://arxiv.org/abs/2203.02155) | TRL; Open Instruct |
| human feedback | InstructGPT methodology | InstructGPT | OpenRLHF / Open Instruct recipes |
| AI feedback and constitutions | [Constitutional AI](https://arxiv.org/abs/2212.08073) | RLAIF / Constitutional AI | project-specific implementations |
| outcome reward | reasoning-model reports | DeepSeekMath / DeepSeek-R1 | TRL GRPO; verl |
| process reward | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | process-supervision and verifier work | [openreasoner/OpenR](https://github.com/openreasoner/openr) |
| reward robustness | reward-model evaluation literature | overoptimization, calibration and distribution shift | Open Instruct reward evaluation |

Keep annotation policy, annotator/model identity, position bias, ties, truncation, rejected-response
source, leakage, reward calibration and uncertainty visible.

---

## 9. Offline Preference Optimization

| Family | Representative paper | Maintained implementation |
|---|---|---|
| DPO | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) |
| IPO | [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) | TRL DPO loss variants |
| KTO | [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) | [TRL KTOTrainer](https://huggingface.co/docs/trl/kto_trainer) |
| ORPO | [ORPO](https://arxiv.org/abs/2403.07691) | TRL |
| SimPO | [SimPO](https://arxiv.org/abs/2405.14734) | TRL DPO loss variants |
| reference-free and robust variants | TRL [paper index](https://huggingface.co/docs/trl/paper_index) | TRL |

Do not group algorithms only by acronym. Record whether they need a reference model, the implicit
reward parameterization, divergence control, length behavior, data assumptions and implementation
memory cost.

---

## 10. Online RL, Reasoning, and Agent Post-Training

### 10.1 Algorithm lineage

| Branch | Intro or primary paper | Representative implementation |
|---|---|---|
| policy-gradient foundation | [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) | [TRL PPOTrainer](https://huggingface.co/docs/trl/ppo_trainer) |
| RLHF PPO | [InstructGPT](https://arxiv.org/abs/2203.02155) | OpenRLHF; TRL |
| group-relative optimization | [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer); verl |
| reasoning RL | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | verl; OpenRLHF; Open Instruct |
| rule/verifiable reward | DeepSeek-R1 and open reasoning recipes | TRL reward functions; Open Instruct |
| rejection sampling / best-of-N training | model post-training reports | Open Instruct generation and filtering |
| agentic RL | [verl agentic RL guide](https://verl.readthedocs.io/en/latest/start/agentic_rl.html) | verl environments and rollout workers |

### 10.2 Systems dataflow

```text
prompt/environment state
→ rollout generation
→ tool/environment interaction
→ reward/verifier
→ trajectory and token-level statistics
→ advantage/target construction
→ policy update
→ weight publication
→ next rollout policy
```

This is a systems pipeline, not merely an optimizer choice. Track:

- policy, reference, reward, value and judge/verifier models;
- colocated versus disaggregated placement;
- rollout engine (Transformers, vLLM, SGLang or custom);
- sampled-token and log-prob parity between training and inference;
- weight transfer, resharding and update frequency;
- on-policy lag and stale trajectories;
- synchronous versus asynchronous rollout/update;
- long-tail rollout lengths and stragglers;
- tool calls, sandbox execution and environment state;
- prefix sharing, KV reuse, speculative decoding and continuous batching;
- replay buffers, filtering and failed/cancelled trajectories;
- reward compute, judge bottlenecks and data serialization;
- generation throughput, useful accepted trajectories/s and end-to-end learner utilization.

---

## 11. Training–Inference Co-Design

| Training choice | Inference consequence |
|---|---|
| tokenizer and chat template | prompt length, correctness, prefix-cache identity and draft compatibility |
| GQA/MQA/MLA/CLA | KV layout, capacity, bandwidth and attention backend |
| MoE routing and expert granularity | grouped GEMM shape, all-to-all, placement and skew |
| MTP/Medusa/EAGLE heads | proposal cost, acceptance, tree verification and checkpoint loading |
| early-exit training | self-speculative decoding compatibility |
| quantization-aware training | supported deployment formats and fused kernels |
| sparsity pattern | actual sparse-kernel availability |
| adapter structure | merge policy and multi-LoRA scheduling |
| reasoning RL | output-length distribution and test-time compute demand |
| diffusion/block objective | iterative mask schedule and non-causal serving state |
| checkpoint sharding | load time, resharding and cold start |

The bridge to inference is [05-single-node-inference-optimization.md](05-single-node-inference-optimization.md)
and [06-decoding-test-time-compute.md](06-decoding-test-time-compute.md).

---

## 12. Local Repository Reading Paths

The paths below are pinned shallow clones recorded in
[`../RESOURCES/SOURCES.md`](../RESOURCES/SOURCES.md).

### Data

```text
../RESOURCES/repos/datatrove/README.md
../RESOURCES/repos/datatrove/examples/fineweb.py
../RESOURCES/repos/datatrove/examples/minhash_deduplication.py
../RESOURCES/repos/datatrove/examples/process_common_crawl_dump.py
../RESOURCES/repos/datatrove/examples/tokenize_c4.py
```

### Open pretraining stack

```text
../RESOURCES/repos/olmo-core/README.md
../RESOURCES/repos/olmo-core/src/olmo_core/
../RESOURCES/repos/olmo-core/src/scripts/official/
../RESOURCES/repos/megatron-lm/megatron/core/
../RESOURCES/repos/megatron-lm/pretrain_gpt.py
../RESOURCES/repos/deepspeed/deepspeed/
```

### SFT, preference learning, and RL

```text
../RESOURCES/repos/trl/trl/trainer/
../RESOURCES/repos/trl/examples/
../RESOURCES/repos/open-instruct/open_instruct/finetune.py
../RESOURCES/repos/open-instruct/open_instruct/reward_modeling.py
../RESOURCES/repos/open-instruct/open_instruct/dpo.py
../RESOURCES/repos/open-instruct/open_instruct/grpo.py
../RESOURCES/repos/verl/docs/hybrid_flow.rst
../RESOURCES/repos/verl/verl/
```

Start with README/documentation, then trace one complete dataflow rather than browsing unrelated
files.

---

## 13. Representative Works Index

### Data, tokenization, and scaling

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [SentencePiece](https://arxiv.org/abs/1808.06226)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [Chinchilla](https://arxiv.org/abs/2203.15556)
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
- [DoReMi](https://arxiv.org/abs/2305.10429)
- [DataComp-LM](https://arxiv.org/abs/2406.11794)
- [FineWeb](https://arxiv.org/abs/2406.17557)

### Distributed training

- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [ZeRO](https://arxiv.org/abs/1910.02054)
- [GPipe](https://arxiv.org/abs/1811.06965)
- [PipeDream](https://arxiv.org/abs/1806.03377)
- [ZeRO-Offload](https://arxiv.org/abs/2101.06840)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

### Instruction tuning and adaptation

- [FLAN](https://arxiv.org/abs/2109.01652)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Self-Instruct](https://arxiv.org/abs/2212.10560)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [LIMA](https://arxiv.org/abs/2305.11206)

### Preference learning and reasoning RL

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [KTO](https://arxiv.org/abs/2402.01306)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [ORPO](https://arxiv.org/abs/2403.07691)
- [SimPO](https://arxiv.org/abs/2405.14734)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

---

## Exit Gate

Continue when the following distinctions can be defended from external sources and real code:

- data quality, data mixture, tokenization and scaling are separate variables;
- training memory is decomposed into parameters, gradients, optimizer states, activations,
  communication buffers and temporary workspaces;
- DDP, FSDP/ZeRO, TP, PP, CP/SP and EP are selected from tensor shape, topology and failure model;
- SFT, PEFT, reward modeling, offline preference optimization and online RL have distinct
  dataflows;
- rollout generation, reward evaluation, weight synchronization and policy staleness are treated
  as systems bottlenecks;
- checkpoint/model choices can be mapped to concrete inference costs and backend requirements.

---

**Previous:** [`03-modern-llm-architecture.md`](03-modern-llm-architecture.md) ·
**Next:** [`05-single-node-inference-optimization.md`](05-single-node-inference-optimization.md) ·
**Decoding bridge:** [`06-decoding-test-time-compute.md`](06-decoding-test-time-compute.md)
