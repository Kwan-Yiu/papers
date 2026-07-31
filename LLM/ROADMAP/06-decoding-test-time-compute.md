# Decoding, Speculative Execution, and Test-Time Compute — Curated Reading Map

> **Purpose:** organize the algorithms and systems that turn model logits into useful outputs:
> ordinary sampling/search, constrained generation, speculative execution, non-autoregressive
> decoding, and quality-seeking test-time compute.
>
> **Method:** learn mechanics from external English guides, verify representative papers, inspect
> reference repositories, and then trace the corresponding production-engine path.
>
> **Boundary:** this document is a navigator, not a self-contained decoding tutorial.

[Roadmap](00-roadmap.md) ·
[Training systems](04-training-post-training-systems.md) ·
[Efficient inference](05-single-node-inference-optimization.md) ·
[Inference engine](07-single-node-inference-engine.md) ·
[SPEC paper library](../SPEC/README.md)

---

## 1. Field Boundary

“Decoding” contains several different problems. Keep their objective, correctness contract, and
resource direction separate.

| Branch | Primary objective | Typical compute direction | Correctness/quality contract |
|---|---|---|---|
| greedy and stochastic sampling | choose one continuation from next-token probabilities | baseline | intentionally changes output distribution |
| beam/contrastive/search decoding | improve task-specific sequence selection | usually more compute | algorithm-defined approximation/search |
| constrained/structured generation | guarantee syntax/schema/tool-call validity | extra CPU/GPU masking, sometimes fewer retries | language/grammar conformance |
| speculative decoding | reduce target-model serial steps | extra draft and verification work | usually exact target distribution; approximate variants must be labeled |
| parallel/Jacobi/lookahead decoding | break or relax token dependency | parallel fixed-point/candidate work | method-specific exactness |
| diffusion/block decoding | change the generative factorization | iterative parallel refinement | model-defined distribution |
| test-time compute scaling | spend more inference compute for answer quality | deliberately more compute/tokens/branches | task accuracy/reward, not distribution parity |
| agent/tool execution | interleave model, tools and environment state | variable multi-call workflow | task/environment success |

Critical distinction:

> Speculative decoding normally spends cheap compute to avoid expensive serial target passes while
> preserving the target distribution. Test-time compute scaling normally spends additional target,
> verifier, search, or tool compute to improve answer quality.

---

## 2. External Tutorial and Survey Spine

| Priority | External resource | Exact target |
|---:|---|---|
| 1 | [Hugging Face — Generation strategies](https://huggingface.co/docs/transformers/generation_strategies) | greedy, sampling, beam, contrastive and custom generation |
| 2 | [Hugging Face — Generation configuration](https://huggingface.co/docs/transformers/main_classes/text_generation) | logits processors, stopping, output scores and generation state |
| 3 | [Hugging Face — Assisted decoding](https://huggingface.co/docs/transformers/main/assisted_decoding) | draft model, prompt lookup, self-speculation and tokenizer compatibility |
| 4 | [Hugging Face — Assisted Generation blog](https://huggingface.co/blog/assisted-generation) | memory-bound motivation and readable execution flow |
| 5 | [Speculative decoding survey / Spec-Bench](../SPEC/SPECSURVEY.pdf) | method taxonomy and benchmark caveats |
| 6 | [vLLM — Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) | currently supported proposers, configuration and limitations |
| 7 | [Speculators documentation](https://docs.vllm.ai/projects/speculators/en/stable/) | training and packaging EAGLE/DFlash-style speculators for vLLM |
| 8 | [TensorRT-LLM — Speculative Decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/speculative-decoding.md) | draft-target, N-gram, Medusa, ReDrafter, EAGLE and lookahead |
| 9 | [llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) | local/consumer-device draft and N-gram paths |
| 10 | [XGrammar documentation](https://xgrammar.mlc.ai/docs/) | grammar compilation, token masks and engine integration |
| 11 | [What, How, Where, and How Well? A Survey on Test-Time Scaling](https://testtimescaling.github.io/) | test-time-compute taxonomy |
| 12 | [Awesome Speculative Decoding](https://github.com/hemingkx/SpeculativeDecodingPapers) | live paper index after the pinned representative set |

Use maintained live indexes only for discovery. Cite primary papers and pin the exact code revision
used for any claim.

---

## 3. Ordinary Autoregressive Generation

### 3.1 Selection and sampling taxonomy

| Sub-aspect | External intro | Representative work or code |
|---|---|---|
| greedy decoding | [HF generation strategies](https://huggingface.co/docs/transformers/generation_strategies#greedy-search) | `transformers.GenerationMixin` |
| temperature and multinomial sampling | [HF generation strategies](https://huggingface.co/docs/transformers/generation_strategies#multinomial-sampling) | `GenerationConfig`, `TemperatureLogitsWarper` |
| top-k | HF generation guide | `TopKLogitsWarper` |
| top-p / nucleus | HF generation guide | [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) |
| min-p | [Transformers GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation) | [Turning Up the Heat](https://arxiv.org/abs/2407.01082) and its [critical re-analysis](https://arxiv.org/abs/2506.13681) |
| typical/epsilon/eta truncation | Transformers logits processors | [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) |
| repetition/length constraints | Transformers logits processors and stopping criteria | production `generation_config.json` |
| beam search | HF generation guide | length-normalized beam variants |
| diverse/constrained beam search | HF generation guide | task-specific sequence search |
| contrastive search | [HF contrastive-search introduction](https://huggingface.co/blog/introducing-csearch) | [A Contrastive Framework for Neural Text Generation](https://arxiv.org/abs/2202.06417) |
| best-of-N / parallel sampling | inference-engine sampling APIs | quality-seeking branch; see test-time compute below |

### 3.2 Systems aspects

- vocabulary projection, logits transfer and sampling placement;
- GPU versus CPU logits processors;
- per-request sampling heterogeneity inside a batch;
- deterministic random-number streams and seed semantics;
- penalties that require output-history state;
- stop-token, stop-string and maximum-length behavior;
- output-score/log-prob retention and memory traffic;
- tokenizer and detokenizer CPU cost;
- cancellation and streaming;
- effect of temperature and task on speculative acceptance.

Sampling configuration is part of the workload definition. Comparing systems under different output
lengths or stopping policies is not a controlled performance comparison.

---

## 4. Constrained and Structured Generation

### 4.1 Method map

| Sub-aspect | External intro | Representative system/repository |
|---|---|---|
| finite-choice constraints | [Outlines quickstart](https://github.com/dottxt-ai/outlines#readme) | Outlines |
| regular-expression constraints | [Guidance constrained generation](https://github.com/guidance-ai/guidance#guarantee-output-syntax-with-constrained-generation) | Guidance; Outlines |
| JSON Schema | [XGrammar JSON generation guide](https://xgrammar.mlc.ai/docs/defining_structures/json_generation.html) | XGrammar |
| context-free grammar / EBNF | [XGrammar EBNF guide](https://xgrammar.mlc.ai/docs/defining_structures/ebnf_grammar.html) | XGrammar |
| type/Pydantic-driven output | Outlines documentation | Outlines |
| function/tool-call schemas | engine structured-output documentation | XGrammar and engine adapters |
| compressed FSM/token masks | [SGLang paper](../SERVING/SGLANG.pdf) | SGLang constrained decoding |
| grammar compilation/cache | [XGrammar paper](https://arxiv.org/abs/2411.15100) | XGrammar |

Representative repositories:

- [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar);
- [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines);
- [guidance-ai/guidance](https://github.com/guidance-ai/guidance);
- [noamgat/lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer).

### 4.2 Systems aspects

- schema/grammar compilation latency;
- cache key and grammar reuse;
- tokenizer-aware valid-token computation;
- dense vocabulary masks versus sparse candidate sets;
- CPU/GPU synchronization and mask transfer;
- per-request automaton state;
- batch divergence caused by different schemas;
- interaction with speculative candidate verification;
- reasoning-parser/tool-parser compatibility;
- schema recursion, ambiguous grammars and denial-of-service limits;
- structural validity versus semantic validity.

Structured output may reduce retries while increasing per-token control overhead. Measure both.

---

## 5. Speculative Decoding Fundamentals

### 5.1 Foundation papers

| Paper | Local copy | Read for |
|---|---|---|
| Fast Inference from Transformers via Speculative Decoding | [`SPECDECODE.pdf`](../SPEC/SPECDECODE.pdf) | draft-and-verify algorithm and exactness |
| Accelerating Large Language Model Decoding with Speculative Sampling | [`SPECSAMPLE.pdf`](../SPEC/SPECSAMPLE.pdf) | rejection/resampling for stochastic decoding |
| Speculative Decoding Survey / Spec-Bench | [`SPECSURVEY.pdf`](../SPEC/SPECSURVEY.pdf) | taxonomy, benchmark and evaluation dimensions |

### 5.2 Core state machine

The external sources should make the following stages traceable in code:

```text
target prefill
→ proposer state initialization
→ draft token(s) or candidate tree
→ target verification forward
→ acceptance / rejection / residual sampling
→ commit accepted path
→ repair or roll back provisional state
→ repeat or fall back to ordinary decode
```

### 5.3 Correctness classes

| Class | Meaning |
|---|---|
| distribution preserving | output distribution matches the declared target-model sampling distribution |
| greedy exact | token sequence matches target greedy decoding under controlled numerics |
| quality preserving by evaluation | output is empirically similar, but distribution parity is not guaranteed |
| approximate | acceptance, ensemble, pruning, quantization or early-exit rule intentionally changes output |

Every method and engine configuration must declare its class. “Lossless” cannot be inferred from the
word speculative.

---

## 6. Proposer Taxonomy at a Glance

| Proposer family | Representative methods | Training needed | Main system cost |
|---|---|---:|---|
| independent small LM | SpecDecode, Speculative Sampling | no if compatible checkpoint exists | extra weights, draft KV, serial draft steps |
| distilled/custom drafter | ReDrafter, EAGLE, EAGLE-2/3 | yes | training pipeline, custom loader and kernels |
| model auxiliary heads | Medusa, Hydra, MTP | yes/uptraining | extra heads, tree candidates, verification |
| self-speculative depth | Draft & Verify, LayerSkip, Kangaroo | usually model support/uptraining | partial-layer execution and shared state |
| retrieval/prompt | REST, N-gram, prompt lookup, suffix | no | lookup/trie/automaton and host overhead |
| parallel fixed-point | Lookahead/Jacobi | no or method-specific | extra parallel token slots and iterations |
| sparse/long-context hierarchy | TriForce, MagicDec, LongSpec | method-specific | draft KV selection and multi-level state |
| diffusion/block drafter | Block Diffusion, DFlash, CADDT | yes/model-defined | block refinement and candidate-tree construction |

---

## 7. Independent Draft-Model Speculation

### 7.1 External path

| Order | Resource | Extract |
|---:|---|---|
| 1 | [Hugging Face assisted decoding](https://huggingface.co/docs/transformers/main/assisted_decoding) | minimal draft-target API and universal assisted decoding |
| 2 | [`SPECDECODE.pdf`](../SPEC/SPECDECODE.pdf) | greedy/exact draft verification |
| 3 | [`SPECSAMPLE.pdf`](../SPEC/SPECSAMPLE.pdf) | stochastic acceptance and residual distribution |
| 4 | [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) | production configuration and limitations |
| 5 | [llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) | local-device execution |

### 7.2 Draft selection

- target/draft tokenizer and vocabulary compatibility;
- model-family compatibility and universal re-encoding;
- draft parameter size and memory footprint;
- draft architecture, quantization and device;
- draft KV/state size;
- draft length or adaptive speculation depth;
- domain/task agreement and acceptance distribution;
- target batch size and verification shape;
- draft tensor parallelism and target tensor parallelism;
- colocated versus remote draft execution.

The smallest draft is not always best: low draft latency can be offset by poor acceptance, while a
larger draft can contend with the target for HBM capacity and bandwidth.

---

## 8. Multi-Token Heads and Recurrent Drafters

| Lineage | Local paper | Reference implementation | Read for |
|---|---|---|---|
| Medusa | [`MEDUSA.pdf`](../SPEC/MEDUSA.pdf) | [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) | multiple decoding heads and tree candidates |
| Hydra | [`HYDRA.pdf`](../SPEC/HYDRA.pdf) | paper-linked code | sequentially dependent draft heads |
| multi-token prediction | [`MTP.pdf`](../SPEC/MTP.pdf) | architecture/report implementations | training objective and auxiliary heads |
| DeepSeek-V3 MTP | [`DEEPSEEKV3.pdf`](../SPEC/DEEPSEEKV3.pdf) | [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | cascaded MTP in a modern MoE/MLA model |
| ReDrafter | [`REDRAFTER.pdf`](../SPEC/REDRAFTER.pdf) | [apple/ml-recurrent-drafter](https://github.com/apple/ml-recurrent-drafter) | recurrent beam drafter and tree attention |
| ParallelSpec | [`PARALLELSPEC.pdf`](../SPEC/PARALLELSPEC.pdf) | paper-linked code | parallel drafter |
| Clover-2 | [`CLOVER2.pdf`](../SPEC/CLOVER2.pdf) | paper-linked code | regressive lightweight speculation |

Systems questions:

- whether draft heads are stored in the target checkpoint or separate;
- head training data and target-hidden-state extraction;
- sequential versus parallel head dependency;
- number, width and topology of candidates;
- tree-mask construction and verification kernel;
- shared embedding/LM head;
- extra activation and KV storage;
- compatibility with quantization, TP, PP, MoE and CUDA Graphs.

---

## 9. Feature-Level Drafting: EAGLE Family

| Method | Local paper | Reference repository | Main delta |
|---|---|---|---|
| EAGLE | [`EAGLE.pdf`](../SPEC/EAGLE.pdf) | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | draft target features rather than tokens alone |
| EAGLE-2 | [`EAGLE2.pdf`](../SPEC/EAGLE2.pdf) | SafeAILab/EAGLE | context-aware dynamic draft trees |
| EAGLE-3 | [`EAGLE3.pdf`](../SPEC/EAGLE3.pdf) | SafeAILab/EAGLE and [Speculators](https://github.com/vllm-project/speculators) | training-time feature fusion and deployable speculators |
| HASS | [`HASS.pdf`](../SPEC/HASS.pdf) | paper-linked code | harmonized drafter/target representations |

External implementation guides:

- [Speculators Getting Started](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/getting_started/);
- [vLLM EAGLE deployment](https://docs.vllm.ai/en/latest/features/speculative_decoding/);
- SGLang server arguments and `python/sglang/srt/speculative/`;
- TensorRT-LLM `_torch/speculative/eagle3.py`.

Trace hidden-state capture, draft model input construction, candidate topology, target verification,
accepted-state commit and metrics. The research repository and production engine often implement
different layouts.

---

## 10. Self-Speculative and Early-Exit Decoding

| Method | Local paper | Reference repository | Mechanism |
|---|---|---|---|
| Draft & Verify | [`DRAFTVERIFY.pdf`](../SPEC/DRAFTVERIFY.pdf) | paper-linked code | skip selected layers while drafting |
| LayerSkip | [`LAYERSKIP.pdf`](../SPEC/LAYERSKIP.pdf) | [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip) | early-exit training plus self-speculation |
| Kangaroo | [`KANGAROO.pdf`](../SPEC/KANGAROO.pdf) | paper-linked code | shallow sub-network and double early exit |

Required distinctions:

- post-hoc layer skipping versus checkpoints trained for early exit;
- shared weights versus extra draft parameters;
- draft KV/state sharing and invalidation;
- fixed versus adaptive exit depth;
- per-token confidence gates;
- exact verification versus approximate early exit;
- kernel/backend support for partial-depth execution.

---

## 11. Retrieval, Prompt Lookup, N-Gram, and Suffix Proposals

| Branch | Representative source | Code path |
|---|---|---|
| retrieval datastore | [`REST.pdf`](../SPEC/REST.pdf) | [FasterDecoding/REST](https://github.com/FasterDecoding/REST) |
| prompt lookup | [HF assisted decoding](https://huggingface.co/docs/transformers/main/assisted_decoding#prompt-lookup-decoding) | Transformers candidate generator |
| N-gram | vLLM and llama.cpp speculative docs | vLLM `ngram_proposer.py`; SGLang `cpp_ngram/`; TensorRT-LLM `ngram.py` |
| suffix decoding | [vLLM speculative docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) | vLLM `suffix_decoding.py`; TensorRT-LLM `suffix_automaton.py` |
| external corpus | REST and SGLang external-corpus code | datastore/trie/corpus manager |

Workload fit:

- code editing and grounded generation with repeated prompt spans;
- templates, logs and highly repetitive domains;
- weak benefit for novel, high-entropy continuations;
- low GPU cost but potentially material CPU lookup and synchronization cost.

Track lookup latency, hit/acceptance length, corpus memory, cache locality, concurrent updates,
tenant isolation and stale/private text exposure.

---

## 12. Candidate Trees and Verification

| Method | Local paper | Reference implementation | Main contribution |
|---|---|---|---|
| SpecInfer | [`SPECINFER.pdf`](../SPEC/SPECINFER.pdf) | [FlexFlow](https://github.com/flexflow/FlexFlow) / [artifact](https://github.com/goliaro/specinfer-ae) | tree-based speculative inference and verification |
| Sequoia | [`SEQUOIA.pdf`](../SPEC/SEQUOIA.pdf) | [Infini-AI-Lab/Sequoia](https://github.com/Infini-AI-Lab/Sequoia) | hardware-aware optimal tree |
| block verification | [`BLOCKVERIFY.pdf`](../SPEC/BLOCKVERIFY.pdf) | paper-linked code | verify blocks rather than only prefix chains |
| dynamic draft trees | [`DDTREE.pdf`](../SPEC/DDTREE.pdf) | paper-linked code | block-diffusion draft tree |

Implementation aspects:

- chain versus tree topology;
- branching factor, depth and candidate budget;
- position IDs for shared prefixes;
- tree/causal attention mask;
- packed/ragged candidate representation;
- target verification attention;
- accepted-prefix/path extraction;
- residual sampling at rejection;
- temporary KV/state allocation;
- accepted-path compaction and rejected-state reclamation;
- fixed graph buckets versus dynamic shapes;
- hardware-aware topology selection.

Accepted tokens per target call is insufficient without draft, mask, verification, compaction and
rejected-work costs.

---

## 13. Lookahead, Jacobi, and Parallel Token Decoding

| Method | Local paper | Reference repository | Classification |
|---|---|---|---|
| Lookahead Decoding | [`LOOKAHEAD.pdf`](../SPEC/LOOKAHEAD.pdf) | [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) | Jacobi-style parallel decoding and verification |
| SpecExec | [`SPECEXEC.pdf`](../SPEC/SPECEXEC.pdf) | paper-linked code | massively parallel speculation for consumer devices |

Do not automatically describe all parallel decoding as classic draft-target speculation. Record:

- initialization and fixed-point/Jacobi iteration;
- n-gram pool or candidate reuse;
- exactness for greedy versus stochastic decoding;
- extra token slots and attention masks;
- parallelism exposed to the accelerator;
- distributed and consumer-device execution;
- comparison against optimized ordinary decoding and draft-target baselines.

---

## 14. Long-Context Speculative Decoding

| Method | Local paper | Reference repository | Long-context mechanism |
|---|---|---|---|
| TriForce | [`TRIFORCE.pdf`](../SPEC/TRIFORCE.pdf) | [Infini-AI-Lab/TriForce](https://github.com/Infini-AI-Lab/TriForce) | hierarchical speculation with sparse/retrieved KV |
| MagicDec | [`MAGICDEC.pdf`](../SPEC/MAGICDEC.pdf) | [Infini-AI-Lab/MagicDec](https://github.com/Infini-AI-Lab/MagicDec) | sparse-KV drafter and throughput/latency analysis |
| LongSpec | [`LONGSPEC.pdf`](../SPEC/LONGSPEC.pdf) | paper-linked code | efficient drafting and verification for long context |

The bottleneck can shift with batch and context:

- ordinary target decode may become KV-bandwidth bound;
- a draft with full KV can reproduce the same bottleneck;
- sparse/retrieved draft KV may improve draft latency;
- verification processes more token positions and may become compute bound;
- acceptance and break-even point vary with context, batch and hardware.

Connect this branch to KV compression/eviction, sparse attention and hierarchical cache policy in
[05-single-node-inference-optimization.md](05-single-node-inference-optimization.md).

---

## 15. Diffusion and Block Drafters

| Method | Local paper | Reference repository | Read for |
|---|---|---|---|
| Block Diffusion | [`BLOCKDIFF.pdf`](../SPEC/BLOCKDIFF.pdf) | [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) | blockwise interpolation between AR and diffusion LM |
| DFlash | [`DFLASH.pdf`](../SPEC/DFLASH.pdf) | [Speculators](https://github.com/vllm-project/speculators) and engine implementations | one-pass block-diffusion drafting |
| cost-aware diffusion draft trees | [`CADDT.pdf`](../SPEC/CADDT.pdf) | paper-linked code | cost-aware tree construction |

Broader non-autoregressive context:

- [MDLM](../ARCHITECTURE/MDLM.pdf);
- [LLaDA](../ARCHITECTURE/LLADA.pdf) and [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA);
- [Block Diffusion](https://github.com/kuleshov-group/bd3lms);
- [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM);
- SGLang diffusion-model runtime paths.

Separate three roles:

1. a diffusion LM as the target generative model;
2. a diffusion/block model used only as a speculative drafter;
3. block verification layered onto an autoregressive target.

Their cache semantics, iteration schedule and correctness contracts differ.

---

## 16. Production Speculative-Decoding Systems

### 16.1 Engine support map

| Engine | Official entry | Representative supported branches | Local source root |
|---|---|---|---|
| Transformers | [Assisted decoding](https://huggingface.co/docs/transformers/main/assisted_decoding) | draft model, universal assisted, prompt lookup, self-speculative | `transformers/src/transformers/generation/` |
| vLLM | [Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) | EAGLE, MTP, draft model, PARD, MLP, N-gram, suffix, dynamic policies | `vllm/v1/spec_decode/` |
| SGLang | [speculative-decoding guide](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/speculative_decoding.mdx) | EAGLE/EAGLE3, NEXTN/MTP, standalone, N-gram, DFlash and adaptive runtime | `sglang/srt/speculative/` |
| TensorRT-LLM | [speculative-decoding guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/speculative-decoding.md) | draft-target, N-gram, Medusa, ReDrafter, EAGLE, MTP, lookahead | `tensorrt_llm/_torch/speculative/` |
| llama.cpp | [speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) | draft model and N-gram/local paths | upstream source and examples |

Feature support changes quickly. Treat the table as a reading index; confirm the pinned source and
current engine documentation before deployment.

### 16.2 End-to-end system layers

| Layer | Questions |
|---|---|
| model loading | separate draft checkpoint, embedded heads, vocab mapping, quantization, memory pools |
| request state | target KV, draft KV, provisional KV, output history, RNG, grammar state |
| scheduler | token budget, candidate budget, preemption, fairness and fallback |
| draft execution | device, stream, graph, TP degree, batch and overlap |
| verification | attention mask/layout, kernel, logits, acceptance and residual sampling |
| commit/rollback | accepted token/state commit, rejected state release, compaction |
| batching | variable acceptance, ragged shapes, head-of-line blocking, graph buckets |
| observability | proposed, accepted, rejected, fallback and time breakdown metrics |
| disaggregation | draft/target placement, transfer payload, RPC latency and failure semantics |

### 16.3 Scheduler interaction

Speculation changes the unit of scheduling from “one target token” to a variable bundle of draft and
verification work. A production scheduler must reason about:

- target token slots versus candidate token slots;
- request-specific and batch-specific speculation depth;
- disabling speculation above a concurrency break-even point;
- variable accepted length and iteration duration;
- fairness between speculative and ordinary requests;
- cancellation during draft/verification;
- request priority and deadline;
- draft resource admission;
- interaction with chunked prefill and prefill/decode disaggregation.

Read the current [vLLM dynamic speculative decoding guide](https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/)
as one production example, then verify the pinned implementation.

---

## 17. Cross-Technique Interaction Matrix

| Other technique | Speculative-decoding interaction |
|---|---|
| quantized target | verification kernel and logits numerics change; parity must be tested against the same target configuration |
| quantized draft | lower draft latency/capacity but possible acceptance loss and dequant overhead |
| GQA/MQA | changes target and draft KV bytes and verification attention |
| MLA | latent KV layout and model-specific EAGLE/MTP integration |
| cross-layer KV sharing | affects draft/target state reuse and rollback |
| sliding-window attention | draft and target window semantics must match |
| sparse attention/KV eviction | can accelerate long-context draft but may alter proposal quality |
| FlashAttention/FlashInfer | verification shapes need a supported kernel, not only an algorithm |
| prefix cache | target prefix may be reusable; draft prefix and hidden state may require separate cache identity |
| continuous batching | effective verification batch is request batch × candidate count |
| CUDA Graphs | candidate depth/tree topology creates dynamic shapes and graph-bucket pressure |
| tensor parallelism | target and draft may prefer different TP sizes; draft collectives can erase benefit |
| MoE | draft/target routing, grouped GEMM and expert communication vary with candidate expansion |
| prefill/decode disaggregation | speculation mainly changes decode workers; target/draft/KV placement becomes a three-way choice |
| structured output | candidate tokens must respect grammar state; verification and automaton updates must agree |
| multi-LoRA | adapter identity applies to target and compatible drafter; heterogeneous adapters fragment batching |
| RL rollout | long-tail batch size changes make adaptive speculation useful; sampled-token/log-prob parity is mandatory |

---

## 18. Test-Time Compute and Reasoning Search

### 18.1 External orientation

| Resource | Read for |
|---|---|
| [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) | compute allocation by problem difficulty; verifier-guided search |
| [Self-Consistency](https://arxiv.org/abs/2203.11171) | sample multiple reasoning paths and aggregate |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | branch, evaluate, backtrack and search over thoughts |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | process reward models and intermediate verification |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | learned long-form reasoning and RL |
| [s1](https://arxiv.org/abs/2501.19393) | budget forcing and simple test-time scaling |
| [test-time-scaling survey site](https://testtimescaling.github.io/) | broad taxonomy and live paper index |

### 18.2 Method taxonomy

| Branch | Representative methods | Systems cost |
|---|---|---|
| longer internal reasoning | fixed/dynamic token budget, budget forcing | longer decode, more KV, uncertain output length |
| parallel sampling | best-of-N, majority vote, self-consistency | N replicas/branches, shared-prefix opportunities |
| verifier/reranker | outcome RM, process RM, LLM judge | extra model placement and inference |
| tree/graph search | beam over reasoning steps, Tree of Thoughts, MCTS | dynamic branching, state lifecycle, irregular batching |
| iterative refinement | critique-revise, self-refine | serial model calls and growing context |
| tool execution | code, search, retrieval, environment actions | external latency, sandbox and resumable state |
| adaptive allocation | difficulty/uncertainty-aware compute budget | online policy, admission and SLO control |
| model cascades | route/escalate between models | quality prediction, cold start and multi-model memory |

### 18.3 Representative repositories

- [simplescaling/s1](https://github.com/simplescaling/s1);
- [princeton-nlp/tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm);
- [openreasoner/openr](https://github.com/openreasoner/openr);
- [testtimescaling/testtimescaling.github.io](https://github.com/testtimescaling/testtimescaling.github.io);
- inference engines for shared-prefix batching and rollout generation.

### 18.4 Systems research surface

- allocate compute from task difficulty, uncertainty and SLO;
- batch branches without destroying prefix locality;
- share immutable prefix KV while isolating branch state;
- schedule short answers against long reasoning tails;
- place policy, verifier, reward and tool services;
- cancel losing branches and reclaim state quickly;
- cache verifier results and tool outputs safely;
- combine speculative decoding with reasoning branches;
- account for useful-answer goodput rather than raw tokens/s;
- expose quality–latency–cost Pareto fronts.

---

## 19. Agent and Multi-Call Inference

Agent workloads extend decoding into a stateful workflow:

```text
prompt/prefix
→ reasoning tokens
→ structured tool call
→ external execution
→ appended observation
→ more model generation
→ optional branching/verifier
→ final response
```

External entries:

- [ReAct](https://arxiv.org/abs/2210.03629);
- [SGLang](../SERVING/SGLANG.pdf) for structured multi-call programs and prefix reuse;
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/);
- [verl agentic RL guide](https://verl.readthedocs.io/en/latest/start/agentic_rl.html).

Systems aspects:

- session and branch state;
- prefix-cache identity and invalidation;
- suspend/resume while waiting for tools;
- structured-output correctness;
- sandbox isolation and tool timeout;
- long-tail and bursty arrivals;
- multi-turn KV/state tiering;
- routing from cache locality and model/adapter availability;
- cancellation, replay and idempotency;
- auditability and privacy boundaries.

---

## 20. Correctness and Reproducibility

### 20.1 Correctness checks

- same tokenizer revision, chat template, special tokens and stop rules;
- same target weights, precision, attention backend and quantization;
- greedy token parity when greedy exactness is claimed;
- seeded distributional testing when sampling parity is claimed;
- residual/rejection sampling implementation matches the paper;
- draft-only randomness cannot corrupt target RNG semantics;
- accepted-token logits/log-probs are defined consistently;
- provisional KV/state is committed or rolled back correctly;
- grammar state advances only on committed tokens;
- cancellation/preemption does not leak provisional state;
- batched and unbatched outputs satisfy the same declared contract;
- CUDA Graph/eager and TP configurations are checked separately.

### 20.2 Version record

Record:

```text
target model + revision
draft/speculator model + revision
tokenizer + chat template
sampling and stopping configuration
engine commit and configuration
attention/speculative/structured-output backend
precision and quantization
hardware and topology
batch/concurrency/context/output distributions
correctness class and test
```

---

## 21. Evaluation Matrix

### 21.1 Speculative-decoding metrics

| Layer | Metrics |
|---|---|
| proposal | draft tokens/s, proposal latency, candidate count/depth, draft memory |
| acceptance | accepted draft tokens, accepted total tokens, acceptance-length distribution, rejection position |
| verification | target calls, verification latency, verified tokens/s, verification FLOPs/bytes |
| wasted work | rejected draft tokens, rejected verification positions, provisional KV bytes, rollback/compaction time |
| engine | CPU overhead, launches, graph hit rate, scheduler delay, fallback rate |
| request | TTFT, TPOT/ITL, E2E latency, output length |
| service | throughput, goodput under SLO, P50/P95/P99, fairness, power/cost |
| quality/correctness | greedy parity, distribution test, benchmark quality, structured validity |

### 21.2 Required workload axes

- target and draft model;
- prompt and output-length distribution;
- task/domain and entropy;
- greedy versus sampling and temperature/truncation;
- batch size, concurrency and arrival process;
- context length and prefix-sharing distribution;
- speculation depth/tree topology;
- hardware, target/draft placement and parallelism;
- quantization and attention backend;
- ordinary decoding, simple draft and production-engine baselines.

Plot the break-even boundary. A speedup at batch size one does not establish service goodput, and an
acceptance-rate increase does not establish latency reduction.

### 21.3 Test-time-compute metrics

- task accuracy, pass@k, reward and calibration;
- total generated/verified tokens;
- policy, verifier, tool and external-service calls;
- FLOPs, GPU seconds, joules and monetary cost per solved task;
- TTFT, time-to-correct-answer and tail latency;
- branch count/depth and cancellation efficiency;
- quality–latency–cost Pareto frontier;
- useful solved tasks per second under a declared SLO/budget.

---

## 22. Local Source-Reading Paths

### Readable method implementations

```text
../RESOURCES/repos/medusa/notebooks/medusa_introduction.ipynb
../RESOURCES/repos/medusa/notebooks/medusa_inference_explained.ipynb
../RESOURCES/repos/layerskip/self_speculation/
../RESOURCES/repos/recurrent-drafter/docs/speculative_sampling.md
../RESOURCES/repos/recurrent-drafter/docs/tree_attention.md
../RESOURCES/repos/lookahead-decoding/minimal.py
../RESOURCES/repos/lookahead-decoding/lade/decoding.py
../RESOURCES/repos/rest/rest/
../RESOURCES/repos/sequoia/Tree/
../RESOURCES/repos/magicdec/Engine/
../RESOURCES/repos/eagle/
```

### Training deployable speculators

```text
../RESOURCES/repos/speculators/docs/
../RESOURCES/repos/speculators/src/
```

### Production engines

```text
../RESOURCES/repos/transformers/src/transformers/generation/candidate_generator.py
../RESOURCES/repos/transformers/src/transformers/generation/utils.py
../RESOURCES/repos/vllm/vllm/v1/spec_decode/
../RESOURCES/repos/sglang/python/sglang/srt/speculative/
../RESOURCES/repos/tensorrt-llm/tensorrt_llm/_torch/speculative/
```

### Structured generation

```text
../RESOURCES/repos/xgrammar/cpp/
../RESOURCES/repos/xgrammar/python/
../RESOURCES/repos/vllm/vllm/v1/structured_output/
../RESOURCES/repos/sglang/python/sglang/srt/constrained/
```

Trace one algorithm end to end:

```text
configuration
→ model/speculator loader
→ scheduler metadata
→ proposer
→ verification kernel
→ acceptance sampler
→ KV/state commit
→ metrics
```

---

## 23. Complete Pinned Representative Set

The local [`SPEC/README.md`](../SPEC/README.md) records provenance and venue details for the pinned
paper library.

| Family | Pinned representative works |
|---|---|
| foundation | [SpecDecode](../SPEC/SPECDECODE.pdf), [Speculative Sampling](../SPEC/SPECSAMPLE.pdf) |
| multi-token/recurrent | [Medusa](../SPEC/MEDUSA.pdf), [MTP](../SPEC/MTP.pdf), [DeepSeek-V3](../SPEC/DEEPSEEKV3.pdf), [Hydra](../SPEC/HYDRA.pdf), [ReDrafter](../SPEC/REDRAFTER.pdf), [ParallelSpec](../SPEC/PARALLELSPEC.pdf), [Clover-2](../SPEC/CLOVER2.pdf) |
| EAGLE | [EAGLE](../SPEC/EAGLE.pdf), [EAGLE-2](../SPEC/EAGLE2.pdf), [EAGLE-3](../SPEC/EAGLE3.pdf), [HASS](../SPEC/HASS.pdf) |
| tree/block verification | [SpecInfer](../SPEC/SPECINFER.pdf), [Sequoia](../SPEC/SEQUOIA.pdf), [Dynamic Draft Trees](../SPEC/DDTREE.pdf), [Block Verification](../SPEC/BLOCKVERIFY.pdf) |
| self-speculative | [Draft & Verify](../SPEC/DRAFTVERIFY.pdf), [LayerSkip](../SPEC/LAYERSKIP.pdf), [Kangaroo](../SPEC/KANGAROO.pdf) |
| training-free/system | [Lookahead](../SPEC/LOOKAHEAD.pdf), [REST](../SPEC/REST.pdf), [SpecExec](../SPEC/SPECEXEC.pdf) |
| long context | [TriForce](../SPEC/TRIFORCE.pdf), [MagicDec](../SPEC/MAGICDEC.pdf), [LongSpec](../SPEC/LONGSPEC.pdf) |
| diffusion/block drafter | [Block Diffusion](../SPEC/BLOCKDIFF.pdf), [DFlash](../SPEC/DFLASH.pdf), [CADDT](../SPEC/CADDT.pdf) |
| survey | [Speculative Decoding Survey / Spec-Bench](../SPEC/SPECSURVEY.pdf) |

---

## Exit Gate

Continue when all of the following can be answered from external sources and traced in code:

- which distribution or quality objective each decoding branch optimizes;
- how ordinary sampling, structured generation, speculation, diffusion decoding and test-time
  compute differ;
- where draft, verification, acceptance, provisional KV, commit and rollback occur;
- how MTP/Medusa, EAGLE, self-speculation, retrieval, trees, long-context and diffusion drafters
  differ;
- why acceptance alone does not predict speedup;
- how batching, scheduler budget, CUDA Graphs, quantization, attention, MoE and parallelism alter the
  break-even point;
- how output correctness/distribution parity is tested;
- how test-time reasoning is evaluated with quality, latency and cost together.

---

**Previous:** [`05-single-node-inference-optimization.md`](05-single-node-inference-optimization.md) ·
**Next:** [`07-single-node-inference-engine.md`](07-single-node-inference-engine.md) ·
**Serving integration:** [`08-kv-scheduling-serving.md`](08-kv-scheduling-serving.md)
