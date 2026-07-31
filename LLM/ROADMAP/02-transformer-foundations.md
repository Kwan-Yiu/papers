# Transformer and Hugging Face — Curated Reading Map

> **Role:** bridge from basic deep learning to decoder-only LLM execution
> **Target:** understand a Transformer from tokens through generation, then locate the same path in Hugging Face Transformers
> **Format:** external English visual guides, executable tutorials, official documentation, papers, and repository source paths
> **Not included:** an original Transformer tutorial or a training schedule

---

## How to Use This Map

Use the resources in four passes:

1. **visual pass** — obtain the end-to-end picture;
2. **executable pass** — connect each block to code and tensor shapes;
3. **decoder-only pass** — move from the original encoder–decoder model to GPT-style LLMs;
4. **framework pass** — trace loading, forward, cache, generation, and sampling in Hugging Face.

Resource labels:

- **Core** — required;
- **Branch** — use for additional explanation or a weak prerequisite;
- **Reference** — consult while reading code;
- **Local PDF** — already stored in this repository;
- **Link** — keep remote unless active source work requires a clone.

---

## Coverage Checklist

### Text and objective

- [ ] bytes, Unicode, vocabulary, token IDs, special tokens, BPE, and decoding;
- [ ] embedding table, tied weights, logits, categorical distribution, and next-token loss;
- [ ] context length, padding, attention mask, and causal mask.

### Transformer block

- [ ] Q/K/V projections and their shapes;
- [ ] scaled dot-product attention and multi-head attention;
- [ ] positional encoding and RoPE;
- [ ] residual stream, LayerNorm/RMSNorm, FFN, and SwiGLU;
- [ ] encoder–decoder versus decoder-only structure;
- [ ] MHA, MQA, and GQA at the shape level.

### Inference

- [ ] training forward pass versus prefill versus decode;
- [ ] autoregressive generation and stopping;
- [ ] greedy, temperature, top-k, top-p, and beam search;
- [ ] KV-cache contents, dimensions, update rule, and memory growth;
- [ ] batching, padding side, position IDs, and attention masks.

### Hugging Face

- [ ] tokenizer, config, model class, checkpoint, and generation config;
- [ ] `AutoTokenizer`, `AutoConfig`, and `AutoModelForCausalLM`;
- [ ] model `forward()`, `past_key_values`, `Cache`, and `generate()`;
- [ ] exact architecture-specific source path under `src/transformers/models/`;
- [ ] generation path under `src/transformers/generation/`.

---

## 1. Visual First Pass

| Priority | Resource | Format | Read / view | Extract |
|---|---|---|---|---|
| Core | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | visual blog | full article | encoder–decoder dataflow, attention, heads, residuals, positions |
| Core | [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | visual blog | full article | decoder-only generation and causal attention |
| Core | [3Blue1Brown — Attention in Transformers](https://www.3blue1brown.com/lessons/attention) | visual lesson | attention lesson and the surrounding Transformer lessons | geometric and token-to-token intuition |
| Branch | [Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) | technical blog | self-attention and multi-head sections | alternative notation and broader attention context |
| Reference | [Attention Is All You Need](../FOUNDATION/ATTENTION.pdf) | Local PDF | abstract, Sections 3–5, Figure 1, Tables 1–3 | canonical definitions and complexity claims |

Stop the visual pass when a decoder-only token path can be drawn from token ID to next-token logits without implementation detail.

---

## 2. Tokenization and Input Construction

### 2.1 Concept and implementation

| Priority | Resource | Format | Exact reading target | Extract |
|---|---|---|---|---|
| Core | [Hugging Face LLM Course — Tokenizers](https://huggingface.co/learn/llm-course/en/chapter6/1) | official course | Chapter 6 sections 1–8 | normalizer, pre-tokenizer, model, post-processor, training |
| Core | [Andrej Karpathy — minbpe](https://github.com/karpathy/minbpe) | lecture + compact repo | `lecture.md`, `minbpe/base.py`, `basic.py`, `regex.py`, `gpt4.py` | BPE training, encoding, regex splitting, special tokens |
| Core | [LLMs from Scratch — Working with Text Data](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02) | notebook + code | `ch02.ipynb`, `dataloader.ipynb` | tokenization, sliding windows, input/target shift |
| Reference | [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers/main/en/index) | official docs | quick tour and components | production tokenizer API boundaries |

### 2.2 Repository reading paths

`karpathy/minbpe`:

1. `minbpe/base.py` — shared vocabulary, merge and render utilities;
2. `minbpe/basic.py` — minimal BPE algorithm;
3. `minbpe/regex.py` — GPT-style splitting and special tokens;
4. `minbpe/gpt4.py` — compatibility with GPT-4 tokenization;
5. `tests/test_tokenizer.py` — expected behavior.

`huggingface/tokenizers`:

1. documentation quick tour;
2. `tokenizers/src/tokenizer/` for the pipeline abstraction;
3. `tokenizers/src/models/` for BPE/WordPiece/Unigram implementations;
4. language bindings only after the Rust core is understood.

Required evidence:

- tokenize one string containing Unicode, whitespace, and a special token;
- record token strings, token IDs, offsets, attention mask, and decoded output;
- explain why the model consumes IDs rather than text.

---

## 3. Original Transformer: Executable Reading

### 3.1 Primary implementation

[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) is the executable companion to the original paper.

Follow this order inside the article:

| Order | Section | Required extraction |
|---:|---|---|
| 1 | Model Architecture | encoder–decoder boundary |
| 2 | Encoder and Decoder Stacks | residual and normalization placement |
| 3 | Attention | Q/K/V shapes, scale, mask, softmax |
| 4 | Applications of Attention | self-attention, cross-attention, causal attention |
| 5 | Position-wise Feed-Forward Networks | per-token dense path |
| 6 | Embeddings and Softmax | input/output projections and weight sharing |
| 7 | Positional Encoding | position injection |
| 8 | Full Model | module composition |
| 9 | Inference | autoregressive loop and mask growth |

Repository: [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer)

Source path:

1. `the_annotated_transformer.py`;
2. search classes/functions in article order: `MultiHeadedAttention`, `attention`, `PositionalEncoding`, `PositionwiseFeedForward`, `make_model`;
3. inspect `subsequent_mask` and `greedy_decode`;
4. defer the translation data pipeline unless encoder–decoder training is a research dependency.

### 3.2 Alternative executable source

| Priority | Resource | Format | Exact target |
|---|---|---|---|
| Branch | [D2L — Attention Mechanisms and Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) | interactive book | 11.1–11.7 |
| Branch | [PyTorch — Language Modeling with `nn.Transformer`](https://docs.pytorch.org/tutorials/beginner/transformer_tutorial.html) | official tutorial | model definition, masks, training loop |
| Reference | [pytorch/tutorials](https://github.com/pytorch/tutorials) | source repo | locate the Transformer tutorial source and rendered notebook |

Use D2L when the Annotated Transformer moves too quickly through attention scoring or tensor shapes.

---

## 4. Decoder-Only GPT from the Bottom Up

### 4.1 Primary book-and-code spine

Use [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) for the decoder-only transition.

| Order | Repository path | Read / run | Exit condition |
|---:|---|---|---|
| 1 | `ch02/` | `ch02.ipynb`, `dataloader.ipynb` | input/target token shift is clear |
| 2 | `ch03/` | `ch03.ipynb`, `multihead-attention.ipynb` | causal multi-head attention shapes are recorded |
| 3 | `ch04/` | `ch04.ipynb`, `gpt.py` | a complete GPT forward path can be traced |
| 4 | `ch05/` | `ch05.ipynb`, `gpt_generate.py` | training, checkpoint loading, and generation are separated |
| Branch | `appendix-A/` | PyTorch introduction notebooks | use only for missing framework prerequisites |

Do not proceed into classification or instruction fine-tuning unless Stage 3 training/post-training is in scope.

### 4.2 Compact codebases for comparison

| Priority | Repository | Exact path | What to compare |
|---|---|---|---|
| Core | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | `model.py` | `CausalSelfAttention`, `MLP`, `Block`, `GPT`, generation |
| Branch | [karpathy/llm.c](https://github.com/karpathy/llm.c) | `train_gpt2.py`, then C/CUDA source | framework implementation versus explicit kernels/runtime |
| Branch | [pytorch-labs/gpt-fast](https://github.com/pytorch-labs/gpt-fast) | `model.py`, `generate.py`, `tp.py` | inference-oriented PyTorch, cache, compile, and TP |
| Branch | [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics) | assignment PDF and test adapters | implementation requirements and correctness tests |

The compact-code comparison should produce one table of:

- normalization placement;
- attention projection layout;
- positional method;
- MLP activation/gating;
- tied embeddings;
- cache representation;
- generation loop.

---

## 5. Modern Decoder Block Branches

This section points forward to [03-modern-llm-architecture.md](03-modern-llm-architecture.md). Read only enough to recognize the components.

| Topic | Core external explanation | Primary paper / local source | Code anchor |
|---|---|---|---|
| RoPE | [EleutherAI — Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) | [RoFormer](../ARCHITECTURE/ROPE.pdf) | Hugging Face Llama/Qwen rotary classes |
| RMSNorm | [Sebastian Raschka — LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) | [RMSNorm](../ARCHITECTURE/RMSNORM.pdf) | `LlamaRMSNorm` |
| SwiGLU / GLU variants | [Noam Shazeer — GLU Variants Improve Transformer](../ARCHITECTURE/GLU-VARIANTS.pdf) | same Local PDF | model MLP class |
| MQA | [Fast Transformer Decoding](../ARCHITECTURE/MQA.pdf) | Local PDF | KV-head count in config/model |
| GQA | [GQA](../ARCHITECTURE/GQA.pdf) | Local PDF | repeat/group KV implementation |
| Decoder families | [Sebastian Raschka — LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) | Llama, Mistral, Qwen, DeepSeek local PDFs | corresponding HF model directories |

Do not treat architecture diagrams as implementation truth. Confirm every shape and configuration field in source.

---

## 6. Hugging Face Practical Course

### 6.1 Official course path

| Priority | Resource | Exact reading target | Extract |
|---|---|---|---|
| Core | [Hugging Face LLM Course — Chapter 1](https://huggingface.co/learn/llm-course/chapter1/1) | Transformer model families and pipeline overview | library vocabulary |
| Core | [Hugging Face LLM Course — Chapter 2](https://huggingface.co/learn/llm-course/chapter2/1) | pipeline internals, models, tokenizers, batching | end-to-end framework path |
| Core | [Hugging Face LLM Course — Chapter 6](https://huggingface.co/learn/llm-course/en/chapter6/1) | tokenizer internals | input construction |
| Branch | [Hugging Face LLM Course — Chapter 10](https://huggingface.co/learn/llm-course/chapter10/1) | build and share model demos | deployment-facing API awareness |

Course repository: [huggingface/course](https://github.com/huggingface/course)

- English content lives under `chapters/en/`.
- Use `_toctree.yml` to map rendered chapters to source files.
- Do not inspect translation directories.

### 6.2 Official generation and cache documentation

| Priority | Resource | Read / inspect | Extract |
|---|---|---|---|
| Core | [Generation strategies](https://huggingface.co/docs/transformers/generation_strategies) | decoding methods and generation configuration | greedy, sampling, beam, speculative/assisted branches |
| Core | [Caching](https://huggingface.co/docs/transformers/cache_explanation) | cache data structure and update | layer/head/sequence dimensions |
| Core | [KV cache strategies](https://huggingface.co/docs/transformers/kv_cache) | dynamic, static, offloaded, quantized cache options | memory/compile trade-offs |
| Core | [Optimizing LLMs for speed and memory](https://huggingface.co/docs/transformers/llm_tutorial_optimization) | autoregressive loop and KV-cache example | framework baseline before serving engines |
| Reference | [Generation API](https://huggingface.co/docs/transformers/main_classes/text_generation) | `GenerationConfig` and `generate()` arguments | exact public API |
| Reference | [Generation internals](https://huggingface.co/docs/transformers/internal/generation_utils) | cache and logits processor classes | source navigation |

---

## 7. Hugging Face Source-Reading Path

Repository: [huggingface/transformers](https://github.com/huggingface/transformers)

Follow a single small decoder model end to end. Qwen2, Llama, or Mistral is sufficient.

### 7.1 Configuration and loading

| Order | Source path | Question |
|---:|---|---|
| 1 | `src/transformers/models/auto/configuration_auto.py` | how does a model type select a config class? |
| 2 | `src/transformers/models/auto/modeling_auto.py` | how does `AutoModelForCausalLM` select a model class? |
| 3 | `src/transformers/modeling_utils.py` | how are modules instantiated and checkpoints loaded? |
| 4 | target model `configuration_*.py` | which fields determine layers, heads, KV heads, positions, and dtypes? |

### 7.2 Model forward

For Llama:

1. `src/transformers/models/llama/configuration_llama.py`;
2. `src/transformers/models/llama/modeling_llama.py`;
3. locate `LlamaRMSNorm`, rotary embedding, attention, MLP, decoder layer, base model, and causal-LM wrapper;
4. trace `input_ids` → embeddings → layers → norm → LM head → logits;
5. record where masks, position IDs, and cache positions enter.

Repeat only the architecture-specific differences for:

- `src/transformers/models/mistral/`;
- `src/transformers/models/qwen2/`;
- `src/transformers/models/deepseek_v3/`, if supported in the current checkout.

### 7.3 Cache and generation

| Source path | Read for |
|---|---|
| `src/transformers/cache_utils.py` | cache abstractions, dynamic/static/offloaded variants |
| `src/transformers/generation/utils.py` | main generation loop and model-input preparation |
| `src/transformers/generation/configuration_utils.py` | generation configuration |
| `src/transformers/generation/logits_process.py` | temperature, top-k, top-p, penalties, constraints |
| `src/transformers/generation/stopping_criteria.py` | termination |
| architecture model file | `prepare_inputs_for_generation` and cache-position behavior |

The framework pass is complete when one call to `generate()` can be mapped to concrete source files without relying on the high-level pipeline abstraction.

---

## 8. Required Evidence

Produce these learning artifacts:

1. a tokenization record with tokens, IDs, offsets, special tokens, and round-trip decode;
2. a shape ledger for one decoder layer:
   - hidden state;
   - Q/K/V;
   - attention scores;
   - attention output;
   - MLP intermediate;
   - logits;
3. a parameter ledger for embeddings, attention projections, MLP, normalization, and LM head;
4. a minimal causal-attention implementation cross-checked against PyTorch;
5. a minimal decoder block or a completed `LLMs-from-scratch` Chapter 4 path;
6. a cached-versus-uncached generation check with matching tokens;
7. a Hugging Face source trace from `AutoModelForCausalLM` through model forward and `generate()`.

For the cache check, record:

- model and revision;
- tokenizer;
- dtype and device;
- prompt tokens;
- deterministic decoding configuration;
- cache tensor shapes per layer;
- the point at which cached and uncached paths are compared.

---

## 9. Repository Index

| Repository | Role | Starting path | Status |
|---|---|---|---|
| [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) | executable original Transformer | article and `the_annotated_transformer.py` | Link |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | decoder-only implementation spine | `ch02/` → `ch05/` | Link |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | compact tokenizer | `lecture.md`, `minbpe/` | Link |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | compact GPT | `model.py` | Link |
| [huggingface/course](https://github.com/huggingface/course) | official course source | `chapters/en/` | Link |
| [huggingface/transformers](https://github.com/huggingface/transformers) | production framework | `src/transformers/models/`, `generation/`, `cache_utils.py` | Link |
| [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | production tokenizer internals | `tokenizers/src/` | Link |
| [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics) | correctness-oriented implementation task | assignment PDF and tests | Link |
| [pytorch-labs/gpt-fast](https://github.com/pytorch-labs/gpt-fast) | inference-oriented PyTorch GPT | `model.py`, `generate.py`, `tp.py` | Link |

Clone only a repository that will be executed, modified, or repeatedly read offline.

---

## 10. What to Defer

Defer until later layers:

- full pretraining and post-training pipelines;
- every Hugging Face model family;
- multimodal processors and architectures;
- distributed generation;
- paged KV allocation and continuous batching;
- FlashAttention kernel implementation;
- quantization kernels;
- speculative decoding systems.

---

## Exit Gate

Continue to [03-modern-llm-architecture.md](03-modern-llm-architecture.md) when:

- [ ] a decoder-only Transformer can be traced from tokens to logits;
- [ ] every major tensor in attention has a recorded shape;
- [ ] causal masking, positional information, residuals, normalization, and MLP gating are located in code;
- [ ] training, prefill, and decode are distinguished;
- [ ] cached and uncached deterministic generation agree;
- [ ] tokenizer/config/checkpoint/model/generation configuration are distinguished;
- [ ] one Hugging Face architecture path and the generic generation/cache paths have been traced;
- [ ] MHA, MQA, and GQA can be compared by query-head and KV-head counts.

The goal is source-level execution literacy, not memorizing every model API.
