# VLA Inference Systems Roadmap (systems-focused)

Continues from [`../../VDB/ROADMAP/00-roadmap.md`](../../VDB/ROADMAP/00-roadmap.md) §9 (transformer/attention math) and §13 (multimodal/VLA).
You already have an ANN/SIGMOD systems foundation, so the goal here is not "learn robotics" but:

- Decompose one VLA inference pass into a cost model: **geometry/probability error + compute + memory + scheduling**;
- Transfer mature **ANN / LLM-serving** techniques (KV compression, quantization, caching, scheduling, speculative decoding) to VLA;
- Do algorithm–system co-design under a **real-time closed-loop constraint** (10–50 Hz);
- Ultimately, find the bottleneck in a VLA system and propose an ANN/quantization/serving-style fix.

> The 📚 pointers below all reference PDFs already downloaded in this repo (`../SURVEY/`, `../CACHE/`, …). **◆ = systems-priority**.

---

## 0. How this relates to the VDB roadmap

```text
VDB roadmap §9   attention = differentiable retrieval, KV cache = dynamic vector DB
VDB roadmap §11  Hoefler-style performance model: T = compute + memory + comm
VDB roadmap §13  RT / OpenVLA / π0 / Diffusion Policy intro
        │
        ▼  (this roadmap goes deeper)
VLA inference as a serving system: latency / throughput / cache / quantization / scheduling
```

The core skill is not "read more robotics papers" but forming the judgment:

```text
Given a VLA inference bottleneck,
decompose it into: vision encode + prefill + decode + action head + control loop + network
then decide: prune tokens? reuse KV? quantize? change decoding? change scheduling? change architecture?
```

---

## 1. First, build the VLA inference pipeline into a cost model

```text
images (multi-view) → vision encoder (ViT/SigLIP) → hundreds of visual tokens ┐
language instruction → text tokens ─────────────────────────────────────────┤
                                                                             ▼
                              LLM backbone (~7B, prefill + autoregressive decode)
                                                                             ▼
                              action head (discrete decode / MLP regress / diffusion / flow)
                                                                             ▼
                              action chunk (K steps) → controller (executes at 30 Hz)
```

Four stages + a systems layer, written in the style of VDB roadmap §11:

```text
T_vla = T_vision_encoder   (compute-bound; high frame-to-frame redundancy → cacheable)
      + T_prefill          (long prompt; chunked prefill / FlashAttention)
      + T_decode           (memory-bound, serial; the main battleground of LLM-serving optimization)
      + T_action_head      (discrete is slow / diffusion is multi-step; continuous regression is fastest)
      + T_control_loop      (real-time closed loop, hard 10–50 Hz constraint → async / double buffering)
```

### Target skill
For any VLA acceleration paper, instantly answer: **which stage does it hit? does it save compute or memory? is it training-free?**

### 📚 Read
- ◆ `../SURVEY/EFFICIENTVLA.pdf`, `../SURVEY/EFFICIENTVLASYS.pdf` (skim first to build the whole map)
- ◆ `../BENCH/VLAPERF.pdf` (How Fast Can I Run My VLA — a profiling template)

---

## 2. Understand a baseline VLA

```text
autoregressive:  OpenVLA (7B, Llama2 + DINOv2/SigLIP, discrete action tokens)
flow:            π0 / π0.5 (flow matching, continuous actions)
diffusion:       RDT-1B
generalist/data: Octo, Open X-Embodiment (cross-embodiment), RT-2
```

### Target skill
Explain the data flow of one OpenVLA forward pass, and contrast the inference cost of three action heads: autoregressive vs flow vs diffusion.

### 📚 Read
- `../MODEL/OPENVLA.pdf` (must-read baseline) · `../MODEL/PI0.pdf` · `../MODEL/OPENX.pdf`
- `../MODEL/OCTO.pdf` · `../MODEL/RDT.pdf` · `../MODEL/RT2.pdf` · `../MODEL/PI05.pdf`

---

## 3. Decoding acceleration: parallel / non-autoregressive / action tokenization

```text
parallel decoding   produce K steps in one forward pass, skipping per-token autoregression
continuous action   MLP regresses pose directly, one forward pass
action chunk        emit multiple future steps at once, lowering call frequency
action tokenization compress the number of action tokens via DCT+BPE
```

This is the "standard answer" of the subfield — internalize it first.

### 📚 Read
- ◆ `../DECODE/OPENVLAOFT.pdf` (**core**: parallel decode + chunking + continuous action → 25–50×)
- ◆ `../DECODE/FAST.pdf` (10× fewer action tokens) · `../DECODE/VOTE.pdf`

---

## 4. Caching: temporal KV reuse (your home turf #1, maps directly to ANN KV compression)

Consecutive robot frames overlap heavily, yet a VLA recomputes the vision encoding every step. This is **cache system design**:

```text
eviction policy   which tokens count as "static" and reusable?
hit rate / accuracy   more reuse = faster, but accuracy may drop → pareto
reuse granularity     token-level / layer-adaptive / chunk-level
recache gate          when to force a refresh
```

Isomorphic to your intuition for ANN/vector-DB caching and KV cache compression. Training-free, easy to ablate.

### Target skill
Plot the "reuse ratio vs success rate" pareto; decide whether to reuse by visual similarity, attention, or task relevance.

### 📚 Read
- ◆ `../CACHE/VLACACHE.pdf` (adaptive token caching, training-free, the entry point)
- ◆ `../CACHE/TEMPOFIT.pdf` (layer-wise temporal KV memory) · `../CACHE/KVEFFICIENT.pdf` (RNN-gated chunked KV)
- ◆ `../CACHE/STATICDYNAMIC.pdf` (static/dynamic token disentanglement)

---

## 5. Pruning: visual tokens (your home turf #2 = approximate top-k selection)

```text
visual tokens are highly redundant → drop task-irrelevant / low-attention tokens → shorter sequence → cheaper prefill/encode
essence: safe top-k token selection under an accuracy constraint (an old ANN problem)
```

### 📚 Read
- ◆ `../PRUNE/VLAPRUNER.pdf` (temporal-aware dual-level pruning)
- Can stack with §4 caching: prune first, then cache the remaining tokens

---

## 6. Quantization / edge deployment (your home turf #3, echoes VDB roadmap §6 on quantization)

```text
weight quant   INT8 / 4-bit / 1-bit
KV quant       preserve attention logits / softmax distribution (echoes ANN KV quantization)
edge runtime   llama.cpp / TensorRT, operator mapping
objective      maximize action success under a bit budget
```

VDB roadmap §6's "which error is the quantizer optimizing?" becomes "how does quantization affect action accuracy?" here.

### Target skill
Produce a "bit-width vs success rate vs memory/bandwidth" table; decide which quantization is friendliest to action accuracy.

### 📚 Read
- ◆ `../QUANT/LITEVLAEDGE.pdf` (Jetson 4-bit, ~6.6 Hz fully on-device)
- ◆ `../QUANT/QAIL.pdf` · `../QUANT/SALIENCYQUANT.pdf` (quantization-aware imitation learning)

---

## 7. Speculative / early-exit decoding (attack the serial decode bottleneck)

```text
speculative   small draft proposes + large model verifies; relaxed acceptance for action tokens
early-exit    exit layers early on easy steps; consistency models cut denoising steps
```

Transferred directly from LLM serving (EAGLE/Medusa ideas); barely started in robotics.

### 📚 Read
- ◆ `../SPEC/SPECVLA.pdf` (speculative + relaxed acceptance) · `../SPEC/CEEDVLA.pdf` (early-exit + consistency)

---

## 8. Async / real-time serving (your home turf #4 = scheduling + queueing)

The core systems problem of the closed loop: a VLA emits a 50-step chunk at low frequency while the controller runs at 30 Hz — there is a time gap.

```text
double buffering  compute the next chunk while executing the current one (predict-execute pipeline)
chunk stitching   treat chunk switching as inpainting; linear blend / interpolation to smooth jitter
future-state      predict the "state at execution time" to align prediction and execution
multi-robot sched many robots share one VLA service → continuous batching / priority (Little's law)
```

Echoes the queueing / scheduling / Little's law of VDB roadmap §11. Pure systems work, lots of room.

### Target skill
Compare "sync vs async" effective control frequency and action continuity; design scheduling for a shared multi-robot service.

### 📚 Read
- ◆ `../SERVING/VLARAIL.pdf` (model-agnostic async middleware) · `../SERVING/VLASH.pdf` (future-state-aware)
- ◆ `../SERVING/MASKEDCHUNK.pdf` · `../SERVING/LEAVENOOBS.pdf` (real-time chunk correction)

---

## 9. Cut latency at the architecture layer: fast-slow dual system / MoE

```text
fast-slow dual system   S2 (slow VLM reasoning, emits latents) + S1 (fast policy, high-frequency actions); the two run async
MoE                     route tokens to sub-networks, activate only a fraction of params → scale capacity at constant compute
```

### 📚 Read
- `../DUALSYS/FASTINSLOW.pdf` · `../DUALSYS/ASYNCFASTSLOW.pdf` (also Helix / GR00T N1, links in §13)
- `../MOE/ADAMOE.pdf` · `../MOE/HIMOEVLA.pdf` · `../MOE/FEDVLA.pdf`

---

## 10. Diffusion / flow action-head acceleration (denoising-step bottleneck)

```text
consistency / distillation  distill multi-step denoising into 1 step → 1.5 Hz → 62 Hz
shortcut flow               single-step flow
```

### 📚 Read
- ◆ `../DIFFUSION/ONEDP.pdf` (One-Step Diffusion Policy, order-of-magnitude speedup)
- `../DIFFUSION/CONSISTENCYPOLICY.pdf` · `../DIFFUSION/ONESTEPFLOW.pdf`

---

## 11. Profiling / benchmark (performance science, fastest to ship)

```text
quantify the bottleneck distribution across VLAs, produce a reproducible benchmark
caveat: benchmarks differ in diagnostic validity (don't be misled by leaderboard gains)
```

### 📚 Read
- ◆ `../BENCH/VLAPERF.pdf` (a profiling template) · `../BENCH/BENCHMARKING.pdf` (benchmark critique, must-read)

---

## 12. Retrieval-augmented robotics (leverages your ANNS expertise directly)

```text
policy/trajectory memory bank → retrieve by multimodal query → augment the policy
essence: semantic vector search + online incremental indexing + low-latency retrieval over huge trajectory stores = core ANNS problems
```

This is your least-crowded differentiation direction.

### 📚 Read
- ◆ `../RETRIEVAL/RAEA.pdf` (CVPR 2024, policy memory-bank retrieval)
- ◆ `../RETRIEVAL/RETRIEVEREASONACT.pdf` · `../RETRIEVAL/MABR.pdf`

---

## 13. ★ Transfer map: LLM/VLM serving → VLA (the biggest opportunity)

| LLM-serving technique | Status in VLA | Opportunity |
|-----------------------|---------------|-------------|
| PagedAttention / paged KV | almost absent | paged management of multi-robot / multi-view KV |
| Continuous batching | absent | throughput scheduling for many policies sharing one VLA service |
| Chunked prefill | rare | chunk the long visual prompt to cut first-token latency |
| FlashAttention / fusion | partial | direct transfer, low-hanging fruit |
| Speculative (draft) | nascent (SPEC/) | designing the draft for action tokens |
| Quantization AWQ/GPTQ/FP8 | scattered (QUANT/) | systematize + KV quantization |
| Prefix caching | absent | reuse prefixes across repeated runs of one instruction; can unify with §4 temporal reuse |
| Disaggregated prefill/decode | absent | deploy vision encoding (prefill-heavy) separately from decode |

> Topic-selection formula: **mature LLM-serving technique X × VLA's real-time/temporal/multi-robot traits = one systems paper**.
> The base papers (vLLM/PagedAttention, FlashAttention) are in `../../VDB/ROADMAP/09-transformer-attention/`.

---

## 14. Suggested learning order

```text
Phase 0  build map     SURVEY (skim) + BENCH/VLAPERF (profiling template)
Phase 1  baseline      MODEL/OPENVLA → run LIBERO → latency breakdown
Phase 2  standard ans  DECODE/OPENVLAOFT (parallel decode / chunking / continuous action)
Phase 3  go deep       CACHE → PRUNE → SERVING (caching + scheduling, training-free, easiest to publish)
Phase 4  fill in       QUANT → SPEC → DIFFUSION → DUALSYS/MOE
Phase 5  pick a topic  pick one X from §13 transfer map + RETRIEVAL (differentiation) → write a proposal
```

Every phase must produce a "number": latency breakdown / speedup / hit rate / bit-vs-success pareto.

---

## 15. The research lenses to end up with

```text
vector/ANN lens:    KV cache = dynamic vector DB; token pruning = safe top-k; trajectory memory = sequence ANN
model lens:         which stage is prefill/decode? how expensive is the action head? how sensitive is softmax to error?
systems lens:       is the bottleneck compute / memory / scheduling? can we batch/prefetch/cache/quantize?
performance lens:   is there an interpretable cost model? is it a benchmark trick or a principled design?
real-time lens:     control frequency / jitter / double buffering; P99 of a shared multi-robot service
```

In one line: **don't chase model accuracy — work on caching + scheduling + quantization + retrieval, porting mature ANN/LLM-serving techniques into VLA's real-time closed loop. That's your arbitrage.**
