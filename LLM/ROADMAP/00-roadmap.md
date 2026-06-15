# LLM Speculative-Decoding Roadmap (systems-focused)

A study path through `LLM/SPEC` for someone coming from an ANN / vector-database systems
background. The framing is the same as the VDB and VLA roadmaps: treat decode acceleration as a
**cost-model + approximation + verification** problem, and lean on the home-turf skills
(memory-bandwidth analysis, top-k decisions under uncertainty, KV/cache compression, scheduling).

Paper handles below (CAPS) refer to files in [`../SPEC/README.md`](../SPEC/README.md).

---

## 0. How this relates to the VDB / VLA roadmaps

- VDB roadmap §5–§6 (estimation theory, rate-distortion) is the math under **lossless** drafting:
  the accept/reject step is a modified rejection sampler, and acceptance length `τ` is an estimator.
- VLA/SPEC (SPECVLA, CEEDVLA) is the *application* of everything here to robot models. This folder
  is the upstream theory; read it first, then the VLA versions read trivially.
- The single biggest transferable idea: speculative decoding is **"approximate then verify,"** the
  same shape as ANN candidate-generation + rerank.

## 1. First, build the decode pipeline into a cost model

Before any method, write down where the time goes. Autoregressive decode is **memory-bandwidth
bound**, not compute bound (GPU utilization often <15%): each step streams all weights to emit one
token. Speculative decoding trades spare compute for fewer serial steps.

### Target skill
Express end-to-end decode time as `T = serial_steps × per_step_bandwidth_cost`, and predict the
speedup of any draft-and-verify scheme from three numbers: acceptance length `τ`, draft cost, and
verify (parallel) cost. Know the metrics: `τ` (mean accepted tokens/round), `S` (speedup), `Acc`.

### 📚 Read
SPECSURVEY (taxonomy + Spec-Bench; read the intro and the metric definitions first).

## 2. Foundations: draft-and-verify, and why it is lossless

The core result: a small draft proposes tokens, the target verifies them in one parallel pass, and a
**modified rejection sampling** step keeps the output distribution exactly equal to the target's.

### Target skill
Re-derive the acceptance rule and prove the output is drawn from the target distribution. This is the
VDB-roadmap estimation lens applied to tokens.

### 📚 Read
SPECDECODE, SPECSAMPLE. (Two near-simultaneous foundations — Google and DeepMind.)

## 3. Kill the separate draft model: multi-token / multi-head self-drafting

Instead of a second model, bolt extra heads onto the target that predict several future tokens.

### Target skill
Tell apart sequentially-*independent* heads (MEDUSA) from sequentially-*dependent* ones (HYDRA,
REDRAFTER), and understand why training a drafter that matches the target distribution matters.

### 📚 Read
MEDUSA (multi-head + tree attention) → MTP (n independent heads, shared trunk) → DEEPSEEKV3
(cascaded single-layer MTP, shipped at scale) → HYDRA → REDRAFTER (RNN drafter) → PARALLELSPEC,
CLOVER2.

## 4. Feature-level drafting: the EAGLE family

Autoregress at the **feature** level rather than the token level; the strongest open line.

### Target skill
Follow the 1→2→3 progression: feature-uncertainty fix → dynamic draft tree → multi-layer fusion +
training-time test. Understand HASS's train/decode-mismatch fix.

### 📚 Read
EAGLE → EAGLE2 → EAGLE3 → HASS.

## 5. Tree-structured drafting and verification

Verify a *tree* of candidates in one pass, not a single chain — directly your ANN candidate-set
intuition, and an optimization problem (best tree under a node budget).

### Target skill
Derive the optimal speculation tree (Sequoia's DP) and the ancestor-only attention mask; see how
block-level verification raises acceptance over token-by-token.

### 📚 Read
SPECINFER (token-tree serving) → SEQUOIA (optimal tree via DP) → BLOCKVERIFY (block-joint
verification) → DDTREE (tree built from a diffusion drafter's per-position distributions).

## 6. Self-speculative / early-exit (home turf: no extra model, no extra memory)

Draft with a subset of the model's own layers, verify with the full model. Training-light or
training-free, fast to ship.

### Target skill
Compare layer-skip drafting (DRAFTVERIFY), an early-exit training recipe (LAYERSKIP), and a shallow
sub-network + adapter with double early-exit (KANGAROO).

### 📚 Read
DRAFTVERIFY → LAYERSKIP → KANGAROO.

## 7. Draft-model-free and deployment systems

Remove the drafter entirely, or engineer the serving path.

### Target skill
See three ways to skip a trained drafter: Jacobi/n-gram parallel decode (LOOKAHEAD), retrieval
(REST), and a large deterministic tree for offloaded inference (SPECEXEC).

### 📚 Read
LOOKAHEAD → REST → SPECEXEC.

## 8. Long-context / KV-cache bottleneck (home turf: maps to ANN + KV compression)

At long context and large batch, the bottleneck moves to the KV cache. Sparse-KV drafting is the
fix — and sparse KV selection *is* approximate top-k retrieval.

### Target skill
Explain why naive speculative decoding stops helping at long context / high batch, and how a
sparse-KV draft restores the win. Connect to VDB quantization/selection.

### 📚 Read
TRIFORCE (hierarchical, retrieval-based sparse-KV draft) → MAGICDEC (speedup even at large batch) →
LONGSPEC (constant-size draft KV + position-index fix).

## 9. Diffusion-based drafters

The newest line: a block-diffusion drafter emits a whole block in one forward pass.

### Target skill
Understand block diffusion as a drafter (one pass → per-position distributions over a block), and how
DFlash/DDTree turn that into lossless block / tree verification.

### 📚 Read
BLOCKDIFF (the conceptual basis) → DFLASH (one-pass block drafting) → CADDT (cost-aware diffusion
draft trees).

## 10. Transfer map (the biggest opportunity)

- **← VDB:** concentration inequalities + estimation theory (VDB roadmap §2, §5) explain acceptance-
  rate bounds; rate-distortion (§6) frames KV/draft compression. Sparse-KV drafting (§8) is ANN top-k.
- **→ VLA:** SPECVLA and CEEDVLA in `../../VLA/SPEC` are these methods ported to action models; read
  them after §4–§6 here.

## 11. Suggested learning order

1. Cost model + metrics: §1 (SPECSURVEY intro).
2. Foundations + losslessness: §2 (SPECDECODE, SPECSAMPLE) — derive the acceptance rule.
3. Pick one strong modern line end-to-end: EAGLE → EAGLE2 → EAGLE3 (§4).
4. Trees: SEQUOIA + BLOCKVERIFY (§5).
5. Home turf first for shipping: self-speculative (§6) and long-context (§8).
6. Breadth: multi-token (§3), draft-model-free (§7), diffusion (§9).

## 12. The research lenses to end up with

- Every method = a choice of **drafter** × **verifier** × **what it preserves** (distribution /
  matrix product / attention logits).
- Acceptance length is an **estimator**; tree shape is an **optimization**; KV sparsity is **ANN**.
- The fastest wins to ship are training-free or self-speculative; the deepest wins are feature-level
  and long-context.
