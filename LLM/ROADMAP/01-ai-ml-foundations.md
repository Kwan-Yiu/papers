# AI, Machine Learning, and PyTorch Foundations

> **Role:** establish the minimum AI foundation needed to understand and modify an LLM
>
> **Audience:** systems researchers who do not yet have a complete ML background
>
> **Boundary:** learn enough training semantics to reason about inference; this is not a general AI survey

[Roadmap index](README.md) ·
[Overview](00-roadmap.md) ·
[Transformer foundations](02-transformer-foundations.md) ·
[Competency gates](COMPETENCY-GATES.md)

---

## Learning Contract

This layer is complete when the concepts below can be used in code and equations. Reading definitions
without being able to inspect tensor shapes, gradients, loss, and model state is insufficient.

```text
data
→ tensors
→ model
→ logits
→ loss
→ gradients
→ optimizer update
→ evaluation
```

### Document guide

| Section | Focus |
|---|---|
| 1 | AI/ML vocabulary and problem formulation |
| 2 | just-in-time mathematics |
| 3 | neural-network mechanics |
| 4 | training, evaluation, and generalization |
| 5 | PyTorch foundations |
| 6 | required builds and exit gate |

---

## 1. AI and Machine-Learning Vocabulary

### 1.1 AI, machine learning, and deep learning

| Term | Working definition |
|---|---|
| Artificial intelligence | systems that perform tasks associated with perception, prediction, reasoning, or action |
| Machine learning | algorithms that fit behavior from data rather than only hand-written rules |
| Deep learning | machine learning based on multi-layer differentiable neural networks |
| Foundation model | a broadly trained model adapted or prompted for many downstream tasks |
| Language model | a model that assigns probabilities to token sequences |
| Large language model | a language model with enough capacity and data to support broad language behavior |

For this roadmap, the important transition is:

```text
statistical model
→ neural network
→ sequence model
→ Transformer language model
→ stateful inference system
```

### 1.2 Learning problem types

| Type | Input | Target / feedback | LLM relevance |
|---|---|---|---|
| supervised learning | labeled examples | explicit target | SFT, classification, reward models |
| self-supervised learning | raw data | target derived from data | next-token pretraining |
| unsupervised learning | unlabeled data | structure/objective | representation and clustering concepts |
| reinforcement learning | state/action interaction | reward | some post-training and agent policies |
| preference learning | ranked/chosen responses | comparative signal | DPO/RLHF-style alignment |

Only self-supervised next-token modeling is required before Transformer inference. The others are
adjacent unless a workload depends on them.

### 1.3 Dataset vocabulary

- **sample/example:** one training instance;
- **feature/input:** information given to the model;
- **label/target:** desired prediction;
- **training set:** data used to update parameters;
- **validation set:** data used to select hyperparameters and detect overfitting;
- **test set:** held-out data used for final reporting;
- **batch:** examples processed together;
- **epoch:** one pass over a finite training dataset;
- **distribution shift:** production data differs from training/evaluation data.

Never compare models or systems without checking whether preprocessing, tokenization, data split, and
quality constraints are equivalent.

---

## 2. Just-in-Time Mathematics

### 2.1 Scalars, vectors, matrices, and tensors

```text
scalar:  x              shape []
vector:  x_i            shape [d]
matrix:  X_ij           shape [m, n]
tensor:  X_ijk...       shape [d1, d2, d3, ...]
```

A tensor is both:

1. a multidimensional array with a shape and dtype;
2. a mathematical object participating in a computation graph.

For every operation, write:

```text
input shapes
output shape
broadcast dimensions
dtype
device
lifetime
```

### 2.2 Matrix multiplication

```text
X : [B, d_in]
W : [d_in, d_out]
Y = XW : [B, d_out]
```

Each output is a weighted sum:

```text
Y[b, j] = Σ_i X[b, i] W[i, j]
```

This one operation underlies embeddings, attention projections, FFNs, output heads, and most
Transformer FLOPs.

Approximate multiply-add work:

```text
FLOPs(XW) ≈ 2 × B × d_in × d_out
```

### 2.3 Dot products, similarity, and projection

```text
dot(x, y) = Σ_i x_i y_i
```

The dot product combines magnitude and directional alignment. Attention uses scaled dot products to
compare queries and keys.

Orthogonal projection onto a unit vector `u`:

```text
projection_u(x) = (x · u) u
```

Projection intuition is useful for embeddings, low-rank approximation, LoRA, and quantization error.

### 2.4 Probability and distributions

For mutually exclusive outcomes:

```text
p_i ≥ 0
Σ_i p_i = 1
```

Conditional probability:

```text
P(A | B) = P(A, B) / P(B)
```

A language model factorizes a sequence:

```text
P(x_1, ..., x_T) = Π_t P(x_t | x_<t)
```

The model does not directly emit words; it emits logits that define a categorical distribution over
the next token.

### 2.5 Softmax and log-sum-exp

For logits `z`:

```text
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

Stable implementation:

```text
softmax(z_i)
= exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
```

Log-sum-exp:

```text
log Σ_j exp(z_j)
= m + log Σ_j exp(z_j - m), where m = max(z)
```

### 2.6 Cross entropy and maximum likelihood

For correct class/token `y`:

```text
loss = -log p(y)
```

Minimizing token cross entropy is equivalent to maximizing the likelihood of observed next tokens.

For a sequence:

```text
L = -Σ_t log P(x_t | x_<t)
```

Perplexity:

```text
perplexity = exp(mean token loss)
```

Perplexity is a model-quality metric, not a serving-performance metric.

### 2.7 Gradients and the chain rule

A gradient records how a scalar loss changes with each parameter:

```text
∂L / ∂W
```

For composed functions:

```text
y = f(g(x))
dy/dx = (df/dg) × (dg/dx)
```

Backpropagation applies the chain rule in reverse topological order through the computation graph.

### 2.8 Optimization

Basic gradient descent:

```text
θ ← θ - η ∇_θ L
```

Where:

- `θ` is the parameter set;
- `η` is the learning rate;
- `∇_θ L` is the gradient.

Know the purpose of:

- SGD and momentum;
- Adam/AdamW;
- learning-rate schedules;
- gradient clipping;
- weight decay;
- gradient accumulation;
- mixed-precision loss scaling.

Detailed optimizer convergence theory is not required for inference-systems work.

### 2.9 Low rank, SVD, and approximation

Singular value decomposition:

```text
W = U Σ Vᵀ
```

Keeping only the largest `r` singular values gives a rank-`r` approximation. This supports intuition
for:

- LoRA and low-rank adapters;
- weight compression;
- low-rank KV/state representations;
- approximation error versus memory/compute reduction.

---

## 3. Neural-Network Mechanics

### 3.1 Parameters, activations, and buffers

| State | Updated by optimizer? | Typical lifetime |
|---|---:|---|
| parameter | yes | model lifetime |
| activation | no | forward/backward step |
| gradient | derived | backward/update step |
| persistent buffer | no | model lifetime |
| optimizer state | optimizer-owned | training lifetime |
| inference cache | no | request/session lifetime |

Do not confuse model parameters with runtime state such as KV cache.

### 3.2 Linear layer

```text
y = xW + b
```

The linear layer changes representation dimension but does not add non-linearity by itself.

### 3.3 Activation functions

Know the shape-preserving behavior and qualitative properties of:

- ReLU;
- GELU;
- sigmoid;
- tanh;
- SiLU/Swish.

Modern decoder FFNs commonly use gated SiLU rather than a single activation.

### 3.4 Multi-layer perceptron

```text
h = activation(xW_1 + b_1)
y = hW_2 + b_2
```

Without a non-linear activation, stacked linear layers collapse into one linear transformation.

### 3.5 Normalization

Understand:

- feature mean/variance;
- LayerNorm;
- RMSNorm;
- trainable scale and bias;
- pre-norm versus post-norm placement.

Normalization changes numerical behavior and kernel structure. Training-mode batch statistics are not
part of LayerNorm/RMSNorm inference.

### 3.6 Residual connections

```text
y = x + F(x)
```

Residual paths improve gradient flow and let blocks learn a change relative to their input. Shape
compatibility is mandatory.

### 3.7 Computation graph

A computation graph records operations and data dependencies:

```text
input
→ linear
→ activation
→ linear
→ logits
→ loss
```

This graph is central to both:

- autograd during training;
- graph capture, compilation, fusion, and runtime optimization during inference.

---

## 4. Training, Evaluation, and Generalization

### 4.1 Training loop

Canonical order:

```text
model.train()
→ load batch
→ zero gradients
→ forward
→ compute loss
→ backward
→ optimizer step
→ update metrics
```

### 4.2 Evaluation loop

Canonical order:

```text
model.eval()
→ disable gradient recording
→ forward
→ compute quality metrics
```

`model.eval()` and `torch.no_grad()` are different:

- `eval()` changes module behavior such as dropout;
- `no_grad()` disables autograd recording.

### 4.3 Underfitting and overfitting

| Pattern | Training loss | Validation loss | Interpretation |
|---|---:|---:|---|
| both high | high | high | underfitting or optimization failure |
| train low, validation high | low | high | overfitting or distribution mismatch |
| both low | low | low | good fit under this evaluation contract |

### 4.4 Quality versus systems metrics

Model metrics:

- loss/perplexity;
- accuracy/F1;
- task score;
- human/preference evaluation.

Systems metrics:

- latency;
- throughput;
- memory;
- energy/cost;
- SLO goodput.

An optimization is invalid if it improves systems metrics by silently weakening the quality contract.

### 4.5 Reproducibility

Record:

```text
random seed
dataset and split
preprocessing
model configuration
initialization/checkpoint
optimizer and learning rate
precision
software versions
hardware
commands
```

Deterministic execution can reduce performance and is not guaranteed across all GPU kernels. Declare
the actual reproducibility level.

---

## 5. PyTorch Foundations

### 5.1 Tensor essentials

Be fluent with:

- `shape`, `dtype`, `device`, and `stride`;
- indexing and slicing;
- `reshape`, `view`, `transpose`, and `permute`;
- broadcasting;
- reductions;
- matrix multiplication and `einsum`;
- CPU/GPU transfer;
- contiguous versus non-contiguous layouts.

### 5.2 Dtypes

Know the practical role of:

| dtype | Typical use |
|---|---|
| FP32 | stable reference and some accumulations |
| FP16 | low memory/compute, narrower numerical range |
| BF16 | low memory/compute with FP32-like exponent range |
| FP8 | accelerator-dependent low-precision execution |
| INT8/INT4 | quantized storage and supported kernels |
| integer token IDs | tokenizer/model interface |

### 5.3 `nn.Module`

Understand:

- parameter registration;
- submodules;
- `forward`;
- buffers;
- `state_dict`;
- train/eval mode;
- device and dtype movement;
- hooks;
- loading checkpoints with missing/unexpected keys.

### 5.4 Autograd

Be able to explain:

- `requires_grad`;
- leaf versus non-leaf tensors;
- `grad_fn`;
- accumulation in `.grad`;
- `backward`;
- graph retention;
- `detach`;
- in-place-operation hazards.

### 5.5 Data input

Know:

- `Dataset`;
- `DataLoader`;
- shuffling;
- collation and padding;
- pinned host memory;
- asynchronous device copies;
- preprocessing/tokenization bottlenecks.

### 5.6 Precision and execution modes

Distinguish:

- training mode;
- evaluation mode;
- `no_grad`;
- `inference_mode`;
- autocast/mixed precision;
- compilation;
- distributed wrappers.

### 5.7 Saving and loading

Understand the difference among:

- Python object serialization;
- `state_dict`;
- model configuration;
- tokenizer assets;
- checkpoint shards;
- SafeTensors;
- optimizer state.

Inference usually needs model configuration, tokenizer files, weights, and generation settings—not
training optimizer state.

---

## 6. Required Builds

### Build A — Linear classifier

Implement:

```text
data → linear layer → logits → cross entropy → gradient update
```

Required evidence:

- tensor shapes;
- decreasing training loss;
- separate validation metric;
- parameter/gradient inspection;
- saved and reloaded `state_dict`.

### Build B — Small MLP

Add:

- hidden layer;
- non-linearity;
- normalization or residual path;
- train/eval comparison.

Explain why the model cannot be collapsed into one linear transformation.

### Build C — Tiny next-token model

Before implementing a Transformer, build a minimal token model such as:

```text
token IDs
→ embedding
→ context aggregation
→ vocabulary logits
→ shifted cross entropy
```

The goal is to understand token prediction, shifting, sequence loss, and autoregressive sampling.

---

## Exit Gate

You can:

1. derive shapes for linear layers and batched matrix multiplication;
2. convert logits to probabilities and compute cross entropy;
3. explain forward, backward, gradients, and optimizer updates;
4. distinguish parameters, activations, optimizer state, and inference state;
5. write training and evaluation loops without copying a framework template blindly;
6. inspect dtype, device, layout, and autograd state in PyTorch;
7. save and reload a model correctly;
8. separate model-quality metrics from system-performance metrics;
9. explain next-token prediction well enough to enter Transformer mechanics.

---

## Primary Resources

- [`../COURSE/CS229-NOTES.pdf`](../COURSE/CS229-NOTES.pdf)
- [`../COURSE/BOYD-CVXBOOK.pdf`](../COURSE/BOYD-CVXBOOK.pdf)
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [MIT 18.065 Matrix Methods](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-fall-2018/)
- [PyTorch documentation and source](https://github.com/pytorch/pytorch)
- [Stanford CS336](https://github.com/stanford-cs336/lectures)

Use these selectively according to the missing concept. Completing every proof is not a prerequisite
for the next layer.

---

**Previous:** [`00-roadmap.md`](00-roadmap.md) ·
**Next:** [`02-transformer-foundations.md`](02-transformer-foundations.md) ·
**Competency gates:** [`COMPETENCY-GATES.md`](COMPETENCY-GATES.md)
