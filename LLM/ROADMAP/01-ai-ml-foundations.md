# AI, Machine Learning, and PyTorch — Curated Reading Map

> **Role:** prerequisite map for a database/systems researcher entering LLM inference
> **Target:** enough mathematics, machine learning, deep learning, and PyTorch to read Transformer and inference-system code
> **Format:** external English tutorials, courses, books, documentation, and GitHub repositories
> **Not included:** a self-contained tutorial, a calendar, or a general data-science curriculum

---

## How to Use This Map

Resource labels:

- **Core** — follow the specified chapters or source paths.
- **Branch** — use when a prerequisite is weak or a topic becomes relevant.
- **Reference** — consult on demand; do not read front to back.
- **Local PDF** — already stored in this repository.
- **Link** — keep remote unless source modification or repeated offline use is needed.

The traversal is dependency-ordered, not time-based:

1. tensor and learning-loop vocabulary;
2. just-in-time mathematics;
3. neural-network mechanics;
4. PyTorch execution;
5. a small next-token model;
6. the exit gate.

---

## Coverage Checklist

### AI and machine learning

- [ ] supervised, unsupervised, self-supervised, and reinforcement learning;
- [ ] parameters, hyperparameters, features, labels, logits, probabilities, and loss;
- [ ] training, validation, test, inference, and generalization;
- [ ] empirical risk, overfitting, regularization, and distribution shift;
- [ ] next-token prediction as self-supervised classification.

### Mathematics used later

- [ ] vectors, matrices, tensors, broadcasting, and matrix multiplication;
- [ ] dot products, norms, orthogonality, projection, eigenvalues, SVD, and low rank;
- [ ] probability distributions, expectation, variance, maximum likelihood, and Bayes rule;
- [ ] entropy, cross entropy, and KL divergence;
- [ ] derivatives, gradients, chain rule, computation graphs, and automatic differentiation;
- [ ] gradient descent, SGD, momentum, Adam, and numerical stability.

### Neural networks and PyTorch

- [ ] linear layers, embeddings, activation functions, normalization, and residual connections;
- [ ] forward pass, loss, backward pass, optimizer step, and evaluation mode;
- [ ] `Tensor`, dtype, device, layout, stride, view, and in-place operation;
- [ ] `nn.Module`, parameters, buffers, state dictionaries, autograd, and `DataLoader`;
- [ ] CPU/GPU transfer, mixed precision, inference mode, and reproducibility.

---

## 1. Fast Orientation

Use one conceptual source and one code-first source. Do not collect several introductory courses that cover the same ground.

| Priority | Resource | Format | Read / inspect | Extract |
|---|---|---|---|---|
| Core | [Dive into Deep Learning — Introduction](https://d2l.ai/chapter_introduction/index.html) | interactive book | Chapter 1 | the standard ML workflow and vocabulary |
| Core | [Andrej Karpathy — Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) | video course + notebooks | course README; first `micrograd` lecture; `makemore_part1_bigrams.ipynb` | how scalar gradients become a next-token model |
| Branch | [fast.ai Practical Deep Learning](https://course.fast.ai/) | course | Lessons 1–3 only if a top-down introduction is more useful | model, loss, data, training loop, and transfer-learning vocabulary |
| Branch | [fastai/fastbook](https://github.com/fastai/fastbook) | book repo | `01_intro.ipynb`, then use the index to locate weak areas | an application-first view of deep learning |

Stop this section when the terms in the AI/ML checklist can be placed into one training-and-inference diagram.

---

## 2. Mathematics: Minimum Core and Branches

### 2.1 Linear algebra

| Priority | Resource | Format | Exact reading target | Why it belongs here |
|---|---|---|---|---|
| Core | [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) | visual series | vectors, linear combinations, matrix multiplication, change of basis, eigenvectors | geometric intuition before notation-heavy sources |
| Core | [Dive into Deep Learning — Linear Algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html) | executable tutorial | entire section, including tensor reductions and matrix products | direct bridge from notation to tensor code |
| Core | [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | university course | lectures on matrix spaces, orthogonality, eigenvalues, and SVD | durable foundation for projections and low-rank methods |
| Branch | [MIT 18.065 Matrix Methods](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) | university course | low-rank matrices, SVD, PCA, and matrix factorization | quantization, compression, and low-rank adaptation |
| Reference | [Mathematics for Machine Learning](https://mml-book.github.io/) | open book + repo | Chapters 2–4; Chapter 10 for dimensionality reduction | formal reference when MIT/D2L coverage is too compressed |
| Reference | [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en) | source repo | `chapter_preliminaries/linear-algebra.md`; `chapter_appendix-mathematics-for-deep-learning/` | Markdown source and executable examples |

Reading check:

- trace the shapes of `X @ W`;
- identify which dimension is reduced by a dot product;
- connect SVD/low rank to weight or activation approximation.

### 2.2 Probability and information theory

| Priority | Resource | Format | Exact reading target | Extract |
|---|---|---|---|---|
| Core | [Dive into Deep Learning — Probability and Statistics](https://d2l.ai/chapter_preliminaries/probability.html) | executable tutorial | probability rules, random variables, expectation, variance | working probability vocabulary |
| Core | [Stanford CS229 Notes](../COURSE/CS229-NOTES.pdf) | Local PDF | probability review; generative learning; maximum-likelihood passages | MLE, Bayes rule, Gaussian models |
| Core | [D2L — Maximum Likelihood](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.html) | tutorial/reference | entire section | connection between likelihood and loss |
| Core | [D2L — Information Theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html) | tutorial/reference | entropy, cross entropy, KL divergence | language-model objectives and distribution comparison |
| Branch | [Mathematics for Machine Learning](https://mml-book.github.io/) | open book | Chapter 6 | a more formal probability treatment |

Reading check:

- derive cross entropy for a categorical next-token target;
- distinguish entropy, cross entropy, and KL divergence;
- explain why maximizing likelihood becomes minimizing negative log likelihood.

### 2.3 Calculus, autograd, and optimization

| Priority | Resource | Format | Exact reading target | Extract |
|---|---|---|---|---|
| Core | [3Blue1Brown — Neural Networks](https://www.3blue1brown.com/topics/neural-networks) | visual series | gradient descent and backpropagation episodes | visual model of gradients and chain rule |
| Core | [Dive into Deep Learning — Calculus](https://d2l.ai/chapter_preliminaries/calculus.html) | executable tutorial | derivatives, partial derivatives, gradients, chain rule | notation needed for backpropagation |
| Core | [D2L — Automatic Differentiation](https://d2l.ai/chapter_preliminaries/autograd.html) | executable tutorial | entire section | computation graph and gradient accumulation |
| Core | [karpathy/micrograd](https://github.com/karpathy/micrograd) | compact source repo | `micrograd/engine.py`, `micrograd/nn.py`, `demo.ipynb`, tests | a complete reverse-mode autodiff engine and MLP |
| Core | [D2L — Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html) | interactive book | 12.1, 12.3–12.6, 12.9–12.10 | gradient descent, SGD, momentum, RMSProp, Adam |
| Branch | [Convex Optimization — Boyd and Vandenberghe](../COURSE/BOYD-CVXBOOK.pdf) | Local PDF | Chapters 2–4 and 9 as reference | convexity, optimality, and descent methods |

Repository reading path for `micrograd`:

1. `micrograd/engine.py` — scalar value, local derivative, topological backward pass;
2. `micrograd/nn.py` — parameterized modules;
3. `demo.ipynb` — loss and optimization loop;
4. `test/test_engine.py` — cross-check against PyTorch.

Stop this section when a PyTorch gradient can be predicted before calling `backward()`.

---

## 3. Machine Learning and Neural-Network Mechanics

### 3.1 Linear models, classification, and generalization

| Priority | Resource | Format | Exact reading target | Extract |
|---|---|---|---|---|
| Core | [D2L — Linear Neural Networks for Regression](https://d2l.ai/chapter_linear-regression/index.html) | interactive book | 3.1, 3.4–3.7 | loss, minibatches, implementation, generalization, weight decay |
| Core | [D2L — Linear Neural Networks for Classification](https://d2l.ai/chapter_linear-classification/index.html) | interactive book | 4.1, 4.4–4.7 | softmax, cross entropy, classification, distribution shift |
| Branch | [Stanford CS229 Notes](../COURSE/CS229-NOTES.pdf) | Local PDF | linear regression, logistic regression, regularization | a more formal derivation |

### 3.2 MLPs, computation graphs, normalization, and residuals

| Priority | Resource | Format | Exact reading target | Extract |
|---|---|---|---|---|
| Core | [D2L — Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/index.html) | interactive book | 5.1–5.6 | MLP, activations, backpropagation, initialization, generalization |
| Core | [Karpathy Zero to Hero — makemore](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore) | notebooks + videos | parts 1–4 | bigram LM, MLP, normalization, manual backprop |
| Core | [D2L — Residual Networks](https://d2l.ai/chapter_convolutional-modern/resnet.html) | tutorial | residual-block sections only | residual path and optimization motivation |
| Branch | [D2L — Batch Normalization](https://d2l.ai/chapter_convolutional-modern/batch-norm.html) | tutorial | normalization mechanics; skip CNN-specific detail | normalization vocabulary before LayerNorm/RMSNorm |

Do not study the full CNN curriculum. The residual and normalization sections are included because those mechanisms recur in Transformers.

---

## 4. PyTorch Reading and Code Map

### 4.1 Official beginner spine

[PyTorch — Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) is the primary framework tutorial. Follow its official order:

| Order | Official section | Required extraction |
|---:|---|---|
| 1 | [Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) | dataset → model → loss → optimizer |
| 2 | [Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) | shape, dtype, device, indexing, broadcasting |
| 3 | [Datasets and DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) | sample, batch, shuffle, iteration |
| 4 | [Transforms](https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html) | preprocessing boundary |
| 5 | [Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) | `nn.Module`, submodules, parameters |
| 6 | [Automatic Differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | graph construction, gradients, disabling gradients |
| 7 | [Optimization Loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | training and evaluation loops |
| 8 | [Save and Load](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) | `state_dict`, serialization boundary |

### 4.2 Systems-relevant PyTorch topics

| Priority | Resource | Format | Read / inspect | Why |
|---|---|---|---|---|
| Core | [Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) | official note | graph recreation, saved tensors, grad modes, in-place correctness | framework execution and memory behavior |
| Core | [CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) | official note | asynchronous execution, streams, memory management, precision | CPU/GPU coordination |
| Core | [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html) | official reference | views, contiguity, reshape behavior | layout-sensitive kernel behavior |
| Core | [Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) | official note | floating-point limits and reduced precision | correctness under mixed/low precision |
| Core | [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | official note | seeds, nondeterminism, deterministic operations | defensible comparisons |
| Branch | [Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) | official recipe | autocast and gradient scaling | dtype and throughput trade-offs |
| Reference | [pytorch/tutorials](https://github.com/pytorch/tutorials) | source repo | `beginner_source/basics/`, `recipes_source/` | source behind the official tutorials |
| Branch | [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning) | course repo | Chapters 00–04; use notebooks for additional practice | slower, code-heavy PyTorch path |

Framework source reading is deferred. At this layer, use documentation and small programs; read PyTorch internals in the GPU/compiler layer.

---

## 5. Minimal Language-Model Bridge

The goal is to cross from generic deep learning into next-token prediction without yet learning the Transformer.

| Priority | Resource | Format | Exact path | Extract |
|---|---|---|---|---|
| Core | [Zero to Hero — makemore part 1](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb) | notebook + video | full notebook | characters, vocabulary, counts, logits, sampling |
| Core | [Zero to Hero — makemore part 2](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb) | notebook + video | full notebook | embeddings, context window, MLP language model |
| Branch | [nanoGPT](https://github.com/karpathy/nanoGPT) | compact training repo | `model.py`, then `train.py`; stop before optimization details | how a small GPT training codebase is organized |

Required evidence:

- a shape ledger from token IDs to logits;
- a short note mapping categorical cross entropy to next-token prediction;
- one small PyTorch program that trains and samples from a character model;
- a record of train/evaluation mode, seed, dtype, device, and loss.

This is a learning artifact, not a research result.

---

## 6. Repository Index

| Repository | Use | Exact starting path | Status |
|---|---|---|---|
| [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en) | book source and notebooks | `chapter_preliminaries/`, `chapter_linear-*`, `chapter_multilayer-perceptrons/` | Link |
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | autograd and MLP internals | `micrograd/engine.py` | Link |
| [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | neural nets and early language models | `lectures/micrograd/`, `lectures/makemore/` | Link |
| [pytorch/tutorials](https://github.com/pytorch/tutorials) | official tutorial sources | `beginner_source/basics/` | Link |
| [mml-book/mml-book.github.io](https://github.com/mml-book/mml-book.github.io) | formal mathematics reference | website and downloadable book | Link |
| [fastai/fastbook](https://github.com/fastai/fastbook) | optional top-down introduction | `01_intro.ipynb` | Link |
| [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning) | optional extended PyTorch practice | numbered chapter notebooks | Link |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | bridge to GPT code | `model.py`, `train.py` | Link |

Do not clone every repository. Clone only a repository whose code will be executed, modified, or repeatedly inspected offline.

---

## 7. What to Defer

Defer these topics until the corresponding roadmap layer:

- CNN architecture surveys and computer-vision pipelines;
- full convex-analysis proofs;
- distributed training internals;
- advanced reinforcement learning;
- model alignment and post-training;
- CUDA kernels and PyTorch compiler internals;
- production serving frameworks.

They are valuable, but they are not prerequisites for beginning Transformer inference.

---

## Exit Gate

Continue to [02-transformer-foundations.md](02-transformer-foundations.md) when all of the following are true:

- [ ] tensor shapes and matrix multiplications can be traced without guessing;
- [ ] softmax, cross entropy, maximum likelihood, and KL divergence can be distinguished;
- [ ] a computation graph and reverse-mode automatic differentiation can be explained from `micrograd`;
- [ ] a PyTorch training loop and evaluation loop can be read and modified;
- [ ] parameters, buffers, gradients, optimizer state, dtype, device, layout, and serialization are distinguishable;
- [ ] a small next-token model can be trained and sampled;
- [ ] numerical or reproducibility claims are accompanied by seed, dtype, device, and evaluation mode.

The standard is operational literacy, not completion of every mathematics branch.
