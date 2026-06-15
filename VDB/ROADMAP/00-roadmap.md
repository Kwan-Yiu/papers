# A Mathematical Roadmap for High-Dimensional Vectors / AI / Large Models / HPC Systems

You have already worked on ANN/SIGMOD, so there is no need to "rebuild math from scratch." The goal of this roadmap is to fill in the next level of research capability:

- Be able to read and improve high-dimensional vector algorithms;
- Be able to analyze the theoretical limits of ANN / RaBitQ / graph search;
- Be able to transfer ANN ideas to attention, KV cache, and VLA memory;
- Be able to understand the **mathematical optimization / performance modeling** that people like Hoefler talk about in HPC/AI systems;
- Be able to analyze algorithms, systems, and hardware bottlenecks within a single unified model.

---

## 0. Overall Roadmap

```text
high-dimensional vector geometry
→ high-dimensional probability and concentration phenomena
→ random projection / sketching / LSH
→ estimation theory and uncertainty
→ quantization and information theory
→ ranking / top-k / decision theory
→ random matrices and randomized linear algebra
→ attention / KV cache / transformer math
→ graph search and irregular memory access
→ HPC / AI systems performance modeling
→ mathematical optimization / co-design / roofline / communication model
```

The core here is not "learn more math," but rather to develop a capability:

```text
given an AI system bottleneck,
be able to abstract it into:
    geometric error + probabilistic error + compute cost + memory-access cost + communication cost
then determine:
    should the algorithm be optimized?
    should the index be optimized?
    should the layout be optimized?
    should the hardware mapping be optimized?
    or should the objective function be changed?
```

---

## 1. High-Dimensional Linear Algebra: Treat All AI Objects as Vectors, Matrices, and Subspaces

### Focus

You don't need to grind through ordinary linear algebra problems again; instead you need to strengthen these concepts:

```text
inner product
norm
orthogonality
projection
rotation
subspace
SVD
low-rank approximation
spectral norm
Frobenius norm
operator norm
condition number
```

Objects in AI are essentially:

```text
embedding vector
token matrix
attention matrix
KV cache tensor
action trajectory
memory vector
graph node vector
```

You should be able to view them from a matrix perspective:

```text
QK^T
softmax(QK^T)V
Wx
low-rank adapter
KV cache compression
trajectory embedding
```

### Capability to Reach

When you see a model or a retrieval system, you can immediately decompose it into:

```text
which parts are inner product?
which parts are projection?
which parts are low-rank?
which parts are nearest-neighbor?
which parts are matrix-vector / matrix-matrix?
which errors get amplified by norm?
```

### 📚 Recommended

- **Matrix Computations — Golub & Van Loan**
- **Numerical Linear Algebra — Trefethen & Bau**
- **Linear Algebra Done Right — Sheldon Axler**
- **Mathematics for Machine Learning — Deisenroth, Faisal, Ong**

---

## 2. High-Dimensional Probability: Understanding "Why High-Dimensional Randomness Is Stable"

### Focus

The core behind high-dimensional vector algorithms is not calculus, but rather:

```text
random variables
expectation
variance
sub-Gaussian
concentration
tail bound
random projection
random rotation
```

You should be very familiar with:

```text
E[X]
Var[X]
P(|X - E[X]| > ε)
```

as well as these tools:

```text
Markov inequality
Chebyshev inequality
Hoeffding inequality
Chernoff bound
Bernstein inequality
sub-Gaussian concentration
Johnson-Lindenstrauss lemma
```

### Why It Matters

ANN, quantization, attention approximation, and KV pruning all use the same idea:

```text
a single random decision is very coarse,
but in high dimensions, averaging over many random decisions becomes stable.
```

### Capability to Reach

You can derive on your own:

```text
D random bits / random projection / sketch
why the error is usually O(1/sqrt(D))
```

You can determine:

```text
is this bound worst-case?
is it high-probability?
is it expectation?
is it instance-wise?
or is it an empirical bound?
```

### 📚 Recommended

- **High-Dimensional Probability — Roman Vershynin**
- **Probability and Computing — Mitzenmacher & Upfal**
- **Foundations of Data Science — Blum, Hopcroft, Kannan**
- **Stat 110 — Joe Blitzstein**

---

## 3. High-Dimensional Geometry: Understanding Embedding Space, ANN, and Attention Space

### Focus

High-dimensional AI objects usually live in:

```text
unit sphere
anisotropic embedding space
multimodal embedding space
token embedding space
trajectory embedding space
```

You should understand:

```text
L2 distance
cosine similarity
inner product
angular distance
MIPS
hubness
anisotropy
concentration on sphere
curse of dimensionality
```

### Key Intuitions

```text
for unit vectors:
L2, cosine, and inner product can be converted to one another.

for non-unit vectors:
norm may carry semantics, and MIPS is not equivalent to L2.

in high-dimensional space:
most random vectors are approximately orthogonal,
the gaps between nearest neighbors can be very small,
and ranking is more fragile than distance estimation.
```

### Capability to Reach

You can distinguish:

```text
when to use cosine?
when to use L2?
when can norm not be dropped?
when is MIPS the essential problem?
when is the top-k margin so small that quantization/ranking becomes unstable?
```

### 📚 Recommended

- **Foundations of Data Science — Blum, Hopcroft, Kannan**
- **Similarity Search: The Metric Space Approach — Zezula et al.**
- **The Concentration of Measure Phenomenon — Michel Ledoux**

---

## 4. Random Projection, LSH, Sketching: The Foundations of High-Dimensional Approximate Computation

### Focus

In ANN and large-model approximate computation, the following appear repeatedly:

```text
random projection
random hyperplane
SimHash
LSH
CountSketch
subspace embedding
approximate matrix multiplication
```

These things belong to the same family as RaBitQ, attention approximation, and KV cache compression.

### What to Master

```text
Johnson-Lindenstrauss lemma
random hyperplane LSH
p-stable LSH
SimHash
MinHash
CountSketch
Frequent Directions
subspace embedding
```

### The Questions You Should Be Able to Ask

```text
am I preserving distance?
preserving angle?
preserving inner product?
preserving top-k?
preserving softmax distribution?
preserving matrix product?
```

Different goals correspond to different sketch / projection / quantization.

### 📚 Recommended

- **Mining of Massive Datasets — Leskovec, Rajaraman, Ullman**
- **Sketching as a Tool for Numerical Linear Algebra — David Woodruff**
- **Randomized Algorithms for Matrices and Data — Mahoney**
- **Finding Structure with Randomness — Halko, Martinsson, Tropp**

---

## 5. Estimation Theory: Upgrading from "Computing Distances" to "Decisions with Uncertainty"

### Focus

For the next breakthroughs after RaBitQ / ANN, the most important thing is estimation theory:

```text
estimator
unbiasedness
bias
variance
MSE
confidence interval
lower bound
upper bound
bias-variance tradeoff
```

You should especially accept one viewpoint:

```text
unbiasedness is not necessarily best for ANN.
ANN cares about the top-k decision,
not whether every distance is unbiased.
```

### What to Focus On

```text
MSE = bias^2 + variance
confidence interval
control variates
importance sampling
empirical Bernstein bound
variance reduction
ranking confidence
```

### Significance for ANN

Traditional estimation:

```text
estimate d(q, x)
```

More advanced questions:

```text
does x belong to the top-k?
is x1 necessarily closer than x2?
does this candidate need rerank?
can this point be safely pruned?
```

### Capability to Reach

Given two candidates:

```text
d1 ∈ [l1, u1]
d2 ∈ [l2, u2]
```

You can determine:

```text
u1 < l2  →  x1 stably ranks ahead of x2
intervals overlap → ranking uncertain
```

This is the move from distance-aware to decision-aware.

### 📚 Recommended

- **All of Statistics — Larry Wasserman**
- **Statistical Inference — Casella & Berger**
- **The Elements of Statistical Learning — Hastie, Tibshirani, Friedman**
- **Understanding Machine Learning — Shalev-Shwartz & Ben-David**

---

## 6. Quantization and Information Theory: Understanding the Boundaries of bit, distortion, and rate

### Focus

You already understand ANN, so this part should be studied in a more "research-oriented" way:

```text
scalar quantization
vector quantization
product quantization
residual quantization
binary quantization
rate-distortion
entropy
mutual information
compression bound
```

Don't just ask:

```text
is the compression accurate or not?
```

Ask:

```text
does this quantizer optimize reconstruction error?
distance estimation error?
inner product error?
ranking error?
attention error?
system cost?
```

### Core Distinctions

```text
PQ / OPQ:
leans more toward reconstruction / codebook / data-dependent.

RaBitQ:
leans more toward randomized distance estimation / distribution-free bound.

LLM quantization:
leans more toward preserving matrix multiplication / model output.

KV cache quantization:
leans more toward preserving attention logits / softmax / output distribution.
```

### Capability to Reach

When you see a compression algorithm, you can determine:

```text
what is the objective function?
how is the bit budget used?
what is the theoretical bound?
is there data dependence?
is the error pointwise or ranking-level?
can it be transferred to attention?
```

### 📚 Recommended

- **Elements of Information Theory — Cover & Thomas**
- **Information Theory, Inference, and Learning Algorithms — David MacKay**
- **Vector Quantization and Signal Compression — Gersho & Gray**
- **Network Information Theory — El Gamal & Kim**

---

## 7. Ranking, top-k, decision theory: The Core Math for the Next ANN Breakthroughs

### Focus

The next breakthroughs in ANN are not necessarily about estimating each distance more accurately, but about making decisions more reliably:

```text
top-k membership
pairwise ranking
candidate pruning
rerank decision
early stopping
search path decision
```

This requires:

```text
order statistics
ranking loss
pairwise comparison
margin
top-k error
confidence bound for differences
sequential decision
multi-armed bandit intuition
```

### Research Intuition

```text
if two candidates differ greatly, a coarse estimate is enough.
if two candidates are very close, a finer estimate is needed.
```

So a better algorithm might be:

```text
rather than making all distances more accurate on average,
spend the compute budget near the top-k boundary.
```

### Capability to Reach

You can rewrite ANN search as:

```text
under a fixed bit / memory / latency budget,
maximize top-k decision correctness.
```

rather than:

```text
minimize the average distance estimation error over all points.
```

### 📚 Recommended

- **Learning to Rank for Information Retrieval — Tie-Yan Liu**
- **Bandit Algorithms — Lattimore & Szepesvári**
- **Prediction, Learning, and Games — Cesa-Bianchi & Lugosi**
- **The Elements of Statistical Learning — chapters related to ranking / classification**

---

## 8. Random Matrices and randomized linear algebra: Connecting ANN and Large Models

### Focus

Many bottlenecks in large models are matrix problems:

```text
QK^T
softmax(QK^T)V
Wx
low-rank adaptation
KV cache
MoE routing
long-context attention
```

Many techniques in ANN can be transferred to these matrix computations, but you need randomized linear algebra as the bridge.

### What to Master

```text
randomized SVD
low-rank approximation
matrix concentration
approximate matrix multiplication
subspace embedding
spectral norm bound
Frobenius norm bound
operator norm perturbation
```

### Significance for Large Models

You can analyze:

```text
after compressing K, how large is the QK^T error?
after compressing V, how large is the attention output error?
after approximating attention, how much does the softmax distribution change?
how do low-rank/sparse/sketch affect the layer output?
```

### 📚 Recommended

- **Randomized Algorithms for Matrices and Data — Mahoney**
- **Finding Structure with Randomness — Halko, Martinsson, Tropp**
- **Matrix Concentration Inequalities — Tropp**
- **High-Dimensional Probability — Vershynin**

---

## 9. Transformer / attention Math: Viewing Large Models as Vector Retrieval Systems

### Focus

You should re-examine the transformer from an ANN perspective:

```text
Q = query
K = database keys
V = payload values
QK^T = inner product search
softmax = normalized retrieval weight
KV cache = dynamic vector memory
```

### Must Know

```text
self-attention
cross-attention
multi-head attention
causal mask
RoPE
KV cache
prefill / decode
FlashAttention
sparse attention
linear attention
MoE routing
```

### Important Questions

```text
is attention a kind of differentiable retrieval?
can the KV cache be viewed as a dynamic vector database?
can token pruning become safe top-k attention?
can an inner product bound yield a softmax bound?
```

### The Research Directions Most Relevant to You

```text
ANN-style KV cache compression
attention as approximate MIPS
quantized attention logits
softmax-safe pruning
VLA memory retrieval
trajectory memory retrieval
```

### 📚 Recommended

- **The Annotated Transformer**
- **Deep Learning — Goodfellow, Bengio, Courville**
- **Understanding Deep Learning — Simon Prince**
- **FlashAttention papers**
- **vLLM / PagedAttention paper**
- **Systems papers related to TensorRT-LLM / DeepSpeed-Inference / FasterTransformer**

---

## 10. Graph Search and Irregular Memory Access: The Math/Systems Substrate for the Next ANN Graph Breakthroughs

### Focus

The bottleneck of Graph ANN is not just distance computation, but:

```text
random memory access
cache miss
pointer chasing
unpredictable traversal
low batching efficiency
GPU-unfriendly access pattern
```

What you need to learn is not just graph theory, but:

```text
graph structure + memory layout + traversal cost + hardware behavior
```

### What to Master

```text
small-world graph
navigability
nearest-neighbor graph
degree distribution
graph expansion
search path length
random walk intuition
cache locality
layout optimization
prefetching
batched traversal
```

### Research Questions

```text
how to preserve the navigability of the graph while reducing random access?
how to make graph search more batch-friendly?
how to make the quantization code layout and the graph edge layout work together?
how to use a performance model to predict graph search latency?
```

### 📚 Recommended

- **Networks, Crowds, and Markets — Easley & Kleinberg**
- **Graph Representation Learning — Hamilton**
- **Mining of Massive Datasets**
- **Papers related to HNSW / NSG / DiskANN / FAISS / ScaNN / SPANN / SymphonyQG**

---

## 11. HPC / AI systems Performance Modeling: The Core of the Hoefler Style

### What Does the "Mathematical Optimization" Hoefler Talks About Roughly Mean

The "mathematical optimization" here is neither the optimizer in ordinary machine learning, nor merely convex optimization.

In the Hoefler / SPCL / HPC context, it is closer to:

```text
Performance as a Science
```

That is:

```text
use mathematical models to describe:
    algorithms
    applications
    hardware
    communication
    storage
    parallel execution

then optimize:
    latency
    throughput
    scalability
    energy
    memory footprint
    communication volume
    load balance
```

So the math here is not a single field, but a combination of several classes of models:

```text
1. Cost model
2. Roofline model
3. Communication model
4. Memory hierarchy model
5. Parallel scalability model
6. Queueing / scheduling model
7. Optimization under constraints
8. Algorithm-hardware co-design
```

### What You Need to Master

```text
roofline model
arithmetic intensity
memory bandwidth model
latency model
throughput model
Amdahl's law
Gustafson's law
Little's law
queueing theory
communication complexity
collective communication cost
load balancing
critical path
work-span model
```

### Significance for ANN

Graph ANN can be modeled as:

```text
T_search =
    #random_access × cost_cache_miss
  + #distance_computation × cost_distance
  + #raw_vector_read × cost_raw_read
  + #branching / traversal overhead
```

Quantized ANN can be modeled as:

```text
T =
    compressed_code_read
  + approximate_distance_compute
  + rerank_cost
  + raw_vector_access
```

Distributed ANN / RAG can be modeled as:

```text
T =
    local_search_time
  + network_routing_time
  + top-k_merge_time
  + communication_volume / bandwidth
```

### Significance for LLM/VLA

LLM decode can be modeled as:

```text
T_decode =
    KV_cache_read_time
  + attention_compute_time
  + MLP_compute_time
  + synchronization_time
```

VLA inference can be modeled as:

```text
T_vla =
    T_vision_encoder
  + T_prefill
  + T_decode
  + T_action_head
  + T_control_loop
  + T_network
```

### The Capability You Should Reach

Not just reporting benchmarks, but being able to answer:

```text
is the bottleneck compute?
memory bandwidth?
cache miss?
PCIe/NVLink?
network communication?
synchronization?
batching?
or is the algorithm's own access pattern simply too poor?
```

And going further to write out:

```text
if bits are halved,
how much does memory traffic drop?
how much does distance compute drop?
how much does recall/rerank increase?
does end-to-end latency actually go down?
```

This is Hoefler-style performance science.

### 📚 Recommended

- **Computer Architecture: A Quantitative Approach — Hennessy & Patterson**
- **The Datacenter as a Computer — Barroso, Hölzle, Ranganathan**
- **Performance Modeling and Design of Computer Systems — Mor Harchol-Balter**
- **Introduction to High Performance Computing for Scientists and Engineers — Hager & Wellein**
- **Using the Roofline Model to Understand Performance — Williams et al.**
- **Hoefler/SPCL papers on performance modeling, communication, distributed DL, benchmarking**
- **FlashAttention / vLLM / TensorRT-LLM / DeepSpeed-Inference papers**

---

## 12. Mathematical Optimization: Not Grinding Convex Optimization, but Learning "co-design Under Constraints"

### Focus

You should view optimization as:

```text
under accuracy / memory / latency / throughput / energy / hardware constraints,
choose algorithm + data layout + precision + parallel strategy.
```

This is more important than just SGD/Adam.

### What to Master

```text
constrained optimization
Lagrangian
KKT conditions
multi-objective optimization
Pareto frontier
integer optimization intuition
dynamic programming
convex relaxation
Bayesian optimization
design space exploration
cost-based optimization
```

### Typical Forms

ANN:

```text
minimize latency
subject to recall ≥ target
           memory ≤ budget
           build time ≤ budget
```

Quantization:

```text
minimize ranking error
subject to bit budget ≤ B
```

LLM inference:

```text
maximize throughput
subject to P99 latency ≤ target
           memory ≤ GPU capacity
```

VLA:

```text
minimize control latency
subject to action quality ≥ target
           power ≤ edge device budget
```

Distributed AI:

```text
minimize step time
subject to GPU memory, network bandwidth, parallelism constraints
```

### The Most Important Intuitions for You

```text
algorithm-optimal ≠ system-optimal
minimal distance error ≠ best top-k
fewest bits ≠ lowest latency
fewest FLOPs ≠ fastest on GPU
low theoretical complexity ≠ good cache behavior
```

### 📚 Recommended

- **Convex Optimization — Boyd & Vandenberghe**
- **Numerical Optimization — Nocedal & Wright**
- **Algorithms for Optimization — Kochenderfer & Wheeler**
- **Computer Architecture: A Quantitative Approach — Hennessy & Patterson**
- **Materials related to database query optimization / cost-based optimization**

---

## 13. Multimodal and VLA: Unifying memory, attention, and action into Vector Problems

### Focus

VLA is not just a "robotics application"; it essentially contains many high-dimensional vector problems:

```text
vision token selection
language-conditioned retrieval
state-action embedding
trajectory memory
action chunking
diffusion action head
online memory update
real-time inference
```

### The Math You Need

```text
multimodal embedding alignment
contrastive learning
trajectory representation
sequence similarity
state-action space
continuous control
diffusion model basics
behavior cloning
RL basics
```

### Entry Points from an ANN Background

```text
KV cache = dynamic vector database
trajectory memory = sequence ANN
action codebook = vector quantization
visual token pruning = approximate top-k selection
cross-attention = inner product retrieval
```

### 📚 Recommended

- **Reinforcement Learning: An Introduction — Sutton & Barto**
- **Algorithms for Decision Making — Kochenderfer et al.**
- **Diffusion Policy paper**
- **RT-1 / RT-2 / OpenVLA / π0 / GR00T / VLA-Perf papers**
- **CLIP / ViT / multimodal representation learning papers**

---

## 14. A Learning Order Better Suited to You

You are not a beginner, so linearly grinding through books is not recommended. It is suggested to fill in by research capability.

### Phase A: High-Dimensional Algorithm Math

```text
High-Dimensional Probability
Random Projection / Sketching
Quantization / Rate-Distortion
Estimator / Confidence Bound
Ranking / Top-k Decision
```

Resulting capability:

```text
be able to read and improve the theory of ANN / RaBitQ / quantized search.
```

### Phase B: From ANN to attention

```text
Inner product estimation
Softmax perturbation
Randomized matrix multiplication
KV cache compression
Approximate attention
```

Resulting capability:

```text
be able to transfer ANN's bound / quantization ideas to LLM attention.
```

### Phase C: From algorithm to performance science

```text
Roofline
Memory hierarchy
Communication model
Queueing / scheduling
Parallel scalability
Cost-based optimization
```

Resulting capability:

```text
be able, like Hoefler/SPCL, to put algorithms and hardware performance into the same mathematical model.
```

### Phase D: From LLM to VLA

```text
VLA inference pipeline
Vision token cost
Action head cost
Trajectory memory
Control latency
Device-server split inference
```

Resulting capability:

```text
be able to find the high-dimensional vector bottlenecks in VLA systems and propose ANN/quantization-style solutions.
```

---

## 15. The Research Perspective You Should Ultimately Form

```text
high-dimensional vector algorithm perspective:
    what is the geometric structure?
    how does the error concentrate?
    how are the bits allocated?
    is the top-k decision reliable?

AI model perspective:
    what is attention retrieving?
    how does the KV cache grow?
    how sensitive is softmax to error?
    how are action token/trajectory represented?

systems perspective:
    is the bottleneck compute, memory, or communication?
    is the access pattern regular?
    can it be batched / prefetched / compressed?
    where does P99 latency come from?

HPC/Hoefler perspective:
    is there an interpretable performance model?
    is the mapping between algorithm and hardware optimal?
    can scalability be predicted through a mathematical model?
    is this optimization just a benchmark trick, or a principled design?
```

---

## 16. The Most Core Reading List

### High-Dimensional Probability / Randomized Algorithms

- **High-Dimensional Probability — Vershynin**
- **Probability and Computing — Mitzenmacher & Upfal**
- **Foundations of Data Science — Blum, Hopcroft, Kannan**
- **Mining of Massive Datasets — Leskovec, Rajaraman, Ullman**

### Linear Algebra / Random Matrices

- **Matrix Computations — Golub & Van Loan**
- **Numerical Linear Algebra — Trefethen & Bau**
- **Randomized Algorithms for Matrices and Data — Mahoney**
- **Finding Structure with Randomness — Halko, Martinsson, Tropp**

### Statistics / Estimation / Learning Theory

- **All of Statistics — Wasserman**
- **Understanding Machine Learning — Shalev-Shwartz & Ben-David**
- **The Elements of Statistical Learning — Hastie, Tibshirani, Friedman**

### Quantization / Information Theory

- **Elements of Information Theory — Cover & Thomas**
- **Information Theory, Inference, and Learning Algorithms — MacKay**
- **Vector Quantization and Signal Compression — Gersho & Gray**

### Optimization / co-design

- **Convex Optimization — Boyd & Vandenberghe**
- **Numerical Optimization — Nocedal & Wright**
- **Algorithms for Optimization — Kochenderfer & Wheeler**

### AI systems / HPC / performance

- **Computer Architecture: A Quantitative Approach — Hennessy & Patterson**
- **The Datacenter as a Computer — Barroso, Hölzle, Ranganathan**
- **Performance Modeling and Design of Computer Systems — Harchol-Balter**
- **Introduction to High Performance Computing for Scientists and Engineers — Hager & Wellein**
- **Hoefler/SPCL papers on performance modeling, distributed DL, communication, benchmarking**

### LLM / VLA

- **Deep Learning — Goodfellow, Bengio, Courville**
- **Understanding Deep Learning — Simon Prince**
- **FlashAttention / vLLM / PagedAttention / TensorRT-LLM papers**
- **RT-1 / RT-2 / OpenVLA / π0 / Diffusion Policy / VLA-Perf papers**

---

## 17. The One-Sentence Version

The math you need to fill in is not traditional "advanced calculus," but rather:

```text
high-dimensional probability + geometry
    to understand vector spaces and random approximation;

estimation theory + ranking decision
    to move from distance estimation toward top-k decisions;

information theory + quantization
    to understand the bit budget and error limits;

random matrices + attention math
    to transfer ANN ideas to LLM/KV cache;

performance models + mathematical optimization
    to model and co-design algorithms, systems, and hardware together, like Hoefler.
```
