# High-Dimensional Vectors / AI / Large Models / HPC Mathematics Roadmap — Resource Library

This directory is the companion resource library for the [`00-roadmap.md`](00-roadmap.md) study roadmap.
The subfolders are organized by the roadmap's 13 chapters; each section holds **legally and freely obtainable** electronic editions
(author homepages / publisher open access / arXiv). For **copyrighted books with no free edition**, this index only lists
links for legitimate purchase/borrowing — no pirated copies are downloaded.

> Already downloaded **43 PDFs (about 314 MB)**. Each section below lists: ✅ downloaded · 🔁 cross-chapter reference · 💰 needs purchase/borrowing.

---

## 01 · Linear Algebra / Matrices · `01-linear-algebra/`
- ✅ Axler — *Linear Algebra Done Right* (4th ed, open access)
- ✅ Deisenroth, Faisal, Ong — *Mathematics for Machine Learning*
- 💰 Trefethen & Bau — *Numerical Linear Algebra* (SIAM)
- 💰 Golub & Van Loan — *Matrix Computations* (JHU Press)

## 02 · High-Dimensional Probability · `02-high-dim-probability/`
- ✅ Vershynin — *High-Dimensional Probability* (2nd ed draft, author homepage)
- ✅ Blum, Hopcroft, Kannan — *Foundations of Data Science*
- 💰 Mitzenmacher & Upfal — *Probability and Computing* (Cambridge)
- 💰 Blitzstein & Hwang — *Introduction to Probability* (the official free site probabilitybook.net currently has downloads disabled; the authors may restore it at any time)

## 03 · High-Dimensional Geometry · `03-high-dim-geometry/`
- 🔁 See Vershynin from 02 (the core is concentration of measure) and Blum/Hopcroft/Kannan
- 💰 Zezula et al. — *Similarity Search: The Metric Space Approach* (Springer)
- 💰 Ledoux — *The Concentration of Measure Phenomenon* (AMS)

## 04 · Random Projection / LSH / Sketching · `04-sketching-lsh/`
- ✅ Leskovec, Rajaraman, Ullman — *Mining of Massive Datasets*
- ✅ Woodruff — *Sketching as a Tool for Numerical Linear Algebra*
- 🔁 See Mahoney and Halko-Martinsson-Tropp from 08 (randomized matrices = the other side of sketching)

## 05 · Estimation Theory / Statistics / Learning Theory · `05-estimation-stats/`
- ✅ Shalev-Shwartz & Ben-David — *Understanding Machine Learning*
- ✅ Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*
- 💰 Wasserman — *All of Statistics* (Springer)
- 💰 Casella & Berger — *Statistical Inference* (Cengage)

## 06 · Information Theory / Quantization · `06-info-theory-quantization/`
- ✅ MacKay — *Information Theory, Inference, and Learning Algorithms*
- ✅ El Gamal & Kim — *Lecture Notes on Network Information Theory* (arXiv 1001.3404 v4, full 640-page version)
- ✅ Gao & Long — *RaBitQ* (arXiv, directly relevant to your direction)
- 💰 Cover & Thomas — *Elements of Information Theory* (Wiley)
- 💰 Gersho & Gray — *Vector Quantization and Signal Compression* (Springer)

## 07 · Ranking / top-k / Decision · `07-ranking-decision/`
- ✅ Lattimore & Szepesvári — *Bandit Algorithms*
- 💰 Tie-Yan Liu — *Learning to Rank for Information Retrieval* (Now Publishers / Springer, no free edition)
- 💰 Cesa-Bianchi & Lugosi — *Prediction, Learning, and Games* (Cambridge)

## 08 · Random Matrices / Randomized Linear Algebra · `08-randomized-linalg/`
- ✅ Mahoney — *Randomized Algorithms for Matrices and Data*
- ✅ Halko, Martinsson, Tropp — *Finding Structure with Randomness*
- ✅ Tropp — *Matrix Concentration Inequalities*
- 🔁 See Vershynin from 02

## 09 · Transformer / Attention · `09-transformer-attention/`
- ✅ Vaswani et al. — *Attention Is All You Need*
- ✅ Dao et al. — *FlashAttention* / *FlashAttention-2* / *FlashAttention-3*
- ✅ Kwon et al. — *vLLM / PagedAttention*
- ✅ Su et al. — *RoFormer (RoPE)*
- ✅ Prince — *Understanding Deep Learning* (general DL textbook, free edition from the author)
- 🔗 *The Annotated Transformer* (web page): https://nlp.seas.harvard.edu/annotated-transformer/
- 💰 Goodfellow, Bengio, Courville — *Deep Learning* (officially free only as a web edition: https://www.deeplearningbook.org/ )

## 10 · Graph Search / Irregular Memory Access · `10-graph-search/`
- ✅ Malkov & Yashunin — *HNSW*
- ✅ Fu et al. — *NSG*
- ✅ Subramanya et al. — *DiskANN (Rand-NSG)*
- ✅ Chen et al. — *SPANN*
- ✅ Guo et al. — *ScaNN (Anisotropic Vector Quantization)*
- ✅ Easley & Kleinberg — *Networks, Crowds, and Markets*
- ✅ Hamilton — *Graph Representation Learning*
- 🔁 The categories one level up in `../` contain the latest SIGMOD graph-index/quantization papers (SymphonyQG, FAISS family, etc.)

## 11 · HPC / AI Systems Performance Modeling · `11-hpc-performance-modeling/`
- ✅ Williams, Waterman, Patterson — *Roofline Model* (Berkeley technical report version)
- ✅ Hoefler & Belli — *Scientific Benchmarking of Parallel Computing Systems*
- ✅ Ben-Nun & Hoefler — *Demystifying Parallel and Distributed Deep Learning*
- ✅ Barroso, Hölzle, Ranganathan — *The Datacenter as a Computer* (4th ed, CC-BY)
- 💰 Hennessy & Patterson — *Computer Architecture: A Quantitative Approach*
- 💰 Harchol-Balter — *Performance Modeling and Design of Computer Systems* (Cambridge)
- 💰 Hager & Wellein — *Introduction to HPC for Scientists and Engineers* (CRC)
- 🔗 More Hoefler/SPCL papers: https://spcl.inf.ethz.ch/Publications/

## 12 · Mathematical Optimization / Co-design · `12-optimization/`
- ✅ Boyd & Vandenberghe — *Convex Optimization*
- ✅ Kochenderfer & Wheeler — *Algorithms for Optimization*
- 💰 Nocedal & Wright — *Numerical Optimization* (Springer)

## 13 · Multimodal / VLA · `13-vla-multimodal/`
- ✅ Sutton & Barto — *Reinforcement Learning: An Introduction*
- ✅ Kochenderfer et al. — *Algorithms for Decision Making*
- ✅ Chi et al. — *Diffusion Policy*
- ✅ Brohan et al. — *RT-1* / *RT-2*
- ✅ Kim et al. — *OpenVLA*
- ✅ Black et al. — *π0*
- ✅ Radford et al. — *CLIP*
- ✅ Dosovitskiy et al. — *ViT*

---

## Suggested Study Order (from roadmap §14)
- **Phase A — High-dimensional algorithmic mathematics**: 02 → 04 → 06 → 05 → 07
- **Phase B — From ANN to attention**: 08 → 09
- **Phase C — From algorithms to performance science**: 11 → 12
- **Phase D — From LLMs to VLA**: 13

## Sourcing Notes
All PDFs come from legal, free channels: author homepages, publisher open access (CC licenses), institutional repositories, or arXiv.
Copyrighted books were not downloaded; they are only marked 💰 above with legitimate access routes given. If you need one of them, go through a university library / legitimate purchase.
