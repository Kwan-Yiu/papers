# VDB — Vector Database / ANNS Paper Library

Papers on approximate nearest-neighbor search and vector-database systems. Organization/naming
follows the repo convention in [`../README.md`](../README.md): theme folders hold the PDFs,
file names use short uppercase abbreviations.

> **53 papers across 8 themes.** Math & systems study path: [`ROADMAP/README.md`](ROADMAP/README.md).
> **🆕 = added from the SIGMOD 2026 program.** Titles below are the authoritative titles (verified via Crossref / page 1).

---

## BENCH · Benchmarks & experimental studies · `BENCH/`
- 🆕 **ATTRFILTERSTUDY** — Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study
- **GRAPHBASEDVECTORSEARCH** — Graph-Based Vector Search: An Experimental Evaluation of the State-of-the-Art
- **PITFALLS** — Reveal Hidden Pitfalls and Navigate the Next Generation of Vector Similarity Search from Task-Centric Views

## FILTER · Attribute / label-filtered ANN · `FILTER/`
- 🆕 **FAVOR** — FAVOR: Efficient Filter-Agnostic Vector ANNS Based on Selectivity-Aware Exclusion Distances
- **NAVIGATINGLABELSVECTORS** — Navigating Labels and Vectors: A Unified Approach to Filtered Approximate Nearest Neighbor Search
- 🆕 **PGFILTERSTUDY** — An In-Depth Study of Filter-Agnostic Vector Search on a PostgreSQL Database System
- **RWALKS** — RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search

## GRAPH · Graph / proximity-graph indexes · `GRAPH/`
- **ACCELERATINGHIGH** — Accelerating High-Dimensional ANN Search via Skipping Redundant Distance Computations
- **DAE-HNSW** — Distribution-Aware Exploration for Adaptive HNSW Search
- **DDFH** — Dynamically Detect and Fix Hardness for Efficient Approximate Nearest Neighbor Search
- **DEG** — DEG: Efficient Hybrid Vector Search Using the Dynamic Edge Navigation Graph
- **DIGRA** — DIGRA: A Dynamic Graph Indexing for Approximate Nearest Neighbor Search with Range Filter
- **FASTKCNA** — Revisiting the Index Construction of Proximity Graph-Based Approximate Nearest Neighbor Search
- **FCPG** — Fast-Convergent Proximity Graphs for Approximate Nearest Neighbor Search
- 🆕 **FGIM** — FGIM: a Fast Graph-based Indexes Merging Framework for Approximate Nearest Neighbor Search
- 🆕 **GEM** — GEM: A Native Graph-based Index for Multi-Vector Retrieval
- **LMG** — Graph Based K-Nearest Neighbor Search Revisited
- **MIRAGE-ANNS** — MIRAGE-ANNS: Mixed Approach Graph-based Indexing for Approximate Nearest Neighbor Search
- 🆕 **PGTUNER** — PGTuner: An Efficient Framework for Automatic and Transferable Configuration Tuning of Proximity Graphs

## HARDWARE · GPU / disk / accelerators · `HARDWARE/`
- **CHAMELEON** — Chameleon: A Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Generation
- **COTRA** — Towards Efficient and Scalable Distributed Vector Search
- **FALCON** — Falcon: Fast Graph Vector Search via Hardware Acceleration and Delayed-Synchronization Traversal
- **FLASHANNS** — FlashANNS: GPU-Driven I/O Pipelining for Eliminating Storage-Compute Bottlenecks in Billion-Scale Similarity Search
- 🆕 **GPS** — GPS: Revisiting the Data Layout for Disk-based High-Dimensional Vector Search
- **GRAPHINDEXINGCPU** — Accelerating Graph Indexing for ANNS on Modern CPUs
- **PDX** — PDX: A Data Layout for Vector Similarity Search
- **SGI-GPU** — Scalable Graph Indexing using GPUs for Approximate Nearest Neighbor Search
- ⚠️ **SINGLEGPU** — *intended:* High-Throughput, Cost-Effective Billion-Scale Vector Search with a Single GPU — **the stored PDF is the wrong file (a solar-physics paper); needs re-download.**

## OPERATOR · Joins / aggregate / predicate operators · `OPERATOR/`
- **BEYONDVECTORSEAR** — Beyond Vector Search: Querying With and Without Predicates
- **CURATOR** — Curator: Efficient Vector Search with Low-Selectivity Filters
- **DARTH** — DARTH: Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search
- **DISKJOIN** — DiskJoin: Large-scale Vector Similarity Join with SSD
- **EAA-NNQ** — On Efficient Approximate Aggregate Nearest Neighbor Queries over Learned Representations
- **FASTSIMJOIN** — Fast Approximate Similarity Join in Vector Databases
- **WOW** — WoW: A Window-to-Window Incremental Index for Range-Filtering Approximate Nearest Neighbor Search

## QUANT · Quantization / compression · `QUANT/`
- **OPTIMALQUANTIZATION** — Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for ANN Search
- **RAIRS** — RAIRS: Optimizing Redundant Assignment and List Layout for IVF-Based ANN Search
- **SAQ** — SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation
- **SYMPHONYQG** — SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search
- **TRIM** — TRIM: Accelerating High-Dimensional Vector Similarity Search with Enhanced Triangle-Inequality-Based Pruning

## RANGE · Range-filtered ANN · `RANGE/`
- **DYNAMICFILTEREDANNS** — Efficient Dynamic Indexing for Range Filtered Approximate Nearest Neighbor Search
- **IRANGEGRAPH** — iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search

## SYSTEM · End-to-end VDB systems · `SYSTEM/`
- **HARMONY** — HARMONY: A Scalable Distributed Vector Database for High-Throughput Approximate Nearest Neighbor Search
- **HONEYBEE** — Honeybee: Efficient Role-based Access Control for Vector Databases via Dynamic Partitioning
- 🆕 **INDEXLAYOUT** — AlayaLaser: Efficient Index Layout and Search Strategy for Large-scale High-dimensional Vector Similarity Search
- 🆕 **INDEXMERGE** — Efficient Vector Index Merging in Vector Databases
- 🆕 **INTEGRATINGVDB** — Integrating Vector Databases across Embedding Models *(Best Paper Honorable Mention)*
- 🆕 **REPSAMPLING** — Balancing Global and Local: Representative Sampling for Large-Scale Vector Data
- 🆕 **SERVERLESSVDB** — Building Stateless Serverless Vector DBs via Block-based Data Partitioning
- **SUBSPACECOLLISION** — Subspace Collision: An Efficient and Accurate Framework for High-dimensional Approximate Nearest Neighbor Search
- 🆕 **TACO** — TaCo: Data-adaptive and Query-aware Subspace Collision for High-dimensional Approximate Nearest Neighbor Search
- **TRIBASE** — Tribase: A Vector Data Query Engine for Reliable and Lossless Pruning Compression using Triangle Inequalities
- **VECFLOW** — VecFlow: A High-Performance Vector Data Management System for Filtered-Search on GPUs

---

## Study path
`ROADMAP/` is a companion math + systems reading path (high-dimensional probability, sketching/LSH,
quantization & rate-distortion, randomized linear algebra, attention, performance modeling…),
with downloaded textbooks/papers organized by topic. See [`ROADMAP/README.md`](ROADMAP/README.md).

## Sources
Most papers are SIGMOD / PACMMOD (2025–2026); a few are arXiv preprints. The 🆕 entries were added
from the SIGMOD 2026 program. Naming convention: `<THEME>/<NAME>.pdf`.
