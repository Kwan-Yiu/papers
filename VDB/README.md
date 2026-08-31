# VDB — Vector Database / ANNS Paper Library

Papers on approximate nearest-neighbor search and vector-database systems. Organization/naming
follows the repo convention in [`../README.md`](../README.md): theme folders hold the PDFs,
file names are `[VENUE'YEAR] SHORTNAME.pdf`.

> **82 papers across 8 themes.** Math & systems study path: [`ROADMAP/README.md`](ROADMAP/README.md).
> **🆕 = added from the SIGMOD 2026 or VLDB 2026 program.** Titles below are the authoritative titles.

---

## BENCH · Benchmarks & experimental studies · `BENCH/`
- 🆕 [SIGMOD'2026] **ATTRFILTERSTUDY** — Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study
- 🆕 [VLDB'2026] **FILTERBENCH** — Revisiting Filtered ANN Benchmarks: A Hardness-Controlled Benchmark Generator for Realistic Evaluation
- [SIGMOD'2025] **GRAPHBASEDVECTORSEARCH** — Graph-Based Vector Search: An Experimental Evaluation of the State-of-the-Art
- 🆕 [VLDB'2026] **HYBRIDBLEND** — Balancing the Blend: An Experimental Analysis of Trade-offs in Hybrid Search
- 🆕 [VLDB'2026] **HYBRIDQUERY** — An Experimental Evaluation of Hybrid Querying on Vectors
- [SIGMOD'2026] **PITFALLS** — Reveal Hidden Pitfalls and Navigate the Next Generation of Vector Similarity Search from Task-Centric Views

## FILTER · Attribute / label-filtered ANN · `FILTER/`
- 🆕 [VLDB'2026] **CGIF** — CGIF: Combining Proximity Graphs and Inverted Files for Efficient Filtered Vector Search over Arbitrary Predicates
- 🆕 [VLDB'2026] **ELASTICAKNN** — Elastic Index Selection for Label-Hybrid AKNN Search
- 🆕 [SIGMOD'2026] **FAVOR** — FAVOR: Efficient Filter-Agnostic Vector ANNS Based on Selectivity-Aware Exclusion Distances
- 🆕 [VLDB'2026] **GAS** — GAS: A Lightweight Framework for Filtered Search over Wide-table Vectors
- 🆕 [VLDB'2026] **HARMONIZE** — Harmonizing Efficiency and Accuracy in Filtered Vector Search
- [SIGMOD'2024] **NAVIGATINGLABELSVECTORS** — Navigating Labels and Vectors: A Unified Approach to Filtered Approximate Nearest Neighbor Search
- 🆕 [SIGMOD'2026] **PGFILTERSTUDY** — An In-Depth Study of Filter-Agnostic Vector Search on a PostgreSQL Database System
- [SIGMOD'2025] **RWALKS** — RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search

## GRAPH · Graph / proximity-graph indexes · `GRAPH/`
- [SIGMOD'2025] **ACCELERATINGHIGH** — Accelerating High-Dimensional ANN Search via Skipping Redundant Distance Computations
- 🆕 [VLDB'2026] **ANNIE** — ANNiE: A Learned Query Cost Estimator for Graph-Based Approximate Nearest Neighbor Search
- 🆕 [VLDB'2026] **BBC** — BBC: Improving Large-k Approximate Nearest Neighbor Search with a Bucket-based Result Collector
- 🆕 [VLDB'2026] **CONDA** — CONDA: A Connectivity-Aware Dynamic Index for Approximate Nearest Neighbor Search over Evolving Data
- [SIGMOD'2026] **DAE-HNSW** — Distribution-Aware Exploration for Adaptive HNSW Search
- [SIGMOD'2025] **DDFH** — Dynamically Detect and Fix Hardness for Efficient Approximate Nearest Neighbor Search
- [SIGMOD'2025] **DEG** — DEG: Efficient Hybrid Vector Search Using the Dynamic Edge Navigation Graph
- [SIGMOD'2025] **DIGRA** — DIGRA: A Dynamic Graph Indexing for Approximate Nearest Neighbor Search with Range Filter
- [VLDB'2025] **FASTKCNA** — Revisiting the Index Construction of Proximity Graph-Based Approximate Nearest Neighbor Search
- [SIGMOD'2026] **FCPG** — Fast-Convergent Proximity Graphs for Approximate Nearest Neighbor Search
- 🆕 [SIGMOD'2026] **FGIM** — FGIM: a Fast Graph-based Indexes Merging Framework for Approximate Nearest Neighbor Search
- 🆕 [SIGMOD'2026] **GEM** — GEM: A Native Graph-based Index for Multi-Vector Retrieval
- 🆕 [VLDB'2026] **HEXA** — HEXA: A Disjoint-Subgraph-Based Indexing Framework for Approximate Nearest Neighbor Search at Billion Scale
- [SIGMOD'2026] **LMG** — Graph Based K-Nearest Neighbor Search Revisited
- [SIGMOD'2026] **MIRAGE-ANNS** — MIRAGE-ANNS: Mixed Approach Graph-based Indexing for Approximate Nearest Neighbor Search
- 🆕 [SIGMOD'2026] **PGTUNER** — PGTuner: An Efficient Framework for Automatic and Transferable Configuration Tuning of Proximity Graphs
- 🆕 [VLDB'2026] **SNGREVISITED** — Sparse Neighborhood Graph-Based Approximate Nearest Neighbor Search Revisited: Theoretical Analysis and Optimization
- 🆕 [VLDB'2026] **TOPOUPDATE** — A Topology-Aware Localized Update Strategy for Graph-Based ANN Index

## HARDWARE · GPU / disk / accelerators · `HARDWARE/`
- [VLDB'2024] **CHAMELEON** — Chameleon: A Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Generation
- [VLDB'2025] **COTRA** — Towards Efficient and Scalable Distributed Vector Search
- 🆕 [VLDB'2026] **DISKGRAPHIO** — I/O Optimizations in Graph-Based Disk-Resident Approximate Nearest Neighbor Search: A Design Space Exploration
- [VLDB'2025] **FALCON** — Falcon: Fast Graph Vector Search via Hardware Acceleration and Delayed-Synchronization Traversal
- [SIGMOD'2026] **FLASHANNS** — FlashANNS: GPU-Driven I/O Pipelining for Eliminating Storage-Compute Bottlenecks in Billion-Scale Similarity Search
- 🆕 [SIGMOD'2026] **GPS** — GPS: Revisiting the Data Layout for Disk-based High-Dimensional Vector Search
- 🆕 [VLDB'2026] **GPUANNS** — GPU-Accelerated ANNS: Quantized for Speed, Built for Change
- [SIGMOD'2025] **GRAPHINDEXINGCPU** — Accelerating Graph Indexing for ANNS on Modern CPUs
- 🆕 [VLDB'2026] **IVFRABITQ** — GPU-Native Approximate Nearest Neighbor Search with IVF-RaBitQ: Fast Index Build and Search
- [SIGMOD'2025] **PDX** — PDX: A Data Layout for Vector Similarity Search
- 🆕 [VLDB'2026] **REDANNS** — RED-ANNS: An RDMA-Enabled Distributed Framework for Graph-Based Approximate Nearest Neighbor Search
- [SIGMOD'2026] **SGI-GPU** — Scalable Graph Indexing using GPUs for Approximate Nearest Neighbor Search
- ⚠️ [VLDB'2024] **SINGLEGPU** — *intended:* High-Throughput, Cost-Effective Billion-Scale Vector Search with a Single GPU — **the stored PDF is the wrong file (a solar-physics paper); needs re-download.**
- 🆕 [VLDB'2026] **SVFUSION** — SVFusion: A CPU-GPU Co-Processing Architecture for Large-Scale Real-Time Vector Search

## OPERATOR · Joins / aggregate / predicate operators · `OPERATOR/`
- [SIGMOD'2025] **BEYONDVECTORSEAR** — Beyond Vector Search: Querying With and Without Predicates
- 🆕 [VLDB'2026] **CONANN** — ConANN: Conformal Approximate Nearest Neighbor Search
- [SIGMOD'2026] **CURATOR** — Curator: Efficient Vector Search with Low-Selectivity Filters
- [SIGMOD'2025] **DARTH** — DARTH: Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search
- [SIGMOD'2025] **DISKJOIN** — DiskJoin: Large-scale Vector Similarity Join with SSD
- [SIGMOD'2026] **EAA-NNQ** — On Efficient Approximate Aggregate Nearest Neighbor Queries over Learned Representations
- [SIGMOD'2025] **FASTSIMJOIN** — Fast Approximate Similarity Join in Vector Databases
- [SIGMOD'2025] **WOW** — WoW: A Window-to-Window Incremental Index for Range-Filtering Approximate Nearest Neighbor Search

## QUANT · Quantization / compression · `QUANT/`
- 🆕 [VLDB'2026] **JHQ** — JHQ: Johnson-Lindenstrauss Enhanced Hierarchical Quantization for High-Dimensional Approximate Nearest Neighbor Search
- [SIGMOD'2025] **OPTIMALQUANTIZATION** — Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for ANN Search
- 🆕 [VLDB'2026] **QUANTPROJ** — Quantization Meets Projection: A Happy Marriage for Approximate k-Nearest Neighbor Search
- [SIGMOD'2026] **RAIRS** — RAIRS: Optimizing Redundant Assignment and List Layout for IVF-Based ANN Search
- [SIGMOD'2025] **SAQ** — SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation
- [SIGMOD'2025] **SYMPHONYQG** — SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search
- [VLDB'2025] **TRIM** — TRIM: Accelerating High-Dimensional Vector Similarity Search with Enhanced Triangle-Inequality-Based Pruning

## RANGE · Range-filtered ANN · `RANGE/`
- [SIGMOD'2025] **DYNAMICFILTEREDANNS** — Efficient Dynamic Indexing for Range Filtered Approximate Nearest Neighbor Search
- [SIGMOD'2024] **IRANGEGRAPH** — iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search
- 🆕 [VLDB'2026] **RNSG** — RNSG: A Range-Aware Graph Index for Efficient Range-Filtered Approximate Nearest Neighbor Search

## SYSTEM · End-to-end VDB systems · `SYSTEM/`
- 🆕 [VLDB'2026] **AKER** — Aker: Density-Aware Approximate Caching for Vector Search
- 🆕 [VLDB'2026] **FEDBRIDGE** — FedBridge: A Federated Query Engine over Embedding-Heterogeneous Vector Databases *(demo)*
- [arXiv'2025] **HARMONY** — HARMONY: A Scalable Distributed Vector Database for High-Throughput Approximate Nearest Neighbor Search
- [arXiv'2025] **HONEYBEE** — Honeybee: Efficient Role-based Access Control for Vector Databases via Dynamic Partitioning
- 🆕 [SIGMOD'2026] **INDEXLAYOUT** — AlayaLaser: Efficient Index Layout and Search Strategy for Large-scale High-dimensional Vector Similarity Search
- 🆕 [SIGMOD'2026] **INDEXMERGE** — Efficient Vector Index Merging in Vector Databases
- 🆕 [SIGMOD'2026] **INTEGRATINGVDB** — Integrating Vector Databases across Embedding Models *(Best Paper Honorable Mention)*
- 🆕 [VLDB'2026] **NOVA** — Nova: A Multi-Purpose Vector Engine for Low-Latency, Multi-Tenant, and Cross-Table Hybrid Retrieval *(industry)*
- 🆕 [VLDB'2026] **PRESTOVEC** — SQL-Native Vector Search at Billion Scale in Presto *(industry)*
- 🆕 [VLDB'2026] **QBAT** — QBAT: Model-based Query Budget Autotuner for Clustering-based Approximate Nearest Neighbor Search
- 🆕 [SIGMOD'2026] **REPSAMPLING** — Balancing Global and Local: Representative Sampling for Large-Scale Vector Data
- 🆕 [SIGMOD'2026] **SERVERLESSVDB** — Building Stateless Serverless Vector DBs via Block-based Data Partitioning
- [SIGMOD'2025] **SUBSPACECOLLISION** — Subspace Collision: An Efficient and Accurate Framework for High-dimensional Approximate Nearest Neighbor Search
- 🆕 [SIGMOD'2026] **TACO** — TaCo: Data-adaptive and Query-aware Subspace Collision for High-dimensional Approximate Nearest Neighbor Search
- 🆕 [VLDB'2026] **TENGINEDBV** — TEngineDB-V: An OLAP-Native Vector Search System for Large-k Workloads at Tencent *(industry)*
- [SIGMOD'2025] **TRIBASE** — Tribase: A Vector Data Query Engine for Reliable and Lossless Pruning Compression using Triangle Inequalities
- 🆕 [VLDB'2026] **V3DB** — V3DB: Audit-on-Demand Zero-Knowledge Proofs for Verifiable Vector Search over Committed Snapshots
- [SIGMOD'2025] **VECFLOW** — VecFlow: A High-Performance Vector Data Management System for Filtered-Search on GPUs

---

## Study path
`ROADMAP/` is a companion math + systems reading path (high-dimensional probability, sketching/LSH,
quantization & rate-distortion, randomized linear algebra, attention, performance modeling…),
with downloaded textbooks/papers organized by topic. See [`ROADMAP/README.md`](ROADMAP/README.md).

## Sources
Most papers are SIGMOD / PACMMOD (2024–2026) or PVLDB / VLDB (2024–2026); a few are arXiv preprints.
The 🆕 entries were added from the SIGMOD 2026 or VLDB 2026 programs.
Naming convention: `<THEME>/[VENUE'YEAR] <NAME>.pdf`.
