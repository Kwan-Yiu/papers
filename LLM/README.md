# LLM — Large-Model Inference Acceleration Papers

Inference-acceleration papers for text LLMs (not VLA/VLM), a sibling of [`../VLA`](../VLA)
(Vision-Language-Action) and [`../VDB`](../VDB) (vector database / ANNS).
Organization and naming follow this repo's [`../README.md`](../README.md) convention:
PDFs live under theme folders, named with short uppercase abbreviations.

> **30 PDFs across one theme so far (`SPEC`).** Room to grow: `CACHE` (KV cache), `QUANT`,
> `SERVING` (scheduling), `MOE`, `ATTENTION`.
> Systems-focused study path: [`ROADMAP/00-roadmap.md`](ROADMAP/00-roadmap.md).

---

## SPEC · speculative decoding · [`SPEC/`](SPEC/README.md)
The full speculative-decoding lineage in 30 papers, grouped into eight sub-themes plus a survey
(FOUNDATION, MULTITOKEN, EAGLE, TREE, SELFSPEC, SYSTEM, LONGCONTEXT, DIFFUSION). See [`SPEC/README.md`](SPEC/README.md).

---

## Relation to VLA/SPEC
`LLM/SPEC` holds the general-purpose methods; `VLA/SPEC` (SPECVLA, CEEDVLA) ports speculative
decoding onto robot VLA models — the former is the upstream of the latter.

## Sources
All PDFs are from arXiv. Every ID was verified against its page-1 title before download.
