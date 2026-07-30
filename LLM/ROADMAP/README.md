# Roadmap Index

- [`00-roadmap.md`](00-roadmap.md) — complete LLM inference / AI systems research roadmap;
- [`01-competency-plan.md`](01-competency-plan.md) — prerequisite graph, artifacts, and competency gates;
- [`02-research-projects.md`](02-research-projects.md) — systems-research project ladder;
- [`03-architecture-taxonomy.md`](03-architecture-taxonomy.md) — architecture components, model families, and systems consequences;
- [`04-moe-deep-dive.md`](04-moe-deep-dive.md) — MoE routing → grouped GEMM → EP communication → serving;
- [`05-inference-systems-taxonomy.md`](05-inference-systems-taxonomy.md) — request, scheduler, state, kernel, distributed, and operations stack;
- [`06-bottleneck-research-map.md`](06-bottleneck-research-map.md) — bottleneck matrix, 15 research directions, and evidence ladder.

## How to use

1. Read Sections 1–3 of `00`, then start with the first unmet gate in `01`.
2. Use `03` when reading a model; do not memorize model names without cost implications.
3. Read `04` end to end before selecting an MoE topic.
4. Use `05` to locate the system layer of each serving repository or paper.
5. Use `06` and `02` to move from bottleneck → hypothesis → experiment.

Do not wait until every mathematical topic is complete before running experiments.
