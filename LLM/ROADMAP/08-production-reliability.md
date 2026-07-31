# Production Reliability and Operations — Curated Reading Map

> **Role:** production layer above inference engines and distributed runtimes
> **Target:** connect SLOs, observability, overload control, routing, autoscaling, deployment, recovery, isolation, and cost to LLM-specific state and metrics
> **Format:** external English books, official documentation, operational guides, and GitHub source/configuration paths
> **Not included:** an original SRE or Kubernetes tutorial, a calendar, or a generic cloud certification path

---

## Scope

```text
client
  |
gateway / authentication / quota
  |
inference-aware router and admission control
  |
model pool / prefill pool / decode pool
  |
engine scheduler / KV state / model runner
  |
GPU, network, storage
  |
metrics + traces + logs + events
  |
autoscaler + operator + deployment controller
```

Resource labels:

- **Core** — required;
- **Branch** — follow when it matches the deployment or research question;
- **Reference** — consult on demand;
- **Local clone** — present under `../RESOURCES/repos/`;
- **Local PDF** — present in this repository;
- **Link** — keep remote unless source work requires a clone.

---

## Coverage Checklist

### Reliability contract

- [ ] SLI, SLO, SLA, error budget, availability, correctness, and durability;
- [ ] TTFT, inter-token latency/TPOT, end-to-end latency, goodput, throughput, and queue delay;
- [ ] request classes by model, prompt/output length, priority, tenant, and cache reuse;
- [ ] percentile distributions and coordinated-omission awareness.

### Control plane and data plane

- [ ] gateway, endpoint picker, router, discovery, operator, autoscaler, and model loader;
- [ ] admission, quota, backpressure, cancellation, retry, timeout, and load shedding;
- [ ] replica routing, KV-aware routing, prefix-aware routing, and P/D routing;
- [ ] cold start, model download, weight loading, compilation, graph capture, and readiness.

### Operations

- [ ] metrics, traces, logs, profiles, events, dashboards, and alerts;
- [ ] pod/process/GPU/network/storage failure;
- [ ] request and KV-state recovery semantics;
- [ ] rolling update, canary, rollback, compatibility, and supply chain;
- [ ] multi-tenancy, isolation, security, cost, energy, and capacity.

---

## 1. SRE Foundation

### 1.1 Google SRE reading path

Primary source: [Site Reliability Engineering — Table of Contents](https://sre.google/sre-book/table-of-contents/)

| Order | Chapter | Why it is core for LLM serving |
|---:|---|---|
| 1 | [Chapter 3 — Embracing Risk](https://sre.google/sre-book/embracing-risk/) | availability versus engineering cost |
| 2 | [Chapter 4 — Service Level Objectives](https://sre.google/sre-book/service-level-objectives/) | SLI/SLO/error-budget vocabulary |
| 3 | [Chapter 6 — Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/) | latency, traffic, errors, saturation |
| 4 | [Chapter 10 — Practical Alerting](https://sre.google/sre-book/practical-alerting/) | symptom-based alerts and paging |
| 5 | [Chapter 17 — Testing for Reliability](https://sre.google/sre-book/testing-reliability/) | fault and recovery validation |
| 6 | [Chapter 19 — Load Balancing at the Frontend](https://sre.google/sre-book/load-balancing-frontend/) | external request distribution |
| 7 | [Chapter 20 — Load Balancing in the Datacenter](https://sre.google/sre-book/load-balancing-datacenter/) | backend selection |
| 8 | [Chapter 21 — Handling Overload](https://sre.google/sre-book/handling-overload/) | admission, queueing, load shedding |
| 9 | [Chapter 22 — Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/) | retries, deadlines, overload amplification |
| 10 | [Chapter 27 — Reliable Product Launches](https://sre.google/sre-book/reliable-product-launches/) | launch criteria |

Branch:

- [The Site Reliability Workbook](https://sre.google/workbook/table-of-contents/) — SLO implementation, alerting, canaries, overload, and incident response;
- [Google SRE — Distributed Systems Observability](https://sre.google/sre-book/monitoring-distributed-systems/) — keep as the primary conceptual reference even when using other telemetry tools.

### 1.2 LLM-specific SLO sources

| Priority | Resource | Format | Read for |
|---|---|---|---|
| Core | [DistServe](../SERVING/DISTSERVE.pdf) | Local PDF | TTFT/TPOT SLOs and prefill/decode interference |
| Core | [Orca](../SERVING/ORCA.pdf) | Local PDF | iteration scheduling and serving objectives |
| Core | [vLLM](../SERVING/VLLM.pdf) | Local PDF | memory utilization and serving throughput |
| Core | [Sarathi-Serve](../SERVING/SARATHI.pdf) | Local PDF | chunked prefill and latency/throughput trade-off |
| Branch | [Splitwise](../SERVING/SPLITWISE.pdf) | Local PDF | phase-specific resource provisioning |
| Branch | [MLSysBook Volume 2](../COURSE/MLSYS-VOL2.pdf) | Local PDF | inference at scale, reliability, security, sustainability |

Required service-contract artifact:

- endpoint and model;
- supported input/output constraints;
- correctness definition;
- TTFT, TPOT/inter-token, end-to-end, and availability SLIs;
- target percentiles and measurement window;
- overload behavior;
- cancellation and retry semantics;
- request classes excluded from the objective.

Do not use one aggregate latency percentile for all model sizes and prompt/output distributions.

---

## 2. Observability

### 2.1 OpenTelemetry path

| Priority | Resource | Format | Exact reading target |
|---|---|---|---|
| Core | [OpenTelemetry Concepts](https://opentelemetry.io/docs/concepts/) | official docs | observability primer, context propagation, components |
| Core | [Signals](https://opentelemetry.io/docs/concepts/signals/) | official docs | traces, metrics, logs, baggage, profiles |
| Core | [Instrumentation](https://opentelemetry.io/docs/concepts/instrumentation/) | official docs | code-based and zero-code instrumentation |
| Core | [Sampling](https://opentelemetry.io/docs/concepts/sampling/) | official docs | head/tail sampling and trace cost |
| Core | [Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/) | official specification | stable attribute naming |
| Reference | [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/) | official docs | receivers, processors, exporters, deployment |

Choose the language-specific getting-started guide matching the serving frontend. Do not copy a language guide unrelated to the deployed stack.

### 2.2 Prometheus and histogram path

| Priority | Resource | Read for |
|---|---|---|
| Core | [Prometheus Instrumentation Practices](https://prometheus.io/docs/practices/instrumentation/) | metric naming, online-serving patterns, failures |
| Core | [Histograms and Summaries](https://prometheus.io/docs/practices/histograms/) | bucket choice and quantiles |
| Core | [Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/) | alert expression and `for` behavior |
| Branch | [Grafana — Exemplars](https://grafana.com/docs/grafana/latest/fundamentals/exemplars/) | jump from latency bucket to request trace |

### 2.3 Engine metrics

| Engine/platform | External source | Local source anchor |
|---|---|---|
| vLLM | [vLLM production metrics](https://docs.vllm.ai/en/stable/usage/metrics/) | search `metrics.py` and scheduler/KV metric emitters in `../RESOURCES/repos/vllm/` |
| SGLang | [SGLang observability](https://docs.sglang.ai/advanced_features/observability.html) | scheduler metric components in `../RESOURCES/repos/sglang/python/sglang/srt/managers/scheduler_components/` |
| Dynamo | [Dynamo metrics](https://docs.nvidia.com/dynamo/latest/user-guides/observability-local/metrics) | `../RESOURCES/repos/dynamo/deploy/observability/` |
| Kubernetes GPU | [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) | deployment chart and metric list |

### 2.4 Required span and metric map

Trace spans:

1. gateway receive;
2. admission/quota;
3. router decision;
4. engine enqueue;
5. queue wait;
6. prefill;
7. KV transfer or cache lookup;
8. decode iterations;
9. detokenization/stream write;
10. cancellation or finish.

Metric groups:

- request: arrival, rejection, cancellation, status, TTFT, TPOT, end-to-end;
- queue: depth, age, token backlog, admitted/rejected tokens;
- scheduler: batch size, scheduled prefill/decode tokens, preemption;
- KV: capacity, use, allocation failure, hit rate, eviction, transfer bytes/time;
- model runner: forward latency, graph hit/miss, compilation;
- GPU: utilization, HBM, power, temperature, ECC/Xid;
- network: collectives, RDMA/TCP transfer, retransmit/error;
- process: CPU, RSS, event-loop lag, file descriptors;
- deployment: ready replicas, cold-start stage, rollout version.

Every label must be reviewed for cardinality. Raw prompt, user ID, request ID, and unbounded model input must not become metric labels.

---

## 3. Kubernetes and GPU Platform Basics

### 3.1 Official Kubernetes path

| Priority | Resource | Exact reading target |
|---|---|---|
| Core | [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) | rollout, revision, rollback, strategy |
| Core | [StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/) | stable identity/storage when required |
| Core | [Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#container-probes) | distinct health semantics |
| Core | [Resource Management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/) | requests, limits, extended resources |
| Core | [Assigning Pods to Nodes](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/) | affinity and topology placement |
| Core | [Pod Disruptions](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/) | voluntary disruption and budgets |
| Core | [Autoscaling Workloads](https://kubernetes.io/docs/concepts/workloads/autoscaling/) | HPA, VPA, in-place resize, event-driven options |
| Core | [Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/) | control loop, custom metrics, stabilization |
| Branch | [Resource Quotas](https://kubernetes.io/docs/concepts/policy/resource-quotas/) | tenant and namespace quotas |
| Branch | [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/) | network isolation |

### 3.2 GPU and topology branches

| Priority | Resource | Read for |
|---|---|---|
| Core | [NVIDIA Kubernetes Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) | GPU discovery, allocation, sharing options |
| Core | [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/) | driver/runtime/device-plugin/DCGM lifecycle |
| Branch | [NVIDIA Network Operator](https://docs.nvidia.com/networking/display/cokan10/network+operator) | RDMA and network-device provisioning |
| Branch | [Kueue](https://kueue.sigs.k8s.io/) | queued batch/gang resource admission |
| Branch | [Grove](https://github.com/NVIDIA/grove) | topology-aware gang scheduling for inference graphs |

Required platform trace:

- deployment or custom resource;
- operator reconciliation;
- pod scheduling and GPU allocation;
- model artifact acquisition;
- engine startup;
- compilation/graph warmup;
- startup probe;
- readiness transition;
- endpoint registration;
- first admitted request.

---

## 4. Inference-Aware Gateway and Routing

### 4.1 Gateway API Inference Extension

| Priority | Resource | Format | Read / inspect |
|---|---|---|---|
| Core | [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) | official docs | concepts, `InferencePool`, endpoint picker, request flow |
| Core | [kubernetes-sigs/gateway-api-inference-extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) | source repo | `docs/`, API types, EPP implementation, examples |
| Core | [InferencePool API](https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/) | official API guide | pool membership, endpoint picker, metrics, and failure mode |

Record:

- which metadata is available at routing time;
- which state belongs in the gateway, EPP, or engine;
- how endpoints publish load;
- how stale load information is handled;
- what happens on timeout, cancellation, endpoint removal, and retry.

### 4.2 Routing policies to compare

| Policy | External implementation/reference | Evidence needed |
|---|---|---|
| round-robin / random | Kubernetes/gateway baseline | imbalance under variable request size |
| least-request / load-aware | Gateway EPP or platform router | load signal freshness and definition |
| prefix/KV-aware | llm-d precise-prefix guide; Dynamo KV router | hit rate, load skew, routing overhead |
| predicted-latency | llm-d predicted-latency guide | predictor features, error, stability |
| P/D-aware | llm-d and Dynamo disaggregation guides | prefill/decode queue state and transfer cost |
| model-aware | multi-model router | load/unload and cache residency |
| tenant/priority-aware | flow-control guide | isolation and starvation |

---

## 5. Production Serving Platforms

### 5.1 llm-d

Repository: [llm-d/llm-d](https://github.com/llm-d/llm-d)
Local clone: `../RESOURCES/repos/llm-d/`

Recommended source order:

| Order | Local path | Read for |
|---:|---|---|
| 1 | `docs/architecture/README.md` | component and request-flow overview |
| 2 | `docs/getting-started/` | deployment object and baseline path |
| 3 | `docs/operations/router.md` | router operation |
| 4 | `docs/operations/readiness-probes.md` | model-server readiness |
| 5 | `docs/api-reference/inferencepool.md` | inference-pool API |
| 6 | `docs/api-reference/endpointpickerconfig.md` | endpoint-picker configuration |
| 7 | `guides/optimized-baseline/` | controlled baseline |
| 8 | `guides/precise-prefix-cache-routing/` | prefix-aware routing |
| 9 | `guides/predicted-latency-routing/` | latency-aware routing |
| 10 | `guides/flow-control/` | admission and overload |
| 11 | `guides/workload-autoscaling/` | HPA/KEDA/SLO-aware variants |
| 12 | `guides/pd-disaggregation/` | P/D architecture and deployment |
| 13 | `guides/tiered-prefix-cache/` | external/tiered cache |
| 14 | `guides/wide-ep-lws/` | large-scale expert parallelism |
| 15 | `.github/scripts/e2e/` | end-to-end validation logic |

Treat the `guides/` workloads and configuration as versioned operational references, not universal defaults.

### 5.2 NVIDIA Dynamo

Repository: [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)
Local clone: `../RESOURCES/repos/dynamo/`
Official docs: [Introduction to Dynamo](https://docs.nvidia.com/dynamo/getting-started/introduction)

External documentation path:

| Order | Resource | Read for |
|---:|---|---|
| 1 | [Introduction](https://docs.nvidia.com/dynamo/getting-started/introduction) | frontend, router, workers, discovery, Kubernetes path |
| 2 | [Feature Guides](https://docs.nvidia.com/dynamo/latest/user-guides) | KV routing, disaggregation, offload, benchmarking |
| 3 | [Disaggregated Serving](https://docs.nvidia.com/dynamo/latest/design-docs/disaggregated-serving) | prefill/decode flow and transfer |
| 4 | [Kubernetes Observability](https://docs.nvidia.com/dynamo/dev/kubernetes/operations/observability) | metrics, traces, logs, health |
| 5 | fault-tolerance and deployment guides from the current docs index | recovery and Kubernetes objects |

Local source/configuration path:

- `components/README.md`;
- `deploy/operator/`;
- `deploy/inference-gateway/`;
- `deploy/observability/`;
- `recipes/README.md` and one model-matched recipe;
- `benchmarks/`;
- backend examples under `examples/backends/`;
- component implementations under `components/` and `lib/`.

Pin docs and repository revisions together; fast-moving platform documentation may describe a newer component layout than the local clone.

### 5.3 AIBrix, KServe, and comparative platforms

| Priority | Resource | Focus |
|---|---|---|
| Branch | [vllm-project/aibrix](https://github.com/vllm-project/aibrix) | Kubernetes-native LLM gateway, routing, autoscaling, model management |
| Branch | [AIBrix documentation](https://aibrix.readthedocs.io/) | architecture, gateway, autoscaler, distributed inference |
| Branch | [kserve/kserve](https://github.com/kserve/kserve) | inference service CRDs, runtimes, autoscaling, canary |
| Branch | [KServe generative inference](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) | LLM serving integration |
| Reference | [Ray Serve LLM](https://docs.ray.io/en/latest/serve/llm/index.html) | Python-native serving and autoscaling |

Use these as comparisons after tracing either llm-d or Dynamo end to end.

---

## 6. Autoscaling and Capacity

### 6.1 Reading path

| Priority | Resource | Read for |
|---|---|---|
| Core | [Kubernetes HPA](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/) | feedback loop, custom metrics, stabilization |
| Core | [llm-d workload autoscaling guides](https://github.com/llm-d/llm-d/tree/main/guides/workload-autoscaling) | inference-pool signals and variants |
| Branch | [KEDA](https://keda.sh/docs/) | event/queue-driven scaling |
| Branch | [AIBrix autoscaler design](https://aibrix.readthedocs.io/latest/designs/aibrix-autoscaler.html) | LLM-specific autoscaling controller |
| Branch | Dynamo planner/autoscaling guides from the current docs | P/D pool planning |

### 6.2 Signals to evaluate

- request rate;
- input-token and output-token arrival rate;
- queue depth and oldest-request age;
- predicted GPU service time;
- scheduled-token utilization;
- KV capacity and eviction pressure;
- TTFT or TPOT objective violation;
- prefill/decode pool imbalance;
- model-load and cold-start duration.

Required capacity artifact:

- workload distribution, not only mean QPS;
- model/precision/parallelism;
- steady-state per-replica capacity;
- cold-start stage breakdown;
- scale-up and scale-down policy;
- hysteresis/stabilization;
- failure and retry load;
- overload fallback;
- cost per successful token/request.

---

## 7. Failure, Recovery, and Deployment

### 7.1 Reliability reading

| Priority | Resource | Read for |
|---|---|---|
| Core | [Google SRE — Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/) | retries, deadlines, overload, failover |
| Core | [Google SRE — Testing for Reliability](https://sre.google/sre-book/testing-reliability/) | fault validation |
| Core | [Kubernetes Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/) | termination, restart, readiness |
| Core | [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) | rollout and rollback |
| Branch | [Dynamo fault-tolerance guide](https://docs.nvidia.com/dynamo/latest/user-guides/fault-tolerance) | inference-component recovery |

### 7.2 Failure matrix

For every row, trace detection, containment, recovery, state loss, client-visible result, and retry safety:

| Failure | State at risk |
|---|---|
| API/gateway process | connection and stream state |
| router/EPP | routing decision and load view |
| tokenizer/frontend | request preprocessing |
| engine core | queue, scheduler, request state |
| worker process | active model step and local KV |
| GPU reset/Xid/ECC | model weights, KV, communicator |
| NCCL/RDMA/network partition | collective or KV transfer |
| prefill worker | generated KV and handoff metadata |
| decode worker | active sequence and output stream |
| model artifact store | startup and scale-out |
| operator/control plane | desired/observed deployment state |
| telemetry pipeline | detection and diagnosis |

### 7.3 Deployment artifacts

Required:

- compatibility matrix for model, tokenizer, engine, kernels, driver, CUDA/ROCm, GPU;
- immutable image and model revisions;
- startup/readiness/liveness definitions;
- canary traffic and rollback conditions;
- request draining and termination grace;
- cache/weight warmup policy;
- configuration and feature-flag provenance;
- postmortem template tied to SLI impact.

---

## 8. Multi-Tenancy, Security, and Supply Chain

| Priority | Resource | Read for |
|---|---|---|
| Core | [Kubernetes Multi-tenancy](https://kubernetes.io/docs/concepts/security/multi-tenancy/) | namespace and cluster isolation models |
| Core | [Kubernetes Resource Quotas](https://kubernetes.io/docs/concepts/policy/resource-quotas/) | resource control |
| Core | [Kubernetes Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/) | traffic isolation |
| Core | [MLSysBook Volume 2](../COURSE/MLSYS-VOL2.pdf) | ML-system security, privacy, responsible operation |
| Branch | [SLSA](https://slsa.dev/) | software supply-chain levels and provenance |
| Branch | [Sigstore documentation](https://docs.sigstore.dev/) | signing and verification |
| Reference | [Kubernetes Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/) | workload hardening |

LLM-specific review:

- prompt/output logging and retention;
- tokenizer and chat-template revision;
- model artifact trust and remote code;
- tenant-specific adapters and cache keys;
- prefix-cache leakage;
- timing and resource side channels;
- quota by requests, tokens, KV bytes, and GPU time;
- cancellation and billing correctness.

---

## 9. Cost, Energy, and Capacity References

| Priority | Resource | Read for |
|---|---|---|
| Core | [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) | compute, memory, communication capacity |
| Core | [MLSysBook](https://mlsysbook.ai/) | sustainable AI and system lifecycle |
| Branch | [CodeCarbon](https://github.com/mlco2/codecarbon) | energy/emission measurement method |
| Branch | [Kepler](https://github.com/sustainable-computing-io/kepler) | Kubernetes energy telemetry |
| Reference | cloud/provider GPU pricing pages for the deployed region | current monetary cost; record access date |

Report:

- successful output tokens per GPU-second;
- good requests per GPU-hour;
- energy per successful token/request;
- model weight and KV memory utilization;
- reserved versus used capacity;
- cold-start amortization;
- failed/retried/cancelled work;
- cost at the stated SLO, not maximum unconstrained throughput.

---

## 10. Required Evidence

Produce:

1. a service contract with LLM-specific SLIs/SLOs;
2. a control-plane/data-plane component diagram;
3. a request trace spanning gateway, router, engine, GPU, and stream;
4. a metrics dictionary with type, unit, labels, cardinality, and owner;
5. an OpenTelemetry span schema;
6. an overload policy with admission, queue limit, deadline, shedding, and retry rules;
7. a cold-start breakdown;
8. a routing comparison under variable prompt/output lengths and cache reuse;
9. an autoscaling control-loop diagram and stability criteria;
10. a failure/recovery matrix;
11. a rollout/rollback checklist;
12. a cost and capacity model at an explicit SLO.

Each artifact must point to the relevant external source and the implementation/configuration path being described.

---

## 11. Repository Index

| Repository | Role | Starting path | Status |
|---|---|---|---|
| [harvard-edge/cs249r_book](https://github.com/harvard-edge/cs249r_book) | broad ML systems textbook | Volume 2 deployment/reliability/security chapters | Link |
| [llm-d/llm-d](https://github.com/llm-d/llm-d) | Kubernetes-native distributed inference | `docs/`, `guides/`, e2e scripts | Local clone |
| [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) | distributed inference platform | `components/`, `deploy/`, `recipes/`, `benchmarks/` | Local clone |
| [kubernetes-sigs/gateway-api-inference-extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) | inference-aware gateway APIs | docs, API, endpoint picker | Link |
| [vllm-project/aibrix](https://github.com/vllm-project/aibrix) | LLM gateway, routing, autoscaling | docs, controllers, gateway, autoscaler | Link |
| [kserve/kserve](https://github.com/kserve/kserve) | model-serving control plane | docs, CRDs, runtimes, controllers | Link |
| [open-telemetry/opentelemetry-collector](https://github.com/open-telemetry/opentelemetry-collector) | telemetry pipeline | docs, receiver/processor/exporter interfaces | Link |
| [prometheus/prometheus](https://github.com/prometheus/prometheus) | metrics and alerting | documentation before source | Link |
| [NVIDIA/dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) | GPU metrics | metric configuration and deployment | Link |
| [NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) | GPU allocation | deployment, config, device-plugin source | Link |

---

## 12. What to Defer

Defer unless the chosen deployment needs it:

- service meshes unrelated to the request path;
- every Kubernetes controller and storage class;
- generic cloud certifications;
- multi-region active-active deployment;
- billing platform implementation;
- compliance frameworks without a stated deployment requirement.

---

## Exit Gate

Continue to [09-bottleneck-research.md](09-bottleneck-research.md) when:

- [ ] the production request path and ownership boundaries are source-grounded;
- [ ] TTFT, TPOT/inter-token, end-to-end latency, throughput, and goodput are defined for explicit request classes;
- [ ] metrics, traces, logs, and alerts cover gateway through GPU;
- [ ] overload, cancellation, timeout, retry, and shedding behavior are explicit;
- [ ] routing and autoscaling policies are tied to measurable engine/KV state;
- [ ] cold-start and rollout stages are measured;
- [ ] representative failures have detection, containment, recovery, and client-visible semantics;
- [ ] multi-tenant cache and resource isolation are reviewed;
- [ ] cost is reported at an explicit SLO;
- [ ] all platform behavior is tied to a pinned documentation/source revision.

The gate is an auditable production model, not familiarity with every cloud-native tool.
