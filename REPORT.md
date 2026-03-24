# Huawei Internship Task Report

## 1) Task Scope

The assignment contains two parts:
1. **Task 1**: Read and summarize the SGLang paper and be ready to discuss it.
2. **Task 2**: Simulate a multimodal LLM inference server on the **full Azure trace** and find the minimal number of accelerators `N` that satisfies:
   - `TTFT <= P` for stages 1+2,
   - `T <= D` for generation completion time.

Workload source:
- [AzureLMMInferenceTrace_multimodal.csv.gz](https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLMMInferenceTrace_multimodal.csv.gz)


## 2) Task 1: SGLang Summary

Paper:
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/html/2312.07104v2)

Main takeaway:
- SGLang is a language-runtime co-design for structured LLM applications.

Key technical ideas:
- **RadixAttention** for prefix/KV reuse across requests and calls.
- **Compressed FSM decoding** for efficient constrained generation.
- **Speculative execution** for API-model workflows.

Practical impact:
- Better throughput and latency on structured/multimodal pipelines through cache reuse, decoding efficiency, and improved orchestration.


## 3) Task 2: Simulation Model

Implementation:
- `./simulate_server.py`

### Stage model
- Stage 1: image preprocessing (`NumImages`) with cost coefficient `A` and memory footprint `X`.
- Stage 2: context/prefill (`ContextTokens`) with cost coefficient `B` and footprint `Y`.
- Stage 3: generation (`GeneratedTokens`) with cost coefficient `C` and footprint `Z`.

### Hardware model
- `N` homogeneous accelerators.
- Per-accelerator memory: `M`.
- Per-accelerator compute capability: `K`.
- Each request stays on one accelerator across all stages.

### Batching model
- Stage duration for batch size `U`:
  - `time = coeff * U^p / K`
  - default `p = 0.5` (`sqrt`), configurable via CLI.
- Memory remains linear in total footprints.
- Stage-specific batch limits are configurable (max requests and max units).

### SLA definitions
- `TTFT = finish_stage2 - arrival`.
- `T = finish_stage3 - arrival`.
- Queue waiting time is included in both metrics.


## 4) Engineering Decisions

### Event-driven simulator
- Uses arrival and stage-completion events instead of fixed time stepping.
- Keeps simulation efficient and precise for 1,000,000 requests.

### Assignment policy
- Default: `balanced` (least projected finish time) for better load distribution.
- Alternative: `round-robin` for comparison baselines.

### Stage scheduling policy
- Default: `edf` (earliest-deadline-first) using request deadlines.
- Alternative: `slack` mode (minimum timing slack).

### Correctness controls
- Per-stage memory safety checks.
- Fast infeasibility pre-checks for single-request memory peaks.
- SLA checks at stage boundaries.
- Numerical tolerance handling for floating-point memory accounting.

### Output quality and reproducibility
- TTFT/T include `min/avg/median/p90/p95/p99/max`.
- Per-accelerator telemetry: utilization, stage times, stage units, queue peaks, memory peak.
- Search trace logs all tested `N` values and reasons for infeasibility.
- Metadata includes trace SHA256, runtime environment, and command line.


## 5) Full-Trace Results (1,000,000 requests)

Output file:
- `./simulation_results_full.json`

Configuration:
- `A=B=C=1.0`
- `X=0.0008`, `Y=0.00035`, `Z=0.0002`
- `M=80.0`, `K=120.0`
- `P=5.0s`, `D=12.0s`
- `cost_power=0.5` (`sqrt`)
- `assignment=balanced`
- `scheduler=edf`

Result:
- **Minimal feasible accelerators: `N = 5`**

Statistics:
- **TTFT**: min `0.0000`, avg `0.3703`, median `0.2863`, p95 `0.8650`, p99 `1.7713`, max `4.2543`
- **T**: min `0.0083`, avg `0.4765`, median `0.3717`, p95 `1.0369`, p99 `1.8463`, max `9.1675`

Search boundary:
- `N=4` is infeasible (TTFT violation).
- `N=5` is feasible.


## 6) Model Limitations

- The assignment does not specify fixed values for `A,B,C,X,Y,Z,M,K,P,D`, so all are configurable.
- This is an abstract capacity/SLA simulator, not a hardware-accurate profiler.
- Low-level GPU effects (kernel scheduling details, allocator fragmentation, interconnect effects) are intentionally abstracted out.
