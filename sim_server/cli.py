"""CLI entrypoint for the multimodal inference simulator."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .engine import find_min_accelerators, summarize, telemetry_to_dict
from .io_ops import compute_trace_sha256, load_trace
from .models import (
    AssignmentPolicy,
    BatchPolicy,
    CostParams,
    Limits,
    MemoryParams,
    SchedulerPolicy,
    SimulationConfig,
)
from .validators import (
    non_negative_float,
    non_negative_int,
    parse_cost_power,
    positive_float,
    positive_int,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Simulate multimodal LLM inference server and find minimal accelerator count"
    )
    parser.add_argument(
        "--trace-path",
        default="AzureLMMInferenceTrace_multimodal.csv.gz",
        help="Path to Azure multimodal trace CSV.GZ",
    )
    parser.add_argument(
        "--max-requests",
        type=positive_int,
        default=None,
        help="Optional cap on request count for debug runs",
    )
    parser.add_argument("--min-n", type=positive_int, default=1, help="Lower bound for N search")
    parser.add_argument("--max-n", type=positive_int, default=2048, help="Upper bound for N search")
    parser.add_argument(
        "--assignment",
        choices=["balanced", "round-robin"],
        default="balanced",
        help="Request-to-accelerator assignment policy",
    )
    parser.add_argument(
        "--scheduler",
        choices=["edf", "slack"],
        default="edf",
        help="Stage scheduling policy: earliest-deadline-first or minimum slack",
    )

    # Cost and hardware parameters.
    parser.add_argument("--A", type=non_negative_float, default=1.0, help="Stage1 image cost")
    parser.add_argument("--B", type=non_negative_float, default=1.0, help="Stage2 context cost")
    parser.add_argument("--C", type=non_negative_float, default=1.0, help="Stage3 generation cost")
    parser.add_argument(
        "--cost-shape",
        type=parse_cost_power,
        default=0.5,
        help="Batch cost exponent: sqrt|cuberoot|power:<p>|<float>",
    )
    parser.add_argument(
        "--K",
        type=positive_float,
        default=120.0,
        help="Compute capability (abstract units/sec)",
    )
    parser.add_argument("--X", type=non_negative_float, default=0.0008, help="Memory per image")
    parser.add_argument("--Y", type=non_negative_float, default=0.00035, help="Memory per context token")
    parser.add_argument("--Z", type=non_negative_float, default=0.0002, help="Memory per generated token")
    parser.add_argument("--M", type=positive_float, default=80.0, help="Memory per accelerator")

    # SLO limits.
    parser.add_argument("--P", type=positive_float, default=5.0, help="TTFT limit (seconds)")
    parser.add_argument("--D", type=positive_float, default=12.0, help="Per-request token limit T (seconds)")

    # Batch controls.
    parser.add_argument("--stage1-max-requests", type=positive_int, default=64)
    parser.add_argument("--stage2-max-requests", type=positive_int, default=64)
    parser.add_argument("--stage3-max-requests", type=positive_int, default=64)
    parser.add_argument("--stage1-max-units", type=non_negative_int, default=2048, help="0 means unlimited")
    parser.add_argument("--stage2-max-units", type=non_negative_int, default=16384, help="0 means unlimited")
    parser.add_argument("--stage3-max-units", type=non_negative_int, default=8192, help="0 means unlimited")
    parser.add_argument(
        "--queue-scan-limit",
        type=positive_int,
        default=256,
        help="How many queued requests to scan when forming a batch",
    )

    parser.add_argument("--output-json", default="simulation_results.json", help="Output report path")
    parser.add_argument("--verbose", action="store_true", help="Print search progress")
    return parser


def main() -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    trace_path = str(Path(args.trace_path).expanduser().resolve())

    cfg = SimulationConfig(
        limits=Limits(ttft_limit=args.P, token_limit=args.D),
        costs=CostParams(
            image_cost=args.A,
            context_cost=args.B,
            gen_cost=args.C,
            cost_power=args.cost_shape,
            compute_capability=args.K,
        ),
        memory=MemoryParams(
            image_footprint=args.X,
            context_footprint=args.Y,
            generated_footprint=args.Z,
            accelerator_memory=args.M,
        ),
        batching=BatchPolicy(
            stage1_max_requests=args.stage1_max_requests,
            stage2_max_requests=args.stage2_max_requests,
            stage3_max_requests=args.stage3_max_requests,
            stage1_max_units=args.stage1_max_units,
            stage2_max_units=args.stage2_max_units,
            stage3_max_units=args.stage3_max_units,
            queue_scan_limit=args.queue_scan_limit,
        ),
        assignment=AssignmentPolicy(strategy=args.assignment),
        scheduler=SchedulerPolicy(mode=args.scheduler),
    )

    print(f"Loading trace: {trace_path}")
    t0 = perf_counter()
    trace = load_trace(trace_path, max_requests=args.max_requests)
    print(f"Loaded {trace.size} requests in {perf_counter() - t0:.2f}s")

    trace_sha256 = compute_trace_sha256(trace_path)
    t1 = perf_counter()
    best_n, result, search_history = find_min_accelerators(
        trace=trace,
        cfg=cfg,
        min_n=args.min_n,
        max_n=args.max_n,
        verbose=args.verbose,
    )
    total_sim_time = perf_counter() - t1

    if result.ttft_values is None or result.t_values is None:
        raise RuntimeError("Statistics were not collected")

    ttft_stats = summarize(result.ttft_values)
    t_stats = summarize(result.t_values)
    telemetry = result.accelerator_telemetry or []
    telemetry_dicts = [telemetry_to_dict(item) for item in telemetry]
    avg_utilization = statistics.fmean([item.utilization for item in telemetry]) if telemetry else 0.0

    report = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python_version": sys.version,
        "platform": platform.platform(),
        "trace_path": trace_path,
        "trace_sha256": trace_sha256,
        "request_count": trace.size,
        "minimal_accelerators_N": best_n,
        "feasible": result.feasible,
        "limits": {"P_ttft_seconds": args.P, "D_token_seconds": args.D},
        "max_observed": {"ttft_seconds": result.max_ttft, "t_seconds": result.max_t},
        "ttft_stats_seconds": ttft_stats,
        "t_stats_seconds": t_stats,
        "accelerator_summary": {
            "count": len(telemetry_dicts),
            "avg_utilization": avg_utilization,
            "max_peak_memory": max((item["peak_memory"] for item in telemetry_dicts), default=0.0),
        },
        "accelerator_telemetry": telemetry_dicts,
        "search_history": [
            {
                "phase": item.phase,
                "accelerator_count": item.accelerator_count,
                "feasible": item.feasible,
                "max_ttft": item.max_ttft,
                "max_t": item.max_t,
                "reason": item.reason,
                "runtime_seconds": item.runtime_seconds,
            }
            for item in search_history
        ],
        "parameters": {
            "A": args.A,
            "B": args.B,
            "C": args.C,
            "X": args.X,
            "Y": args.Y,
            "Z": args.Z,
            "M": args.M,
            "K": args.K,
            "cost_power": args.cost_shape,
            "assignment": args.assignment,
            "scheduler": args.scheduler,
            "stage1_max_requests": args.stage1_max_requests,
            "stage2_max_requests": args.stage2_max_requests,
            "stage3_max_requests": args.stage3_max_requests,
            "stage1_max_units": args.stage1_max_units,
            "stage2_max_units": args.stage2_max_units,
            "stage3_max_units": args.stage3_max_units,
            "queue_scan_limit": args.queue_scan_limit,
        },
        "runtime_seconds": total_sim_time,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=== Simulation Complete ===")
    print(f"Minimal feasible N: {best_n}")
    print(
        "TTFT stats (s): "
        f"min={ttft_stats['min']:.4f} avg={ttft_stats['avg']:.4f} "
        f"median={ttft_stats['median']:.4f} p95={ttft_stats['p95']:.4f} "
        f"p99={ttft_stats['p99']:.4f} max={ttft_stats['max']:.4f}"
    )
    print(
        "T stats (s): "
        f"min={t_stats['min']:.4f} avg={t_stats['avg']:.4f} "
        f"median={t_stats['median']:.4f} p95={t_stats['p95']:.4f} "
        f"p99={t_stats['p99']:.4f} max={t_stats['max']:.4f}"
    )
    print(f"Average accelerator utilization: {avg_utilization:.4f}")
    print(f"Search+stats runtime: {total_sim_time:.2f}s")
    print(f"JSON report: {args.output_json}")
    return 0

