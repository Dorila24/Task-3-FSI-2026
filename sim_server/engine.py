"""Core simulation and search logic."""

from __future__ import annotations

import math
import statistics
from array import array
from collections import deque
from heapq import heappop, heappush
from time import perf_counter
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

from .constants import EPS, INF, MEMORY_ROUNDING_TOL
from .models import (
    AcceleratorTelemetry,
    SearchRecord,
    SimulationConfig,
    SimulationResult,
    TraceData,
)


def stage_time(units: int, unit_cost: float, costs) -> float:
    """Compute stage duration for a batch with nonlinear speedup."""
    if units <= 0:
        return 0.0
    return unit_cost * (units ** costs.cost_power) / costs.compute_capability


def estimate_request_work(req_idx: int, trace: TraceData, cfg: SimulationConfig) -> float:
    img = trace.num_images[req_idx]
    ctx = trace.context_tokens[req_idx]
    gen = trace.generated_tokens[req_idx]
    return (
        stage_time(img, cfg.costs.image_cost, cfg.costs)
        + stage_time(ctx, cfg.costs.context_cost, cfg.costs)
        + stage_time(gen, cfg.costs.gen_cost, cfg.costs)
    )


def assign_requests_balanced(
    trace: TraceData,
    accelerator_count: int,
    cfg: SimulationConfig,
) -> List[List[int]]:
    """Assign requests to accelerators by minimum projected finish time."""
    assignments: List[List[int]] = [[] for _ in range(accelerator_count)]
    heap: List[Tuple[float, int]] = [(0.0, acc_id) for acc_id in range(accelerator_count)]

    for req_idx in range(trace.size):
        projected_finish, acc_id = heappop(heap)
        arrival = trace.arrivals[req_idx]
        start = arrival if arrival > projected_finish else projected_finish
        finish = start + estimate_request_work(req_idx, trace, cfg)
        assignments[acc_id].append(req_idx)
        heappush(heap, (finish, acc_id))

    return assignments


def assign_requests_round_robin(trace: TraceData, accelerator_count: int) -> List[List[int]]:
    """Assign requests in fixed cyclic order."""
    assignments: List[List[int]] = [[] for _ in range(accelerator_count)]
    for req_idx in range(trace.size):
        assignments[req_idx % accelerator_count].append(req_idx)
    return assignments


def pick_assignments(
    trace: TraceData,
    accelerator_count: int,
    cfg: SimulationConfig,
) -> List[List[int]]:
    if cfg.assignment.strategy == "round-robin":
        return assign_requests_round_robin(trace, accelerator_count)
    return assign_requests_balanced(trace, accelerator_count, cfg)


def request_deadline(req_idx: int, trace: TraceData, deadline_delta: float) -> float:
    return trace.arrivals[req_idx] + deadline_delta


def build_stage_batch(
    stage: int,
    queue: Deque[int],
    trace: TraceData,
    cfg: SimulationConfig,
    memory_in_use: float,
) -> Tuple[List[int], int]:
    """Select a memory-safe and policy-safe stage batch.

    Returns:
        A tuple of (selected_request_indices, total_units_in_batch).
    """
    if not queue:
        return [], 0

    b = cfg.batching
    if stage == 1:
        max_requests = b.stage1_max_requests
        max_units = b.stage1_max_units
    elif stage == 2:
        max_requests = b.stage2_max_requests
        max_units = b.stage2_max_units
    else:
        max_requests = b.stage3_max_requests
        max_units = b.stage3_max_units

    selected: List[int] = []
    skipped: List[int] = []
    units_sum = 0
    added_peak_memory = 0.0
    scanned = 0
    scan_cap = b.queue_scan_limit if b.queue_scan_limit > 0 else len(queue)

    while queue and len(selected) < max_requests and scanned < scan_cap:
        req_idx = queue.popleft()
        scanned += 1

        if stage == 1:
            req_units = int(trace.num_images[req_idx])
            req_mem = trace.num_images[req_idx] * cfg.memory.image_footprint
        elif stage == 2:
            req_units = int(trace.context_tokens[req_idx])
            req_mem = trace.context_tokens[req_idx] * cfg.memory.context_footprint
        else:
            req_units = int(trace.generated_tokens[req_idx])
            req_mem = trace.generated_tokens[req_idx] * cfg.memory.generated_footprint

        if max_units > 0 and selected and units_sum + req_units > max_units:
            skipped.append(req_idx)
            break

        peak_if_added = memory_in_use + added_peak_memory + req_mem
        if peak_if_added > cfg.memory.accelerator_memory + EPS:
            skipped.append(req_idx)
            continue

        selected.append(req_idx)
        units_sum += req_units
        added_peak_memory += req_mem

        if max_units > 0 and units_sum >= max_units:
            break

    if skipped:
        queue.extendleft(reversed(skipped))

    if not selected:
        return [], 0
    return selected, units_sum


def choose_next_stage(
    now: float,
    q1: Deque[int],
    q2: Deque[int],
    q3: Deque[int],
    trace: TraceData,
    cfg: SimulationConfig,
    memory_in_use: float,
) -> Tuple[int, List[int], int]:
    """Choose next stage to execute based on scheduling policy and deadlines."""
    candidates: List[Tuple[float, int]] = []

    def stage_front_units(stage: int, req_idx: int) -> int:
        if stage == 1:
            return int(trace.num_images[req_idx])
        if stage == 2:
            return int(trace.context_tokens[req_idx])
        return int(trace.generated_tokens[req_idx])

    if q1:
        req_idx = q1[0]
        deadline = request_deadline(req_idx, trace, cfg.limits.ttft_limit)
        if cfg.scheduler.mode == "slack":
            est = stage_time(stage_front_units(1, req_idx), cfg.costs.image_cost, cfg.costs)
            priority = deadline - now - est
        else:
            priority = deadline
        candidates.append((priority, 1))
    if q2:
        req_idx = q2[0]
        deadline = request_deadline(req_idx, trace, cfg.limits.ttft_limit)
        if cfg.scheduler.mode == "slack":
            est = stage_time(stage_front_units(2, req_idx), cfg.costs.context_cost, cfg.costs)
            priority = deadline - now - est
        else:
            priority = deadline
        candidates.append((priority, 2))
    if q3:
        req_idx = q3[0]
        deadline = request_deadline(req_idx, trace, cfg.limits.token_limit)
        if cfg.scheduler.mode == "slack":
            est = stage_time(stage_front_units(3, req_idx), cfg.costs.gen_cost, cfg.costs)
            priority = deadline - now - est
        else:
            priority = deadline
        candidates.append((priority, 3))

    if not candidates:
        return 0, [], 0

    # Tie-breaker prefers later stages to reduce in-flight memory.
    candidates.sort(key=lambda x: (x[0], -x[1]))
    for _, stage in candidates:
        queue = q1 if stage == 1 else q2 if stage == 2 else q3
        batch, units = build_stage_batch(stage, queue, trace, cfg, memory_in_use)
        if batch:
            return stage, batch, units
    return 0, [], 0


def simulate_single_accelerator(
    accelerator_id: int,
    assigned: Sequence[int],
    trace: TraceData,
    cfg: SimulationConfig,
    collect_stats: bool,
    ttft_out: Optional[array],
    t_out: Optional[array],
) -> Tuple[bool, str, float, float, AcceleratorTelemetry]:
    """Run event-driven simulation for a single accelerator queue."""
    telemetry = AcceleratorTelemetry(
        accelerator_id=accelerator_id,
        assigned_requests=len(assigned),
    )
    if not assigned:
        return True, "ok", 0.0, 0.0, telemetry

    q1: Deque[int] = deque()
    q2: Deque[int] = deque()
    q3: Deque[int] = deque()

    arrival_pos = 0
    total = len(assigned)
    done = 0
    now = trace.arrivals[assigned[0]]
    busy_stage = 0
    busy_until = INF
    running_batch: List[int] = []

    max_ttft_seen = 0.0
    max_t_seen = 0.0
    memory_in_use = 0.0

    def update_peaks() -> None:
        if memory_in_use > telemetry.peak_memory:
            telemetry.peak_memory = memory_in_use
        if len(q1) > telemetry.peak_q1_len:
            telemetry.peak_q1_len = len(q1)
        if len(q2) > telemetry.peak_q2_len:
            telemetry.peak_q2_len = len(q2)
        if len(q3) > telemetry.peak_q3_len:
            telemetry.peak_q3_len = len(q3)

    def enqueue_arrivals(up_to_time: float) -> None:
        nonlocal arrival_pos
        while arrival_pos < total:
            req_idx = assigned[arrival_pos]
            if trace.arrivals[req_idx] > up_to_time + EPS:
                break
            q1.append(req_idx)
            arrival_pos += 1
        update_peaks()

    def finalize_stage(stage: int, batch: Iterable[int], finish_time: float) -> Tuple[bool, str]:
        nonlocal done, memory_in_use, max_ttft_seen, max_t_seen

        if stage == 1:
            for req_idx in batch:
                image_mem = trace.num_images[req_idx] * cfg.memory.image_footprint
                memory_in_use += image_mem
                if memory_in_use > cfg.memory.accelerator_memory + EPS:
                    return False, f"memory overflow after stage1 on request {req_idx}"
                q2.append(req_idx)
                update_peaks()
            return True, "ok"

        if stage == 2:
            for req_idx in batch:
                image_mem = trace.num_images[req_idx] * cfg.memory.image_footprint
                context_mem = trace.context_tokens[req_idx] * cfg.memory.context_footprint
                memory_in_use += context_mem - image_mem
                if memory_in_use > cfg.memory.accelerator_memory + EPS:
                    return False, f"memory overflow after stage2 on request {req_idx}"
                if memory_in_use < 0.0:
                    if memory_in_use > -MEMORY_ROUNDING_TOL:
                        memory_in_use = 0.0
                    else:
                        return False, f"negative memory accounting on request {req_idx}"
                ttft = finish_time - trace.arrivals[req_idx]
                if ttft > max_ttft_seen:
                    max_ttft_seen = ttft
                if collect_stats and ttft_out is not None:
                    ttft_out[req_idx] = ttft
                if ttft > cfg.limits.ttft_limit + EPS:
                    return False, f"TTFT violation on request {req_idx}"
                q3.append(req_idx)
                update_peaks()
            return True, "ok"

        # stage == 3
        for req_idx in batch:
            context_mem = trace.context_tokens[req_idx] * cfg.memory.context_footprint
            memory_in_use -= context_mem
            if memory_in_use < 0.0:
                if memory_in_use > -MEMORY_ROUNDING_TOL:
                    memory_in_use = 0.0
                else:
                    return False, f"negative memory accounting after stage3 on request {req_idx}"
            total_t = finish_time - trace.arrivals[req_idx]
            if total_t > max_t_seen:
                max_t_seen = total_t
            if collect_stats and t_out is not None:
                t_out[req_idx] = total_t
            if total_t > cfg.limits.token_limit + EPS:
                return False, f"T violation on request {req_idx}"
            done += 1
            telemetry.completed_requests += 1
            update_peaks()
        return True, "ok"

    while done < total:
        if busy_stage == 0:
            enqueue_arrivals(now)
            stage, batch, units = choose_next_stage(now, q1, q2, q3, trace, cfg, memory_in_use)

            if stage == 0:
                if arrival_pos < total:
                    next_t = max(now, trace.arrivals[assigned[arrival_pos]])
                    telemetry.idle_time += max(0.0, next_t - now)
                    now = next_t
                    enqueue_arrivals(now)
                    continue
                if q1 or q2 or q3:
                    return (
                        False,
                        "deadlock: queues not empty but no runnable batch",
                        max_ttft_seen,
                        max_t_seen,
                        telemetry,
                    )
                break

            running_batch = batch
            busy_stage = stage

            if stage == 1:
                duration = stage_time(units, cfg.costs.image_cost, cfg.costs)
                telemetry.stage1_batches += 1
                telemetry.stage1_units += units
            elif stage == 2:
                duration = stage_time(units, cfg.costs.context_cost, cfg.costs)
                telemetry.stage2_batches += 1
                telemetry.stage2_units += units
            else:
                duration = stage_time(units, cfg.costs.gen_cost, cfg.costs)
                telemetry.stage3_batches += 1
                telemetry.stage3_units += units

            if duration <= EPS:
                ok, reason = finalize_stage(stage, running_batch, now)
                if not ok:
                    return False, reason, max_ttft_seen, max_t_seen, telemetry
                busy_stage = 0
                running_batch = []
                continue

            if busy_stage == 1:
                telemetry.stage1_time += duration
            elif busy_stage == 2:
                telemetry.stage2_time += duration
            else:
                telemetry.stage3_time += duration
            busy_until = now + duration

        next_arrival_time = trace.arrivals[assigned[arrival_pos]] if arrival_pos < total else INF
        if next_arrival_time < busy_until - EPS:
            now = next_arrival_time
            enqueue_arrivals(now)
            continue

        now = busy_until
        ok, reason = finalize_stage(busy_stage, running_batch, now)
        if not ok:
            return False, reason, max_ttft_seen, max_t_seen, telemetry
        busy_stage = 0
        running_batch = []

    if memory_in_use > EPS:
        return False, "memory leak: non-zero memory after completion", max_ttft_seen, max_t_seen, telemetry
    return True, "ok", max_ttft_seen, max_t_seen, telemetry


def simulate(
    trace: TraceData,
    accelerator_count: int,
    cfg: SimulationConfig,
    collect_stats: bool,
) -> SimulationResult:
    """Run full simulation for all accelerators."""
    if accelerator_count <= 0:
        raise ValueError("accelerator_count must be > 0")

    # Quick infeasibility checks on single-request memory demands.
    for req_idx in range(trace.size):
        image_mem = trace.num_images[req_idx] * cfg.memory.image_footprint
        context_mem = trace.context_tokens[req_idx] * cfg.memory.context_footprint
        generated_mem = trace.generated_tokens[req_idx] * cfg.memory.generated_footprint
        if image_mem > cfg.memory.accelerator_memory + EPS:
            return SimulationResult(False, f"request {req_idx}: image preprocessing exceeds memory", 0.0, 0.0)
        if image_mem + context_mem > cfg.memory.accelerator_memory + EPS:
            return SimulationResult(False, f"request {req_idx}: stage2 peak exceeds memory", 0.0, 0.0)
        if context_mem + generated_mem > cfg.memory.accelerator_memory + EPS:
            return SimulationResult(False, f"request {req_idx}: stage3 peak exceeds memory", 0.0, 0.0)

    assignments = pick_assignments(trace, accelerator_count, cfg)
    ttft_values = array("d", [0.0]) * trace.size if collect_stats else None
    t_values = array("d", [0.0]) * trace.size if collect_stats else None
    telemetries: List[AcceleratorTelemetry] = []

    global_max_ttft = 0.0
    global_max_t = 0.0
    for acc_id, acc_assigned in enumerate(assignments):
        ok, reason, max_ttft, max_t, telemetry = simulate_single_accelerator(
            accelerator_id=acc_id,
            assigned=acc_assigned,
            trace=trace,
            cfg=cfg,
            collect_stats=collect_stats,
            ttft_out=ttft_values,
            t_out=t_values,
        )
        telemetries.append(telemetry)
        if max_ttft > global_max_ttft:
            global_max_ttft = max_ttft
        if max_t > global_max_t:
            global_max_t = max_t
        if not ok:
            return SimulationResult(
                False,
                reason,
                global_max_ttft,
                global_max_t,
                ttft_values,
                t_values,
                telemetries,
            )

    return SimulationResult(
        True,
        "ok",
        global_max_ttft,
        global_max_t,
        ttft_values,
        t_values,
        telemetries,
    )


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Return aggregate and percentile statistics for a numeric sample."""
    sorted_values = sorted(values)

    def percentile(q: float) -> float:
        if not sorted_values:
            return 0.0
        if q <= 0.0:
            return sorted_values[0]
        if q >= 1.0:
            return sorted_values[-1]
        idx = q * (len(sorted_values) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return sorted_values[lo]
        w = idx - lo
        return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w

    return {
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "avg": statistics.fmean(sorted_values),
        "median": statistics.median(sorted_values),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def telemetry_to_dict(telemetry: AcceleratorTelemetry) -> dict[str, float | int]:
    """Convert telemetry dataclass to JSON-friendly dictionary."""
    return {
        "accelerator_id": telemetry.accelerator_id,
        "assigned_requests": telemetry.assigned_requests,
        "completed_requests": telemetry.completed_requests,
        "stage1_batches": telemetry.stage1_batches,
        "stage2_batches": telemetry.stage2_batches,
        "stage3_batches": telemetry.stage3_batches,
        "stage1_units": telemetry.stage1_units,
        "stage2_units": telemetry.stage2_units,
        "stage3_units": telemetry.stage3_units,
        "stage1_time": telemetry.stage1_time,
        "stage2_time": telemetry.stage2_time,
        "stage3_time": telemetry.stage3_time,
        "busy_time": telemetry.busy_time,
        "idle_time": telemetry.idle_time,
        "utilization": telemetry.utilization,
        "peak_memory": telemetry.peak_memory,
        "peak_q1_len": telemetry.peak_q1_len,
        "peak_q2_len": telemetry.peak_q2_len,
        "peak_q3_len": telemetry.peak_q3_len,
    }


def find_min_accelerators(
    trace: TraceData,
    cfg: SimulationConfig,
    min_n: int,
    max_n: int,
    verbose: bool,
) -> Tuple[int, SimulationResult, List[SearchRecord]]:
    """Find minimal feasible accelerator count using expand + binary search."""
    hi = min_n
    feasible_result: Optional[SimulationResult] = None
    search_history: List[SearchRecord] = []

    while hi <= max_n:
        t0 = perf_counter()
        result = simulate(trace, hi, cfg, collect_stats=False)
        dt = perf_counter() - t0
        record = SearchRecord(
            phase="expand",
            accelerator_count=hi,
            feasible=result.feasible,
            max_ttft=result.max_ttft,
            max_t=result.max_t,
            reason=result.reason,
            runtime_seconds=dt,
        )
        search_history.append(record)
        if verbose:
            print(
                f"[search] N={hi:<4} feasible={result.feasible:<5} "
                f"max_ttft={result.max_ttft:.4f}s max_t={result.max_t:.4f}s "
                f"time={dt:.2f}s reason={result.reason}"
            )
        if result.feasible:
            feasible_result = result
            break
        hi *= 2

    if feasible_result is None:
        raise RuntimeError(f"No feasible N found up to max_n={max_n}")

    left = min_n
    right = hi
    best_n = hi
    while left <= right:
        mid = (left + right) // 2
        t0 = perf_counter()
        result = simulate(trace, mid, cfg, collect_stats=False)
        dt = perf_counter() - t0
        search_history.append(
            SearchRecord(
                phase="binary",
                accelerator_count=mid,
                feasible=result.feasible,
                max_ttft=result.max_ttft,
                max_t=result.max_t,
                reason=result.reason,
                runtime_seconds=dt,
            )
        )
        if verbose:
            print(
                f"[binary] N={mid:<4} feasible={result.feasible:<5} "
                f"max_ttft={result.max_ttft:.4f}s max_t={result.max_t:.4f}s "
                f"time={dt:.2f}s reason={result.reason}"
            )
        if result.feasible:
            best_n = mid
            feasible_result = result
            right = mid - 1
        else:
            left = mid + 1

    final_result = simulate(trace, best_n, cfg, collect_stats=True)
    if not final_result.feasible:
        raise RuntimeError(
            "Unexpected: final feasible run failed with stats collection enabled"
        )
    return best_n, final_result, search_history

