"""Typed data structures for simulator configuration and outputs."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TraceData:
    arrivals: array  # 'd'
    num_images: array  # 'I'
    context_tokens: array  # 'I'
    generated_tokens: array  # 'I'

    @property
    def size(self) -> int:
        return len(self.arrivals)


@dataclass
class Limits:
    ttft_limit: float
    token_limit: float


@dataclass
class CostParams:
    image_cost: float
    context_cost: float
    gen_cost: float
    cost_power: float
    compute_capability: float


@dataclass
class MemoryParams:
    image_footprint: float
    context_footprint: float
    generated_footprint: float
    accelerator_memory: float


@dataclass
class BatchPolicy:
    stage1_max_requests: int
    stage2_max_requests: int
    stage3_max_requests: int
    stage1_max_units: int
    stage2_max_units: int
    stage3_max_units: int
    queue_scan_limit: int


@dataclass
class AssignmentPolicy:
    strategy: str


@dataclass
class SchedulerPolicy:
    mode: str


@dataclass
class SimulationConfig:
    limits: Limits
    costs: CostParams
    memory: MemoryParams
    batching: BatchPolicy
    assignment: AssignmentPolicy
    scheduler: SchedulerPolicy


@dataclass
class AcceleratorTelemetry:
    accelerator_id: int
    assigned_requests: int
    completed_requests: int = 0
    stage1_batches: int = 0
    stage2_batches: int = 0
    stage3_batches: int = 0
    stage1_units: int = 0
    stage2_units: int = 0
    stage3_units: int = 0
    stage1_time: float = 0.0
    stage2_time: float = 0.0
    stage3_time: float = 0.0
    idle_time: float = 0.0
    peak_memory: float = 0.0
    peak_q1_len: int = 0
    peak_q2_len: int = 0
    peak_q3_len: int = 0

    @property
    def busy_time(self) -> float:
        return self.stage1_time + self.stage2_time + self.stage3_time

    @property
    def utilization(self) -> float:
        denom = self.busy_time + self.idle_time
        return self.busy_time / denom if denom > 0.0 else 0.0


@dataclass
class SimulationResult:
    feasible: bool
    reason: str
    max_ttft: float
    max_t: float
    ttft_values: Optional[array] = None
    t_values: Optional[array] = None
    accelerator_telemetry: Optional[List[AcceleratorTelemetry]] = None


@dataclass
class SearchRecord:
    phase: str
    accelerator_count: int
    feasible: bool
    max_ttft: float
    max_t: float
    reason: str
    runtime_seconds: float

