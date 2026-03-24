"""CLI argument validators and parsers."""

from __future__ import annotations

import argparse


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected > 0, got {value}")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected >= 0, got {value}")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected > 0, got {value}")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"Expected >= 0, got {value}")
    return parsed


def parse_cost_power(value: str) -> float:
    lowered = value.strip().lower()
    if lowered in {"sqrt", "square-root"}:
        return 0.5
    if lowered in {"cuberoot", "cube-root"}:
        return 1.0 / 3.0
    if lowered.startswith("power:"):
        power = float(lowered.split(":", 1)[1])
        if power <= 0.0:
            raise argparse.ArgumentTypeError("power must be > 0")
        return power
    try:
        power = float(lowered)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected sqrt, cuberoot, power:<float> or a float"
        ) from exc
    if power <= 0.0:
        raise argparse.ArgumentTypeError("cost power must be > 0")
    return power

