"""Trace input/output utility functions."""

from __future__ import annotations

import csv
import gzip
import hashlib
from array import array
from datetime import date
from typing import Optional

from .constants import SHA256_CHUNK_SIZE_BYTES
from .models import TraceData


def parse_timestamp_seconds(ts: str, day_cache: dict[str, int]) -> float:
    """Fast parser for ISO-like `YYYY-MM-DDTHH:MM:SS(.fff)Z`."""
    day_key = ts[:10]
    day_base = day_cache.get(day_key)
    if day_base is None:
        y = int(ts[0:4])
        m = int(ts[5:7])
        d = int(ts[8:10])
        day_base = date(y, m, d).toordinal() * 86_400
        day_cache[day_key] = day_base

    hour = int(ts[11:13])
    minute = int(ts[14:16])
    sec = int(ts[17:19])
    frac = 0.0

    if len(ts) > 19 and ts[19] == ".":
        end = ts.find("Z", 20)
        if end < 0:
            end = len(ts)
        frac_digits = ts[20:end]
        if frac_digits:
            frac = int(frac_digits) / (10 ** len(frac_digits))

    return day_base + hour * 3600 + minute * 60 + sec + frac


def load_trace(path: str, max_requests: Optional[int] = None) -> TraceData:
    """Load trace rows and normalize timestamps to seconds since first arrival."""
    arrivals = array("d")
    num_images = array("I")
    context_tokens = array("I")
    generated_tokens = array("I")

    first_ts: Optional[float] = None
    day_cache: dict[str, int] = {}

    def parse_reader(reader: csv.DictReader) -> None:
        nonlocal first_ts
        for idx, row in enumerate(reader):
            if max_requests is not None and idx >= max_requests:
                break

            ts = parse_timestamp_seconds(row["TIMESTAMP"], day_cache)
            if first_ts is None:
                first_ts = ts

            arrivals.append(ts - first_ts)
            num_images.append(int(row["NumImages"]))
            context_tokens.append(int(row["ContextTokens"]))
            generated_tokens.append(int(row["GeneratedTokens"]))

    if path.endswith(".gz"):
        with gzip.open(path, "rt", newline="") as f:
            parse_reader(csv.DictReader(f))
    else:
        with open(path, "rt", newline="") as f:
            parse_reader(csv.DictReader(f))

    if first_ts is None:
        raise ValueError("Trace is empty")

    return TraceData(
        arrivals=arrivals,
        num_images=num_images,
        context_tokens=context_tokens,
        generated_tokens=generated_tokens,
    )


def compute_trace_sha256(path: str) -> str:
    """Compute SHA256 checksum for trace reproducibility."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(SHA256_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

