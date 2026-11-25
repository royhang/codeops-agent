from __future__ import annotations

import math
from typing import List


def percentile(values: List[float], p: float) -> float:
    """Compute linear-interpolated percentile (e.g. p=95 for P95 latency)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    rank = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(ordered[int(rank)], 2)
    frac = rank - low
    val = ordered[low] * (1.0 - frac) + ordered[high] * frac
    return round(val, 2)
