"""
utils/metrics.py
-----------------
Small shared helpers used across the AI pipeline for timing algorithms
and computing evaluation metrics, so each AI module doesn't repeat the
same boilerplate.
"""

import time
from contextlib import contextmanager


@contextmanager
def timer():
    """Context manager that measures wall-clock runtime in seconds.

    Usage:
        with timer() as t:
            do_work()
        print(t.elapsed)
    """
    class _Result:
        elapsed = None

    result = _Result()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed = time.perf_counter() - start


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two lat/lon points.
    Used as a fast straight-line fallback/estimate distinct from the
    network travel distance produced by the routing module.
    """
    import math
    r = 6371000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))
