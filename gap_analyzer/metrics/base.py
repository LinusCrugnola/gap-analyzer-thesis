"""The two metric interfaces and the helper that applies a metric to one episode.

Frame metric:  f(real HxWx3 uint8, imag HxWx3 uint8) -> (scalar, error_map HxW float | None)
    With batched=True the same function must also accept (N,H,W,3) batches and return
    ((N,) scalars, (N,H,W) maps | None). The helper then calls it on chunks of frames.
Clip metric:   f(real_clip TxHxWx3, imag_clip TxHxWx3) -> ((T,) scalars, (T,H,W) maps | None)
    Called once per action chunk (frames sharing a chunk id) and per view.
"""
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .. import config


@dataclass(frozen=True)
class Metric:
    name: str
    kind: Literal["frame", "clip"]
    fn: Callable
    lower_is_better: bool = True
    batched: bool = False
    description: str = ""


def apply_metric(metric: Metric, real: np.ndarray, imag: np.ndarray, chunk_ids: np.ndarray,
                 want_maps: bool = True):
    """Apply a metric to one episode.

    real, imag: (V,T,H,W,3) uint8; chunk_ids: (T,).
    Returns values (V,T) float64 and maps (V,T,H,W) float16 or None.
    """
    V, T, H, W, _ = real.shape
    values = np.empty((V, T), np.float64)
    maps = np.zeros((V, T, H, W), np.float16) if want_maps else None
    have_maps = False
    for v in range(V):
        if metric.kind == "frame" and metric.batched:
            spans = [(s, min(s + config.METRIC_BATCH, T)) for s in range(0, T, config.METRIC_BATCH)]
        elif metric.kind == "frame":
            spans = [(t, t + 1) for t in range(T)]
        else:
            bounds = np.flatnonzero(np.diff(chunk_ids)) + 1
            edges = [0, *bounds.tolist(), T]
            spans = list(zip(edges[:-1], edges[1:]))
        for s, e in spans:
            if metric.kind == "frame" and not metric.batched:
                val, m = metric.fn(real[v, s], imag[v, s])
                val, m = np.atleast_1d(val), (None if m is None else np.asarray(m)[None])
            else:
                val, m = metric.fn(real[v, s:e], imag[v, s:e])
            values[v, s:e] = val
            if m is not None and want_maps:
                maps[v, s:e] = m
                have_maps = True
    return values, (maps if have_maps else None)
