"""Metric registry. Every module in this package is imported automatically, so adding a
metric = adding a decorated function in a new or existing module here.

    @frame_metric("pixel_mse", lower_is_better=True, batched=True)
    def pixel_mse(real, imag): ...

    @clip_metric("my_temporal_metric", lower_is_better=True)
    def my_temporal_metric(real_clip, imag_clip): ...
"""
import importlib
import pkgutil

from .base import Metric, apply_metric

REGISTRY: dict[str, Metric] = {}


def _register(metric: Metric):
    if metric.name in REGISTRY:
        raise ValueError(f"duplicate metric name {metric.name!r}")
    REGISTRY[metric.name] = metric


def frame_metric(name: str, lower_is_better: bool = True, batched: bool = False, description: str = ""):
    def deco(fn):
        _register(Metric(name, "frame", fn, lower_is_better, batched, description or (fn.__doc__ or "").strip()))
        return fn
    return deco


def clip_metric(name: str, lower_is_better: bool = True, description: str = ""):
    def deco(fn):
        _register(Metric(name, "clip", fn, lower_is_better, False, description or (fn.__doc__ or "").strip()))
        return fn
    return deco


def get(name: str) -> Metric:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}, known: {sorted(REGISTRY)}") from None


for _m in pkgutil.iter_modules(__path__):
    if _m.name != "base":
        importlib.import_module(f"{__name__}.{_m.name}")

__all__ = ["REGISTRY", "Metric", "apply_metric", "frame_metric", "clip_metric", "get"]
