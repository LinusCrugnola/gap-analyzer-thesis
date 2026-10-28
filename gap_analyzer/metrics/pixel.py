"""Pixel-space example metrics. Both work on single frames (H,W,3) and batches (N,H,W,3)."""
import numpy as np

from . import frame_metric


@frame_metric("pixel_mse", lower_is_better=True, batched=True)
def pixel_mse(real, imag):
    """Mean squared error over pixels and channels (0-255 scale), same as DeVA's logged mse."""
    d = real.astype(np.float32) - imag.astype(np.float32)
    m = (d * d).mean(-1)
    return m.mean((-2, -1)), m


@frame_metric("pixel_l1", lower_is_better=True, batched=True)
def pixel_l1(real, imag):
    """Mean absolute error over pixels and channels (0-255 scale)."""
    m = np.abs(real.astype(np.float32) - imag.astype(np.float32)).mean(-1)
    return m.mean((-2, -1)), m
