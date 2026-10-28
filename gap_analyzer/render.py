"""Compose one viewer image per frame from the in-memory buffers:
row 1 = real views, row 2 = imagined views, row 3 = error maps (fixed colormap scale)."""
import io

import numpy as np
from PIL import Image
from plotly.colors import hex_to_rgb, sequential

from . import config

COLORSCALE = sequential.Inferno
GAP = 4  # px between tiles (after upscaling)
_stops = np.array([hex_to_rgb(c) for c in COLORSCALE], np.float32)
LUT = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_stops)), _stops[:, ch])
                for ch in range(3)], 1).astype(np.uint8)


def colorize(maps: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """(...,H,W) float -> (...,H,W,3) uint8 via the LUT, linear between lo and hi."""
    x = (maps.astype(np.float32) - lo) / max(hi - lo, 1e-12)
    return LUT[np.clip(x * 255, 0, 255).astype(np.uint8)]


def compose(real: np.ndarray, imag: np.ndarray, maps: np.ndarray | None, lo: float, hi: float) -> np.ndarray:
    """real, imag: (V,H,W,3) uint8 of one frame; maps: (V,H,W) or None -> composite (H',W',3) uint8."""
    V, H, W, _ = real.shape
    err = colorize(maps, lo, hi) if maps is not None else np.full_like(real, 40)
    s = config.DISPLAY_SCALE
    th, tw = H * s, W * s
    out = np.full((3 * th + 2 * GAP, V * tw + (V - 1) * GAP, 3), 255, np.uint8)
    for r, row in enumerate((real, imag, err)):
        for v in range(V):
            tile = row[v].repeat(s, 0).repeat(s, 1) if s > 1 else row[v]
            y, x = r * (th + GAP), v * (tw + GAP)
            out[y:y + th, x:x + tw] = tile
    return out


def to_jpeg(img: np.ndarray, quality: int = config.JPEG_QUALITY) -> bytes:
    im = Image.fromarray(img)
    if im.width > config.MAX_COMPOSITE_WIDTH:
        im = im.resize((config.MAX_COMPOSITE_WIDTH, round(im.height * config.MAX_COMPOSITE_WIDTH / im.width)),
                       Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def css_gradient() -> str:
    return "linear-gradient(to top, " + ", ".join(COLORSCALE) + ")"
