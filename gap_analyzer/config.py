"""Paths, run allowlist and defaults.

Only the dev run is allowed. The held-out run must never be loaded: any path that
contains "heldout" is refused, and any run name not in ALLOWED_RUNS is refused.
"""
import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]

# Directory that holds the DeVA run directories (read-only). Override only for local testing
# with a copy of the dev run; the allowlist and the heldout check apply either way.
RUNS_ROOT = Path(os.environ.get("GAP_ANALYZER_RUNS_ROOT", "/data/crugnola/sim/deva/output/runs"))
ALLOWED_RUNS = ("20261027_1743_gap_dev_h16",)
DEFAULT_RUN = ALLOWED_RUNS[0]
MODEL_DIR = "base"  # run_dir/<model>/<task>/epNNN_success_*_{frames,traj}.npz, *_meta.json

CACHE_DIR = Path(os.environ.get("GAP_ANALYZER_CACHE", APP_ROOT / "cache"))

# Each stored frame is a 256x256 mosaic of 128x128 tiles: TL=left, TR=right, BL=wrist, BR=empty.
# (verified against the per-view MSE logged in meta.json "step_errors")
VIEWS = ("left", "right", "wrist")
TILE = 128
VIEW_SLICES = {
    "left": (slice(0, 128), slice(0, 128)),
    "right": (slice(0, 128), slice(128, 256)),
    "wrist": (slice(128, 256), slice(0, 128)),
}

GROUP_SIZE = 8           # rollouts per (task, initial state)
GRASP_LIFT_M = 0.03      # grasp event = first step the object is this far above its initial height

# Viewer / preloading
LRU_GROUPS = 2
PRELOAD_WORKERS = 1      # labu data disk is an HDD: sequential reads are as fast and keep a clicked rollout quick
# Real time: RoboCasa steps the controller at control_freq=20 Hz (robocasa Kitchen default, not
# overridden by DeVA) and one frame is stored per env step, so 20 frames/s = real time.
# (DeVA's FPS=16 is only the video model's frame-rate conditioning, not the sim rate.)
REALTIME_FPS = 20
DEFAULT_FPS = REALTIME_FPS
PREFETCH_FRAMES = 12     # frames the browser fetches ahead of the current one
JPEG_QUALITY = 85
DISPLAY_SCALE = 2        # integer nearest-neighbour upscale of the 128 px tiles
MAX_COMPOSITE_WIDTH = 1800
METRIC_BATCH = 128       # frames per vectorized metric call
SCALE_SAMPLES_PER_EPISODE = 100_000  # error-map values sampled per episode for the 1st/99th percentile


class ForbiddenRunError(ValueError):
    pass


def check_run_dir(run_dir) -> Path:
    """Return the resolved run dir if it is allowed, else raise ForbiddenRunError."""
    raw = Path(run_dir)
    resolved = raw.resolve()
    for p in (raw, resolved):
        if "heldout" in str(p).lower():
            raise ForbiddenRunError(f"refusing held-out data: {p}")
    if resolved.name not in ALLOWED_RUNS:
        raise ForbiddenRunError(f"run {resolved.name!r} is not in the allowlist {ALLOWED_RUNS}")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def run_dir(name: str = DEFAULT_RUN) -> Path:
    return check_run_dir(RUNS_ROOT / name)
