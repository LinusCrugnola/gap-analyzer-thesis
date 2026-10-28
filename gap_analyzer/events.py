"""Episode events: chunk boundaries, success step, grasp step.

Step convention: frame i (0-based) in frames.npz is the observation after env step i+1
(frames.npz "step" = 1..T). traj.npz state arrays have T+1 entries, index k = after step k.
"""
import numpy as np

from . import config


def chunk_start_indices(chunk_ids: np.ndarray) -> np.ndarray:
    """Frame indices where a new action chunk (= new model query) starts, excluding frame 0."""
    return np.flatnonzero(np.diff(chunk_ids)) + 1


def chunk_boundary_steps(chunk_ids: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """x positions (in steps) of the boundaries between chunks, halfway between two frames."""
    return steps[chunk_start_indices(chunk_ids)] - 0.5


def success_step(meta: dict):
    return meta.get("success_step") if meta.get("success") else None


def grasp_step(traj) -> int | None:
    """First step at which the task object is GRASP_LIFT_M above its initial height, or None."""
    if "task_obj_pos_obj" not in traj:
        return None
    z = np.asarray(traj["task_obj_pos_obj"])[:, 2]
    lifted = np.flatnonzero(z[1:] - z[0] > config.GRASP_LIFT_M)
    return int(lifted[0]) + 1 if lifted.size else None
