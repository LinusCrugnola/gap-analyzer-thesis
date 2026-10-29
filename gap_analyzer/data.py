"""Episode index, episode loading and the in-memory group store used by the viewer.

There are no videos in the runs: each episode stores real (`sim`) and imagined (`pred`) frames
as (T,256,256,3) uint8 arrays in a compressed epNNN_success_*_frames.npz, one frame per env
step, with the 3 views tiled into a 2x2 mosaic (see config.VIEW_SLICES). Loading = decompress
both arrays once (sequential, no seeking) and cut out the views. In open-loop replay episodes
`pred` is the dream (the model only saw the real observation at query 0) and `sim` is the
simulator executing the dreamed actions.
"""
import glob
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config, events
from .metrics import apply_metric, get as get_metric

META_KEYS = ("task", "episode", "initial_state_id", "rollout_idx", "policy_seed", "success",
             "success_step", "episode_length", "instruction", "open_loop")


@lru_cache(maxsize=4)
def build_index(run_name: str = config.DEFAULT_RUN) -> pd.DataFrame:
    """One row per episode of one run, built from the *_meta.json files (read-only)."""
    rd = config.run_dir(run_name)
    rows = []
    for path in sorted(glob.glob(str(rd / config.MODEL_DIR / "*" / "ep*_meta.json"))):
        with open(path) as f:
            meta = json.load(f)
        row = {k: meta.get(k) for k in META_KEYS}
        row["open_loop"] = row["open_loop"] or "off"  # older closed-loop runs don't have the key
        row["stem"] = path[: -len("_meta.json")]
        row["run"] = run_name
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no episodes found under {rd}")
    df = pd.DataFrame(rows).rename(columns={"initial_state_id": "state", "rollout_idx": "rollout",
                                            "episode_length": "length"})
    df["success"] = df["success"].astype(bool)
    return df.sort_values(["task", "state", "rollout"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def build_full_index() -> pd.DataFrame:
    """Dev run plus the replay run (if any), merged into the same (task, state) groups.

    Columns added: replay (bool), twin (rollout id of the closed-loop twin, replay rows only).
    Replay rollout r gets the rollout id REPLAY_ROLLOUT_OFFSET + r so it sits next to closed-loop
    rollout r; the original rollout_idx is kept in rollout_idx. A replay episode must have a
    closed-loop twin in the dev run on (task, initial state, policy seed), otherwise it is refused
    (so states that are not in the dev run, e.g. held-out ones, can never come in this way).
    """
    dev = build_index(config.DEFAULT_RUN).assign(replay=False, rollout_idx=lambda d: d.rollout, twin=pd.NA)
    name = config.replay_run_name()
    if name is None:
        return dev
    rep = build_index(name)
    bad = rep[rep.open_loop != "replay"]
    if len(bad):
        raise ValueError(f"{name}: {len(bad)} episodes are not open_loop=replay, e.g. {bad.stem.iloc[0]}")
    twins = dev.set_index(["task", "state", "policy_seed"]).rollout
    keys = list(zip(rep.task, rep.state, rep.policy_seed))
    missing = [k for k in keys if k not in twins.index]
    if missing:
        raise config.ForbiddenRunError(f"{name}: {len(missing)} replay episodes have no closed-loop twin in "
                                       f"{config.DEFAULT_RUN}, e.g. {missing[0]}")
    rep = rep.assign(replay=True, rollout_idx=rep.rollout, twin=[int(twins.loc[k]) for k in keys],
                     rollout=rep.rollout + config.REPLAY_ROLLOUT_OFFSET)
    print(f"[index] {len(dev)} closed-loop episodes ({config.DEFAULT_RUN}) + {len(rep)} replay episodes ({name})",
          flush=True)
    return pd.concat([dev, rep], ignore_index=True).sort_values(["task", "state", "rollout"]).reset_index(drop=True)


def rollout_label(rollout: int, replay: bool) -> str:
    """Display name: r3 for closed loop, r0 dream for the open-loop replay of rollout 0."""
    return f"r{int(rollout) - config.REPLAY_ROLLOUT_OFFSET} dream" if replay else f"r{int(rollout)}"


@dataclass
class Episode:
    task: str
    state: int
    rollout: int
    real: np.ndarray        # (V,T,H,W,3) uint8
    imag: np.ndarray        # (V,T,H,W,3) uint8
    steps: np.ndarray       # (T,) env step of each frame, 1..T
    chunk_ids: np.ndarray   # (T,) action chunk / query index of each frame
    load_s: float = 0.0

    @property
    def n_frames(self) -> int:
        return self.real.shape[1]

    @property
    def nbytes(self) -> int:
        return self.real.nbytes + self.imag.nbytes


def _split_views(mosaic: np.ndarray) -> np.ndarray:
    return np.stack([mosaic[:, ys, xs] for ys, xs in (config.VIEW_SLICES[v] for v in config.VIEWS)])


def load_episode(row) -> Episode:
    t0 = time.perf_counter()
    with np.load(row["stem"] + "_frames.npz") as z:
        real = _split_views(z["sim"])
        imag = _split_views(z["pred"])
        steps, chunk_ids = z["step"], z["chunk"]
    if real.shape != imag.shape or real.shape[1] != len(steps):
        raise ValueError(f"{row['stem']}: real {real.shape} vs imagined {imag.shape} vs {len(steps)} steps")
    return Episode(row["task"], int(row["state"]), int(row["rollout"]), real, imag, steps, chunk_ids,
                   time.perf_counter() - t0)


def load_traj(row):
    return np.load(row["stem"] + "_traj.npz")


# ---------------------------------------------------------------- group store

@dataclass
class _Slot:
    row: dict
    lock: threading.Lock = field(default_factory=threading.Lock)
    episode: Episode | None = None
    maps: dict = field(default_factory=dict)  # metric name -> (V,T,H,W) float16 or None; current metric only
    maps_s: float = 0.0


@dataclass
class Group:
    task: str
    state: int
    slots: dict                     # rollout -> _Slot
    t_start: float = 0.0
    preload_s: float | None = None  # wall time until all rollouts are decoded and have error maps
    cancelled: bool = False

    @property
    def n_loaded(self) -> int:
        return sum(s.episode is not None for s in self.slots.values())


class GroupStore:
    """Keeps the decoded frames and error maps of the last LRU_GROUPS opened groups in RAM."""

    def __init__(self, index: pd.DataFrame, max_groups: int = config.LRU_GROUPS):
        self.index = index
        self.max_groups = max_groups
        self.metric = None
        self._groups: OrderedDict[tuple, Group] = OrderedDict()
        self._lock = threading.Lock()
        self._urgent = 0  # user requests for rollouts that are not in memory yet
        self._urgent_cv = threading.Condition()

    def open_group(self, task: str, state: int, metric: str) -> Group:
        """Return the group, starting a background preload if it is not in memory yet."""
        key = (task, int(state))
        self.set_metric(metric)
        with self._lock:
            if key in self._groups:
                self._groups.move_to_end(key)
                return self._groups[key]
            rows = self.index[(self.index.task == task) & (self.index.state == int(state))]
            group = Group(task, int(state), {int(r.rollout): _Slot(r._asdict()) for r in rows.itertuples(index=False)},
                          t_start=time.perf_counter())
            self._groups[key] = group
            while len(self._groups) > self.max_groups:
                _, old = self._groups.popitem(last=False)
                old.cancelled = True
        threading.Thread(target=self._preload, args=(group,), daemon=True).start()
        return group

    def _preload(self, group: Group):
        def work(rollout):
            with self._urgent_cv:  # don't start a new rollout while a user request is loading
                self._urgent_cv.wait_for(lambda: self._urgent == 0)
            if not group.cancelled:
                self._ensure(group, rollout, self.metric)
        with ThreadPoolExecutor(config.PRELOAD_WORKERS) as ex:
            list(ex.map(work, sorted(group.slots)))
        if not group.cancelled and group.preload_s is None:
            group.preload_s = time.perf_counter() - group.t_start
            print(f"[preload] {group.task} state {group.state}: {group.n_loaded} rollouts in "
                  f"{group.preload_s:.1f} s", flush=True)

    def _ensure(self, group: Group, rollout: int, metric: str):
        slot = group.slots[rollout]
        with slot.lock:  # a user request for a rollout the preloader has not reached yet loads it right away
            if slot.episode is None:
                slot.episode = load_episode(slot.row)
            if metric not in slot.maps:
                t0 = time.perf_counter()
                ep = slot.episode
                _, maps = apply_metric(get_metric(metric), ep.real, ep.imag, ep.chunk_ids)
                slot.maps = {metric: maps}  # drop maps of the previous metric
                slot.maps_s = time.perf_counter() - t0
            return slot.episode, slot.maps[metric]

    def get(self, task: str, state: int, rollout: int, metric: str):
        """(Episode, error maps) for one rollout, loading it first if needed."""
        group = self.open_group(task, state, metric)
        if rollout not in group.slots:
            raise KeyError((task, state, rollout))
        slot = group.slots[rollout]
        if slot.episode is not None and metric in slot.maps:
            return slot.episode, slot.maps[metric]
        with self._urgent_cv:
            self._urgent += 1
        try:
            return self._ensure(group, rollout, metric)
        finally:
            with self._urgent_cv:
                self._urgent -= 1
                self._urgent_cv.notify_all()

    def set_metric(self, metric: str):
        """Switch the current metric; recompute error maps of all preloaded groups in the background."""
        if metric == self.metric:
            return
        self.metric = metric
        with self._lock:
            groups = list(self._groups.values())
        for g in groups:
            threading.Thread(target=self._recompute, args=(g, metric), daemon=True).start()

    def _recompute(self, group: Group, metric: str):
        for r in sorted(group.slots):
            if group.cancelled or self.metric != metric:
                return
            if group.slots[r].episode is not None:
                self._ensure(group, r, metric)

    def status(self) -> list[dict]:
        with self._lock:
            groups = list(self._groups.values())
        out = []
        for g in reversed(groups):
            nbytes = sum(s.episode.nbytes for s in g.slots.values() if s.episode is not None)
            nbytes += sum(m.nbytes for s in g.slots.values() for m in s.maps.values() if m is not None)
            out.append({"task": g.task, "state": g.state, "loaded": g.n_loaded, "total": len(g.slots),
                        "preload_s": g.preload_s, "gb": nbytes / 1e9})
        return out


def rss_gb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except OSError:
        return float("nan")
