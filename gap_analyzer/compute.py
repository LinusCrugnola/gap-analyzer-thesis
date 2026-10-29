"""Compute metrics over all dev-run (and replay-run, if present) episodes into the parquet cache.

    python -m gap_analyzer.compute --metric pixel_mse [--metric pixel_l1 ...] [--force] [--workers 8]
    python -m gap_analyzer.compute --all

Outputs per metric (in config.CACHE_DIR):
    <metric>.parquet           one row per (episode, frame)
    <metric>_episodes.parquet  one row per episode: mean, final, success, events
    <metric>_scale.json        1st/99th percentile of error-map values (fixed colormap scale) + metric info
A cache whose episodes differ from the index (e.g. a replay run appeared) is recomputed.
Resumable: finished episodes are kept in cache/.parts/<metric>/ until all episodes are done.
Error maps are never written; only a random sample of their values is kept for the scale.
"""
import argparse
import json
import os
import shutil
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config, data, events, metrics


def cache_paths(metric: str) -> dict:
    c = config.CACHE_DIR
    return {"frames": c / f"{metric}.parquet", "episodes": c / f"{metric}_episodes.parquet",
            "scale": c / f"{metric}_scale.json", "parts": c / ".parts" / metric}


def _part_name(row) -> str:
    return f"{'replay_' if row['replay'] else ''}{row['task']}_ep{int(row['episode']):03d}"


def _write_atomic(path, write):
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _process_episode(row: dict, metric_names: list[str]):
    """Load one episode once, compute all requested metrics, write one part per metric."""
    ep = data.load_episode(row)
    with data.load_traj(row) as traj:
        grasp = events.grasp_step(traj)
        # replay episodes run to the max length after success; did the success hold at the end?
        success_final = bool(np.asarray(traj["task_success"])[-1]) if "task_success" in traj else None
    T = ep.n_frames
    for name in metric_names:
        m = metrics.get(name)
        values, maps = metrics.apply_metric(m, ep.real, ep.imag, ep.chunk_ids)
        df = pd.DataFrame({
            "task": row["task"], "state": int(row["state"]), "rollout": int(row["rollout"]),
            "episode": int(row["episode"]), "frame": np.arange(T, dtype=np.int32),
            "step": ep.steps.astype(np.int32), "t_norm": ep.steps / T, "chunk": ep.chunk_ids.astype(np.int32),
            **{f"value_view{v}": values[v] for v in range(values.shape[0])},
            "value_mean": values.mean(0), "success": bool(row["success"]), "replay": bool(row["replay"]),
        })
        ep_row = {"task": row["task"], "state": int(row["state"]), "rollout": int(row["rollout"]),
                  "episode": int(row["episode"]), "replay": bool(row["replay"]),
                  "rollout_idx": int(row["rollout_idx"]), "success": bool(row["success"]),
                  "success_final": success_final, "length": T,
                  "success_step": int(row["success_step"]) if row["success"] else None, "grasp_step": grasp,
                  "mean": float(df.value_mean.mean()), "final": float(df.value_mean.iloc[-1])}
        parts = cache_paths(name)["parts"]
        stem = parts / _part_name(row)
        if maps is not None:
            rng = np.random.default_rng([zlib.crc32(row["task"].encode()), int(row["episode"])])
            flat = maps.reshape(-1)
            sample = flat[rng.integers(0, flat.size, config.SCALE_SAMPLES_PER_EPISODE)].astype(np.float32)
            _write_atomic(stem.with_suffix(".npy"), lambda p: np.save(open(p, "wb"), sample))
        _write_atomic(stem.with_suffix(".json"), lambda p: p.write_text(json.dumps(ep_row)))
        # the frames part is written last: its presence marks the episode as done
        _write_atomic(stem.with_suffix(".parquet"), lambda p: df.to_parquet(p, index=False))
    return ep.load_s


def _finalize(name: str, index: pd.DataFrame):
    p = cache_paths(name)
    m = metrics.get(name)
    stems = [p["parts"] / _part_name(r) for r in index.to_dict("records")]
    frames = pd.concat([pd.read_parquet(s.with_suffix(".parquet")) for s in stems], ignore_index=True)
    eps = pd.DataFrame([json.loads(s.with_suffix(".json").read_text()) for s in stems])
    for col in ("success_step", "grasp_step"):
        eps[col] = eps[col].astype("Int64")
    # colormap scale from closed-loop episodes only, so it does not change when a replay run is added
    # (dream drift is larger and saturates the scale, which is visible on purpose)
    closed = [s for s, r in zip(stems, index.replay) if not r] or stems
    samples = [np.load(s.with_suffix(".npy")) for s in closed if s.with_suffix(".npy").exists()]
    if samples:
        allv = np.concatenate(samples)
        p1, p99 = (float(x) for x in np.percentile(allv, [1, 99]))
    else:
        p1 = p99 = None
    scale = {"metric": name, "kind": m.kind, "lower_is_better": m.lower_is_better, "p1": p1, "p99": p99,
             "runs": sorted(index.run.unique()), "scale_from": "closed-loop episodes",
             "n_episodes": len(eps), "n_frames": len(frames),
             "views": {f"view{i}": v for i, v in enumerate(config.VIEWS)}}
    _write_atomic(p["frames"], lambda q: frames.to_parquet(q, index=False))
    _write_atomic(p["episodes"], lambda q: eps.to_parquet(q, index=False))
    _write_atomic(p["scale"], lambda q: q.write_text(json.dumps(scale, indent=1)))
    shutil.rmtree(p["parts"])
    if not any(p["parts"].parent.iterdir()):
        p["parts"].parent.rmdir()
    print(f"[{name}] wrote {p['frames'].name} ({len(frames)} rows), {p['episodes'].name} ({len(eps)} episodes), "
          f"scale p1={p1} p99={p99}")


def _cache_matches(name: str, index: pd.DataFrame) -> bool:
    eps = pd.read_parquet(cache_paths(name)["episodes"])
    if "replay" not in eps:
        return not index.replay.any()
    key = ["task", "state", "rollout", "episode"]
    return sorted(map(tuple, eps[key].to_numpy().tolist())) == sorted(map(tuple, index[key].to_numpy().tolist()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", action="append", default=[], help="metric name (repeatable)")
    ap.add_argument("--all", action="store_true", help="all registered metrics")
    ap.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--list", action="store_true", help="list registered metrics and exit")
    args = ap.parse_args(argv)

    if args.list:
        for m in metrics.REGISTRY.values():
            print(f"{m.name:20s} {m.kind:5s} {'lower' if m.lower_is_better else 'higher'} is better  {m.description}")
        return 0
    names = sorted(metrics.REGISTRY) if args.all else args.metric
    if not names:
        ap.error("give --metric NAME or --all")
    for n in names:
        metrics.get(n)

    index = data.build_full_index()  # raises for held-out / non-allowlisted runs
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if config.RUNS_ROOT.resolve() in config.CACHE_DIR.resolve().parents:
        sys.exit("refusing: cache dir is inside the runs directory")

    todo_metrics = []
    for n in names:
        p = cache_paths(n)
        if p["frames"].exists() and not args.force:
            if _cache_matches(n, index):
                print(f"[{n}] cache exists, skipping (use --force to recompute)")
                continue
            print(f"[{n}] cache does not match the episode index (new run?), recomputing")
        if args.force and p["parts"].exists():
            shutil.rmtree(p["parts"])
        p["parts"].mkdir(parents=True, exist_ok=True)
        todo_metrics.append(n)
    if not todo_metrics:
        return 0

    jobs = []
    for row in index.to_dict("records"):
        missing = [n for n in todo_metrics
                   if not (cache_paths(n)["parts"] / _part_name(row)).with_suffix(".parquet").exists()]
        if missing:
            jobs.append((row, missing))
    print(f"{len(index)} episodes, {len(index) - len(jobs)} already done, metrics {todo_metrics}, "
          f"{args.workers} workers")
    if jobs:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(_process_episode, row, ms) for row, ms in jobs]
            for f in tqdm(as_completed(futs), total=len(futs), unit="ep"):
                f.result()
    for n in todo_metrics:
        _finalize(n, index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
