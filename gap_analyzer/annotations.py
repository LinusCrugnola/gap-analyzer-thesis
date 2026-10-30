"""Per-episode user annotations: a star (marks an interesting episode) and a free-text note.

Stored in one JSON file outside the run directories (which stay read-only) and outside the
metric cache (so recomputing the cache keeps them):
    config.ANNOTATIONS_FILE = {"<run>/<task>/epNNN": {"starred": true, "note": "...", "task": ..., "state": ...,
                                                      "rollout": ..., "replay": ..., "updated": "..."}}
Keyed by run, task and episode number, which do not change when the index or cache is rebuilt.
Written atomically (tmp file + rename) after every change.
"""
import json
import os
import threading
import time

from . import config

_lock = threading.Lock()
_data: dict | None = None


def episode_key(row) -> str:
    return f"{row['run']}/{row['task']}/ep{int(row['episode']):03d}"


def _load() -> dict:
    global _data
    if _data is None:
        p = config.ANNOTATIONS_FILE
        _data = json.loads(p.read_text()) if p.exists() else {}
    return _data


def get(key: str) -> dict:
    with _lock:
        a = _load().get(key, {})
    return {"starred": bool(a.get("starred", False)), "note": a.get("note", "")}


def set_(key: str, row, starred: bool | None = None, note: str | None = None) -> dict:
    """Update the star and/or note of one episode; an episode with neither is removed from the file."""
    with _lock:
        d = _load()
        a = dict(d.get(key, {}))
        if starred is not None:
            a["starred"] = bool(starred)
        if note is not None:
            a["note"] = note.strip()
        if a.get("starred") or a.get("note"):
            a.update(task=row["task"], state=int(row["state"]), rollout=int(row["rollout_idx"]),
                     replay=bool(row["replay"]), updated=time.strftime("%Y-%m-%d %H:%M:%S"))
            d[key] = a
        else:
            d.pop(key, None)
        p = config.ANNOTATIONS_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
        os.replace(tmp, p)
    return get(key)
