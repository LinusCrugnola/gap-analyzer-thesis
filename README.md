# Gap Analyzer

Local Dash app to explore the imagination-vs-reality gap of DeVA rollouts in RoboCasa, with
pluggable gap metrics. Reads the dev run
`/data/crugnola/sim/deva/output/runs/20261027_1743_gap_dev_h16` and, if present, an open-loop
replay run `<YYYYMMDD_HHMM>_gap_replay_h16` next to it (read-only; see "Replay runs").

The held-out run is never loaded: `config.py` has a hard-coded allowlist containing only the dev
run (plus run names matching the replay pattern), and any path containing "heldout" is refused
(also after resolving symlinks). Every replay episode must have a closed-loop twin in the dev run
on (task, initial state, policy seed), otherwise the index refuses the run, so no state outside
the dev run can come in through a replay run.
Nothing is written into the run directory or the DeVA repo. The only output goes to `cache/`.
No DeVA code is imported.

## Setup (once)

```bash
cd /data/crugnola/sim/gap_analyzer
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The app uses its own `.venv`. Don't install anything into the DeVA venv.

## Run

```bash
cd /data/crugnola/sim/gap_analyzer
.venv/bin/python -m gap_analyzer.compute --metric pixel_mse --metric pixel_l1   # ~3-5 min, once per metric
.venv/bin/python -m gap_analyzer.app                                            # then open http://127.0.0.1:8050
```

compute options: `--all` (all registered metrics), `--force` (recompute), `--workers N`
(default 8, about 1 GB RAM per worker), `--list` (show registered metrics).
compute is resumable: if it is interrupted, rerun the same command and finished episodes are skipped.
app options: `--port`, `--host`.

## Data layout (what the app expects)

The run has no videos. Each episode in `<run>/base/<Task>/` stores:

| file | content |
|---|---|
| `epNNN_success_{true,false}_frames.npz` | `sim` (real) and `pred` (imagined), both (T,256,256,3) uint8, one frame per env step; `step` (1..T), `chunk` (action chunk id per frame), `query_steps` |
| `epNNN_..._meta.json` | task, initial_state_id, rollout_idx, policy_seed, success, success_step, episode_length, ... |
| `epNNN_..._traj.npz` | actions, eef pose, `task_obj_pos_obj` (T+1,3), contact flags, ... |

Each 256x256 frame is a 2x2 mosaic of 128x128 views: top-left = left, top-right = right,
bottom-left = wrist, bottom-right = unused. Real and imagined frames are aligned 1:1.
Imagined frames of consecutive chunks (16 steps each, `action_horizon=16`) are concatenated.
For failures `success_step` is null and the episode runs to the max length (rounded up to a whole chunk).

## Replay runs (open loop, "dream + replay")

A replay run (`open_loop: "replay"` in meta.json; missing key = `"off"` = closed loop) feeds the
model the real observation only at query 0 and its own predicted frame afterwards. The simulator
executes the dreamed actions but the model never sees it again. So in these episodes `pred` is
the dream, `sim` is the uncorrected simulator, and the gap metric measures accumulated drift.
`success` / `success_step` are replay success (the simulator succeeded at some point, not the
dream). Episodes run to the max length; whether the success held is `task_success` at the last
step of traj.npz (cache column `success_final`).

The newest run under `RUNS_ROOT` matching `^\d{8}_\d{4}_gap_replay_h16$` is used
(`GAP_ANALYZER_REPLAY_RUN=<name>` picks one). Its episodes are merged into the dev groups with the
same (task, initial state): replay rollout r gets the rollout id 100 + r and is shown as
"r0 dream" next to its closed-loop twin "r0". Chunk 0 of the two is identical.

When a replay run appears, rerun compute for each metric: it notices that the cache does not
match the episode index and recomputes (dev + replay, a few minutes). The colormap scale is still
taken from closed-loop episodes only, so dream error maps saturate where the drift is larger.

Events: chunk boundaries come from `chunk`. The success step comes from meta. The grasp step is
the first step where the task object is 3 cm above its initial height (`config.GRASP_LIFT_M`).

## Cache

`cache/<metric>.parquet` has one row per (episode, frame) with the columns task, state, rollout,
episode, frame (0-based), step (1-based env step), t_norm (= step / T), chunk, value_view0..2
(view0 = left, view1 = right, view2 = wrist), value_mean (mean over views), success, replay.

`cache/<metric>_episodes.parquet` has one row per episode: mean (over time), final (last frame),
success, success_final (task_success at the last step), length, success_step, grasp_step,
replay, rollout_idx (the run's own rollout index; `rollout` is 100 + rollout_idx for replay).

Caches written before replay support have no replay columns; the app treats them as closed loop.

`cache/<metric>_scale.json` holds the 1st/99th percentile of the error-map values over the dev
run (estimated from 100k random samples per episode), used as the fixed colormap range, plus
`kind` and `lower_is_better`.

Error maps are never written to disk. They are recomputed in RAM when a group is opened.

## Adding a metric

Add a decorated function to any module in `gap_analyzer/metrics/` (new modules are imported
automatically):

```python
import numpy as np
from . import frame_metric, clip_metric

# Per-frame: real, imag are (H,W,3) uint8 -> (scalar, error map (H,W) or None).
# With batched=True it must also accept (N,H,W,3) and return ((N,), (N,H,W) or None);
# it is then called on batches of 128 frames per view.
@frame_metric("pixel_mse", lower_is_better=True, batched=True)
def pixel_mse(real, imag):
    d = real.astype(np.float32) - imag.astype(np.float32)
    m = (d * d).mean(-1)
    return m.mean((-2, -1)), m

# Per-clip (temporal): called once per action chunk and view with (T,H,W,3) uint8 clips
# -> ((T,) per-frame scalars, (T,H,W) error maps or None).
@clip_metric("my_temporal_metric", lower_is_better=True)
def my_temporal_metric(real_clip, imag_clip):
    ...
```

Then run `python -m gap_analyzer.compute --metric <name>` and restart the app (or reload the
page). The metric appears in the dropdown. Per-view values are averaged over the 3 views.
If a metric returns no error maps, the error row stays grey.

## Using the app

* **Panel A** lists one row per (task, initial state): the success count k/8, and the mean and
  final metric averaged over the 8 rollouts. Green rows have a mixed outcome: a closed-loop
  success rate between 30% and 70% (`config.MIXED_OUTCOME`), i.e. 3-5 of 8. All columns use
  the closed-loop rollouts only; "replay success" shows the replay rollouts of the group. "Within-group rank" is the fraction of
  (success, failure) pairs where the success has the better episode-mean metric (respects
  lower_is_better, ties count 0.5, empty for uniform groups). "Spread" is the std of the 8
  episode means. Click a row to open the group.
* **Panel B** plots the metric over time for the 8 rollouts (green = success, red = failure,
  ★ = success step, ◆ = grasp). Replay rollouts are dotted lines named "r0 dream". You can switch between absolute steps and normalized time.
  Dashed lines are chunk boundaries (shown in absolute mode only). Click a line to open the rollout.
  The "rollout stats up to step" slider under the plot (absolute mode only) moves a purple
  cursor in the plot (steps after it are greyed out). The avg / max in the Panel C rollout list
  are then computed over steps 1…cursor only. Rollouts that ended earlier use all their steps,
  shown in the "steps" column. The list updates when you release the slider. You can also type
  a step into the box and press Enter. Opening a new group resets the cursor to the end.
* **Panel C** has a rollout list to the left of the video: one row per rollout, green for
  success and red for failure (replay success for dream rows), with the loop type (closed or
  open (dream)), steps used, avg (mean over time) and max (max over time) of the view-averaged metric
  (over steps 1…cursor, see Panel B). Click a header
  to sort, click a row to view that rollout. When a group is opened, the list is sorted by avg
  and the rollout with the lowest avg is shown right away. Switching the metric keeps the
  selected rollout. The shown rollout is bold with a darker colour. A banner above the list says
  whether it is closed loop or full imagination (open-loop dream), names its twin and whether
  a replay success held to the end; the middle viewer row is then labelled "dream (open loop)".
  Next to the list it shows real / imagined / error maps for the 3 views, a metric plot with the
  current frame marker, and slider, |<, <, play/pause, >, >| and fps controls.
  Real time is 20 fps: RoboCasa runs the controller at `control_freq=20` and one frame is stored
  per env step. DeVA's `FPS=16` is only the video model's frame-rate conditioning. The default is
  20 fps (1x). The ¼× ½× 1× 2× buttons set the fps, and the text next to them shows the speed
  relative to real time. Playback follows the wall clock. Frames are decoded off-screen, drawn on
  a canvas and prefetched 12 frames ahead, so there are no blank frames between images. A frame
  is only skipped if the server can't keep up.

Opening a group decodes all 8 rollouts (9 with a replay rollout) into RAM in a background thread and computes their
error maps (about 6-11 s per group from the HDD when not in the page cache, about 3 s when cached; 0.9-1.7 GB). The 2 most recently opened groups are kept.
A rollout you click is loaded first. Switching the metric recomputes the error maps of the
groups in memory. The header shows the preload time, the server time per frame and the process
RSS. Below the viewer the browser-side step time (request to display) is shown.

## Files

```
gap_analyzer/config.py         paths, allowlist + heldout check, defaults
gap_analyzer/data.py           episode index (meta.json, dev + replay with twin check), npz loading, GroupStore
gap_analyzer/events.py         chunk boundaries, success step, grasp step
gap_analyzer/metrics/          registry (__init__), interfaces (base), pixel_mse / pixel_l1 (pixel)
gap_analyzer/compute.py        CLI: metric over all dev episodes -> parquet cache
gap_analyzer/render.py         composite image (3x3 tiles, colormap) -> JPEG
gap_analyzer/app.py            Dash app; frames are served at /frame/<metric>/<task>/<state>/<rollout>/<i>.jpg
gap_analyzer/assets/           CSS + viewer JS (frame loader that drops stale frames)
```

For testing on another machine with a copy of the dev run, set
`GAP_ANALYZER_RUNS_ROOT=/path/to/runs`. The allowlist and heldout checks still apply.
