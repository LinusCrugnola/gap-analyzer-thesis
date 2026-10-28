"""Dash app for qualitative gap analysis of the dev run.

    python -m gap_analyzer.app [--port 8050] [--host 127.0.0.1]

Panel A: group overview (task, initial state). Panel B: metric over time for the 8 rollouts of a
group. Panel C: rollout viewer (real / imagined / error maps) with a synced metric plot.
Viewer frames are composed on the server from in-memory buffers and served as JPEG at
/frame/<metric>/<task>/<state>/<rollout>/<i>.jpg.
"""
import argparse
import json
import statistics
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme
from flask import Response, abort

from . import config, data, render
from .compute import cache_paths
from .metrics import REGISTRY

GREEN, RED, GREY = "#1a9850", "#d73027", "#b0b4ba"

INDEX = data.build_index()  # raises for held-out / non-allowlisted runs
STORE = data.GroupStore(INDEX)
FRAME_TIMES = deque(maxlen=200)  # server time per /frame request (ms), excluding first loads


# ---------------------------------------------------------------- cache access

def cached_metrics() -> list[str]:
    return sorted(m for m in REGISTRY
                  if all(cache_paths(m)[k].exists() for k in ("frames", "episodes", "scale")))


@lru_cache(maxsize=8)
def _load_metric(metric: str, mtime: float):
    p = cache_paths(metric)
    frames = pd.read_parquet(p["frames"])
    eps = pd.read_parquet(p["episodes"])
    scale = json.loads(p["scale"].read_text())
    return frames, eps, scale


def load_metric(metric: str):
    return _load_metric(metric, cache_paths(metric)["frames"].stat().st_mtime)


def group_table(metric: str, tasks) -> list[dict]:
    _, eps, scale = load_metric(metric)
    lib = scale["lower_is_better"]
    if tasks:
        eps = eps[eps.task.isin(tasks)]
    rows = []
    for (task, state), g in eps.groupby(["task", "state"], sort=True):
        s, f = g[g.success]["mean"].to_numpy(), g[~g.success]["mean"].to_numpy()
        if len(s) and len(f):
            better = (s[:, None] < f[None, :]) if lib else (s[:, None] > f[None, :])
            ties = s[:, None] == f[None, :]
            rank = float((better + 0.5 * ties).mean())
        else:
            rank = None
        rows.append({"id": f"{task}|{state}", "task": task, "state": int(state),
                     "success": f"{int(g.success.sum())}/{len(g)}", "mean": g["mean"].mean(),
                     "final": g["final"].mean(), "rank": rank, "spread": g["mean"].std()})
    return rows


# ---------------------------------------------------------------- figures

def _chunk_shapes(steps: np.ndarray) -> list[dict]:
    return [dict(type="line", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1,
                 line=dict(color="#c9ced6", width=1, dash="dash"), layer="below") for x in steps]


def cutoff_shapes(cutoff: int, xmax: int) -> list[dict]:
    """Stats cursor in the group plot: shapes[0] = line at the cutoff, shapes[1] = excluded steps."""
    return [dict(type="line", xref="x", yref="paper", x0=cutoff, x1=cutoff, y0=0, y1=1,
                 line=dict(color="#5b2a86", width=2.5)),
            dict(type="rect", xref="x", yref="paper", x0=cutoff, x1=xmax + 1, y0=0, y1=1,
                 fillcolor="rgba(120,120,130,0.13)", line=dict(width=0), layer="below")]


def group_figure(metric: str, task: str, state: int, xmode: str, selected, cutoff: int | None = None) -> go.Figure:
    frames, eps, scale = load_metric(metric)
    fr = frames[(frames.task == task) & (frames.state == state)]
    ep = eps[(eps.task == task) & (eps.state == state)].set_index("rollout")
    norm = xmode == "norm"
    xcol = "t_norm" if norm else "step"
    fig = go.Figure()
    for r, g in fr.groupby("rollout"):
        e = ep.loc[r]
        color = GREEN if e.success else RED
        sel = selected is not None and int(selected) == int(r)
        fig.add_trace(go.Scattergl(
            x=g[xcol], y=g.value_mean, mode="lines", name=f"r{r} {'✓' if e.success else '✗'}",
            line=dict(color=color, width=3.5 if sel else 1.6), opacity=1 if sel or selected is None else 0.55,
            customdata=np.full(len(g), r), legendgroup=str(r),
            hovertemplate=f"rollout {r} ({'success' if e.success else 'failure'})<br>"
                          f"{'t' if norm else 'step'} %{{x}}<br>{metric} %{{y:.4g}}<extra></extra>"))
        for col, symbol, label in (("success_step", "star", "success"), ("grasp_step", "diamond", "grasp")):
            st = e[col]
            if pd.notna(st):
                row = g[g.step == int(st)]
                if len(row):
                    fig.add_trace(go.Scatter(
                        x=row[xcol], y=row.value_mean, mode="markers", legendgroup=str(r), showlegend=False,
                        marker=dict(symbol=symbol, size=11 if symbol == "star" else 8, color=color,
                                    line=dict(color="black", width=1)),
                        customdata=[r], hovertemplate=f"rollout {r} {label} at step {int(st)}<extra></extra>"))
    shapes = []
    if not norm:
        b = fr.loc[fr.chunk.diff().fillna(0) != 0, "step"].unique()
        shapes = _chunk_shapes(np.sort(b) - 0.5)
        if cutoff is not None:
            shapes = cutoff_shapes(cutoff, int(fr.step.max())) + shapes
    fig.update_layout(
        shapes=shapes, height=380, margin=dict(l=55, r=10, t=30, b=40), template="plotly_white",
        title=dict(text=f"{task}, initial state {state}: {metric} (mean over views)", font=dict(size=13)),
        xaxis_title="normalized time" if norm else "env step", yaxis_title=metric,
        legend=dict(orientation="v", font=dict(size=11)), hovermode="closest", uirevision=f"{task}{state}{xmode}")
    return fig


def episode_figure(metric: str, task: str, state: int, rollout: int):
    frames, eps, scale = load_metric(metric)
    fr = frames[(frames.task == task) & (frames.state == state) & (frames.rollout == rollout)]
    e = eps[(eps.task == task) & (eps.state == state) & (eps.rollout == rollout)].iloc[0]
    fig = go.Figure()
    for v, (name, c) in enumerate(zip(config.VIEWS, ("#4c78a8", "#f58518", "#54a24b"))):
        fig.add_trace(go.Scatter(x=fr.step, y=fr[f"value_view{v}"], mode="lines", name=name,
                                   line=dict(color=c, width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=fr.step, y=fr.value_mean, mode="lines", name="mean",
                               line=dict(color="#222", width=2.2)))
    steps = fr.step.to_numpy()
    marker = dict(type="line", xref="x", yref="paper", x0=steps[0], x1=steps[0], y0=0, y1=1,
                  line=dict(color="#7b3294", width=2))
    shapes = [marker] + _chunk_shapes(steps[np.flatnonzero(np.diff(fr.chunk.to_numpy())) + 1] - 0.5)
    for col, color, label in (("success_step", GREEN, "success"), ("grasp_step", "#2c7fb8", "grasp")):
        if pd.notna(e[col]):
            shapes.append(dict(type="line", xref="x", yref="paper", x0=int(e[col]), x1=int(e[col]), y0=0, y1=1,
                               line=dict(color=color, width=2, dash="dot")))
            fig.add_annotation(x=int(e[col]), y=1, yref="paper", text=label, showarrow=False,
                               font=dict(size=10, color=color), yanchor="bottom")
    fig.update_layout(shapes=shapes, height=230, margin=dict(l=55, r=10, t=24, b=34), template="plotly_white",
                      xaxis_title="env step", yaxis_title=metric, hovermode="x unified",
                      legend=dict(orientation="h", y=1.18, x=0, xanchor="left", font=dict(size=11)))
    info = {"base": f"/frame/{metric}/{task}/{state}/{rollout}", "n": int(len(fr)),
            "steps": steps.tolist(), "chunks": fr.chunk.tolist(), "success": bool(e.success)}
    return fig, info


# ---------------------------------------------------------------- layout

ROLLOUT_SORT_DEFAULT = [{"column_id": "mean", "direction": "asc"}]


def rollout_rows(metric: str, task: str, state: int, cutoff: int | None = None) -> list[dict]:
    """One row per rollout of a group: outcome, and mean / max of the view-averaged metric over
    steps 1..cutoff (whole episode if cutoff is None; rollouts that ended earlier use all their steps)."""
    frames, eps, _ = load_metric(metric)
    fr = frames[(frames.task == task) & (frames.state == state)]
    if cutoff is not None:
        fr = fr[fr.step <= cutoff]
    stats = fr.groupby("rollout").value_mean.agg(["mean", "max", "size"])
    g = eps[(eps.task == task) & (eps.state == state)].sort_values("rollout")
    return [{"id": int(r.rollout), "rollout": int(r.rollout), "success": int(r.success),
             "outcome": "✓" if r.success else "✗", "steps": int(stats.loc[r.rollout, "size"]),
             "mean": float(stats.loc[r.rollout, "mean"]), "max": float(stats.loc[r.rollout, "max"])}
            for r in g.itertuples()]


def group_length(task: str, state: int) -> int:
    g = INDEX[(INDEX.task == task) & (INDEX.state == state)]
    return int(g.length.max())


def _default_metric(metrics_):
    return "pixel_mse" if "pixel_mse" in metrics_ else (metrics_[0] if metrics_ else None)


def _btn(id_, label, title):
    return html.Button(label, id=id_, title=title, n_clicks=0)


def serve_layout():
    metrics_ = cached_metrics()
    tasks = sorted(INDEX.task.unique())
    fmt = Format(precision=2, scheme=Scheme.fixed)
    columns = [
        {"name": "task", "id": "task"}, {"name": "state", "id": "state", "type": "numeric"},
        {"name": "success", "id": "success"},
        {"name": "mean", "id": "mean", "type": "numeric", "format": fmt},
        {"name": "final", "id": "final", "type": "numeric", "format": fmt},
        {"name": "within-group rank", "id": "rank", "type": "numeric", "format": fmt},
        {"name": "spread (std)", "id": "spread", "type": "numeric", "format": fmt},
    ]
    return html.Div(className="page", children=[
        html.Div(className="header", children=[
            html.H2(f"Gap Analyzer · {config.DEFAULT_RUN}"),
            html.Div(className="ctl", children=["metric", dcc.Dropdown(
                id="metric", options=metrics_, value=_default_metric(metrics_), clearable=False,
                style={"width": 200})]),
            html.Div(className="ctl", children=["task filter", dcc.Dropdown(
                id="tasks", options=tasks, value=[], multi=True, placeholder="all tasks", style={"width": 480})]),
            html.Div(id="status", className="status"),
        ]),
        html.Div(className="hint", style={"marginBottom": 8}, children=(
            "" if metrics_ else "No cached metrics. Run: python -m gap_analyzer.compute --metric pixel_mse")),
        html.Div(className="panel", children=[
            html.H3("A · Groups (task, initial state)"),
            html.Div(className="hint", children=(
                "Click a row to open the group. within-group rank = fraction of (success, failure) pairs where the "
                "success has the better episode-mean metric (empty if the group is all success or all failure). "
                "spread = std of the 8 episode means.")),
            dash_table.DataTable(
                id="groups", columns=columns, data=[], sort_action="native", page_action="none",
                sort_by=[], style_table={"maxHeight": "340px", "overflowY": "auto"}, fixed_rows={"headers": True},
                style_cell={"fontSize": 13, "padding": "3px 8px", "fontFamily": "system-ui", "minWidth": 80},
                style_cell_conditional=[{"if": {"column_id": "task"}, "textAlign": "left", "minWidth": 200}],
                style_header={"fontWeight": 600, "backgroundColor": "#f0f2f5"}),
        ]),
        html.Div(className="panel", children=[
            html.Div(style={"display": "flex", "gap": 16, "alignItems": "center"}, children=[
                html.H3("B · Group"),
                dcc.RadioItems(id="xmode", options=[{"label": " absolute steps", "value": "abs"},
                                                    {"label": " normalized time", "value": "norm"}],
                               value="abs", inline=True, inputStyle={"marginLeft": 10}),
                html.Span(id="group-info", className="hint"),
            ]),
            dcc.Graph(id="group-plot", config={"displaylogo": False}, figure=go.Figure(
                layout=dict(height=380, template="plotly_white",
                            annotations=[dict(text="select a group in panel A", showarrow=False,
                                              xref="paper", yref="paper", x=0.5, y=0.5)]))),
            html.Div(className="cutoff-row", children=[
                html.Span("rollout stats up to step", className="hint"),
                html.Div(className="cutoff-slider", children=dcc.Slider(
                    id="cutoff", min=1, max=1, step=1, value=1, marks=None, updatemode="mouseup",
                    disabled=True)),
                html.Span(id="cutoff-label", className="counter"),
            ]),
            html.Div(className="hint", children=(
                "green = success, red = failure. ★ success step, ◆ grasp (object lifted 3 cm). Dashed lines = chunk "
                "boundaries (absolute mode only; in normalized time they differ per rollout). Click a line to view it. "
                "The purple cursor (slider above, absolute steps only) sets up to which step avg / max in the "
                "rollout list of panel C are computed.")),
        ]),
        html.Div(className="panel", children=[
            html.H3("C · Rollout"),
            html.Div(className="panel-c-top", children=[
                html.Div(className="rollout-list", children=[
                    html.Div(id="rollouts-hint", className="hint"),
                    dash_table.DataTable(
                        id="rollouts", data=[], sort_action="native", sort_by=ROLLOUT_SORT_DEFAULT,
                        columns=[{"name": "rollout", "id": "rollout", "type": "numeric"},
                                 {"name": "", "id": "outcome"},
                                 {"name": "steps", "id": "steps", "type": "numeric"},
                                 {"name": "avg", "id": "mean", "type": "numeric",
                                  "format": Format(precision=2, scheme=Scheme.fixed)},
                                 {"name": "max", "id": "max", "type": "numeric",
                                  "format": Format(precision=2, scheme=Scheme.fixed)}],
                        style_cell={"fontSize": 13, "padding": "4px 8px", "fontFamily": "system-ui"},
                        style_header={"fontWeight": 600, "backgroundColor": "#f0f2f5"},
                        style_cell_conditional=[{"if": {"column_id": "outcome"}, "textAlign": "center",
                                                 "fontWeight": 700, "width": "24px"}]),
                ]),
                html.Div(children=[
                    html.Div(className="col-labels", children=[html.Div(v) for v in config.VIEWS]),
                    html.Div(className="viewer-wrap", children=[
                        html.Div(className="row-labels",
                                 children=[html.Div(html.Span(t)) for t in ("real", "imagined", "error map")]),
                        html.Div(id="viewer"),
                        html.Div(className="cbar-col", children=[html.Div(className="cbar", children=[
                            html.Div(className="cbar-grad", style={"background": render.css_gradient()}),
                            html.Div(id="cbar-labels", className="cbar-labels"),
                        ])]),
                    ]),
                ]),
            ]),
            dcc.Graph(id="ep-plot", config={"displaylogo": False}, figure=go.Figure(
                layout=dict(height=230, template="plotly_white"))),
            dcc.Slider(id="frame", min=0, max=1, step=1, value=0, marks=None, updatemode="drag",
                       allow_direct_input=False),
            html.Div(className="controls", children=[
                _btn("btn-first", "|<", "first frame"), _btn("btn-prev", "<", "previous frame"),
                _btn("btn-play", "▶", "play / pause"), _btn("btn-next", ">", "next frame"),
                _btn("btn-last", ">|", "last frame"),
                html.Span("fps", className="hint", style={"marginLeft": 12}),
                dcc.Input(id="fps", type="number", value=config.DEFAULT_FPS, min=1, max=60, step=1,
                          style={"width": 60}),
                *[html.Button(label, id={"type": "speed", "factor": f}, n_clicks=0, title=f"{f}x real time")
                  for label, f in (("¼×", 0.25), ("½×", 0.5), ("1×", 1), ("2×", 2))],
                html.Span(id="speed", className="hint"),
                html.Span(id="counter", className="counter"),
            ]),
            html.Div(id="step-time", className="status", style={"marginTop": 6}),
        ]),
        dcc.Store(id="sel-group"), dcc.Store(id="rollout"), dcc.Store(id="ep-info"), dcc.Store(id="play", data={"playing": False}),
        dcc.Interval(id="play-tick", interval=1000 // config.DEFAULT_FPS, disabled=True),
        dcc.Interval(id="status-tick", interval=1000),
    ])


app = Dash(__name__, title="Gap Analyzer", assets_folder=str(Path(__file__).parent / "assets"),
           update_title=None)
app.layout = serve_layout
server = app.server


# ---------------------------------------------------------------- frame route

@server.route("/frame/<metric>/<task>/<int:state>/<int:rollout>/<int:i>.jpg")
def frame_jpeg(metric, task, state, rollout, i):
    if metric not in REGISTRY or task not in set(INDEX.task):
        abort(404)
    t0 = time.perf_counter()
    try:
        ep, maps = STORE.get(task, state, rollout, metric)
    except KeyError:
        abort(404)
    t1 = time.perf_counter()
    i = min(max(i, 0), ep.n_frames - 1)
    scale = load_metric(metric)[2]
    img = render.compose(ep.real[:, i], ep.imag[:, i], None if maps is None else maps[:, i],
                         scale["p1"] or 0.0, scale["p99"] or 1.0)
    body = render.to_jpeg(img)
    t2 = time.perf_counter()
    if t1 - t0 < 0.05:  # don't count requests that had to wait for a load
        FRAME_TIMES.append((t2 - t0) * 1000)
    return Response(body, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=600"})


# ---------------------------------------------------------------- callbacks

@app.callback(Output("groups", "data"), Input("metric", "value"), Input("tasks", "value"))
def update_table(metric, tasks):
    if not metric:
        return []
    return group_table(metric, tasks)


@app.callback(Output("sel-group", "data"), Output("groups", "style_data_conditional"),
              Input("groups", "active_cell"), prevent_initial_call=True)
def select_group(cell):
    if not cell or "row_id" not in cell:
        return no_update, no_update
    task, state = cell["row_id"].split("|")
    style = [{"if": {"filter_query": f'{{id}} = "{cell["row_id"]}"'}, "backgroundColor": "#dbe7ff"}]
    return {"task": task, "state": int(state)}, style


@app.callback(Output("group-plot", "figure"), Output("group-info", "children"),
              Output("cutoff", "max"), Output("cutoff", "value"), Output("cutoff", "disabled"),
              Input("sel-group", "data"), Input("metric", "value"), Input("xmode", "value"),
              Input("rollout", "data"), State("cutoff", "value"))
def update_group_plot(sel, metric, xmode, ro, cutoff):
    if not sel or not metric:
        return no_update, "", no_update, no_update, no_update
    rollout = ro["rollout"] if ro and (ro["task"], ro["state"]) == (sel["task"], sel["state"]) else None
    STORE.open_group(sel["task"], sel["state"], metric)  # background preload of all 8 rollouts
    instr = INDEX[(INDEX.task == sel["task"]) & (INDEX.state == sel["state"])].instruction.iloc[0]
    n = group_length(sel["task"], sel["state"])
    # a new group starts with the cursor at the end (= whole episodes); otherwise keep it
    cutoff = n if "sel-group.data" in ctx.triggered_prop_ids or not cutoff else min(int(cutoff), n)
    fig = group_figure(metric, sel["task"], sel["state"], xmode, rollout, cutoff)
    return fig, f"“{instr}”", n, cutoff, xmode != "abs"


@app.callback(Output("rollouts", "data"), Output("rollouts", "sort_by"), Output("rollout", "data"),
              Output("rollouts-hint", "children"),
              Input("sel-group", "data"), Input("metric", "value"), Input("group-plot", "clickData"),
              Input("rollouts", "active_cell"), Input("cutoff", "value"), State("cutoff", "max"),
              State("rollout", "data"), prevent_initial_call=True)
def select_rollout(sel, metric, click, cell, cutoff, cutoff_max, current):
    """Rollout list of the group; on a new group sort by mean and select the lowest-mean rollout."""
    if not sel or not metric:
        return [], no_update, None, ""
    key = {"task": sel["task"], "state": sel["state"]}
    # Dash can merge several triggers into one call, so look at all of them (a group change wins)
    trig = set(ctx.triggered_prop_ids)
    same_group = bool(current) and (current["task"], current["state"]) == (sel["task"], sel["state"])
    new_group = "sel-group.data" in trig or not same_group
    n = group_length(sel["task"], sel["state"])
    # on a new group the cursor is reset to the end by update_group_plot; don't use the old group's value
    cut = n if new_group or not cutoff else min(int(cutoff), n)
    rows = rollout_rows(metric, sel["task"], sel["state"], cut)
    hint = (f"avg / max over steps 1–{cut}" + ("" if cut >= n else " (cursor in panel B)") +
            " · click a rollout · click a header to sort")
    if new_group:
        best = min(rows, key=lambda r: r["mean"])["rollout"]
        return rows, ROLLOUT_SORT_DEFAULT, {**key, "rollout": best}, hint
    selection = no_update
    if "group-plot.clickData" in trig and click:
        selection = {**key, "rollout": int(click["points"][0]["customdata"])}
    elif "rollouts.active_cell" in trig and cell and "row_id" in cell:
        selection = {**key, "rollout": int(cell["row_id"])}
    if trig <= {"group-plot.clickData", "rollouts.active_cell"}:
        # selection only (active_cell is also cleared by sorting and by highlight_rollout: then nothing)
        return no_update, no_update, selection, no_update
    return rows, no_update, selection, hint  # metric / cursor: new numbers, keep selection and sort order


@app.callback(Output("rollouts", "style_data_conditional"), Output("rollouts", "active_cell"),
              Input("rollout", "data"))
def highlight_rollout(ro):
    styles = [{"if": {"filter_query": "{success} = 1", "column_id": "outcome"}, "color": GREEN},
              {"if": {"filter_query": "{success} = 0", "column_id": "outcome"}, "color": RED}]
    if ro:
        styles.append({"if": {"filter_query": f"{{id}} = {ro['rollout']}"},
                       "backgroundColor": "#dbe7ff", "fontWeight": 600})
    return styles, None  # clear the active cell so clicking the same row again re-selects it


@app.callback(Output("ep-plot", "figure"), Output("ep-info", "data"), Output("frame", "max"),
              Output("frame", "value"), Output("cbar-labels", "children"), Output("play", "data"),
              Input("rollout", "data"), Input("metric", "value"),
              State("frame", "value"), State("ep-info", "data"))
def update_episode(ro, metric, frame, prev_info):
    if not ro or not metric:
        return go.Figure(layout=dict(height=230, template="plotly_white")), None, 1, 0, [], {"playing": False}
    # load the clicked rollout first, ahead of the rest of the group
    task, state, rollout = ro["task"], ro["state"], ro["rollout"]
    threading.Thread(target=STORE.get, args=(task, state, rollout, metric), daemon=True).start()
    fig, info = episode_figure(metric, task, state, rollout)
    scale = load_metric(metric)[2]
    labels = [html.Div(f"{scale['p99']:.4g}" if scale["p99"] is not None else "n/a"),
              html.Div(metric, style={"writingMode": "vertical-rl"}),
              html.Div(f"{scale['p1']:.4g}" if scale["p1"] is not None else "")]
    same_episode = prev_info and prev_info["base"].split("/", 3)[3] == info["base"].split("/", 3)[3]
    frame = min(frame or 0, info["n"] - 1) if same_episode else 0  # keep the frame on a metric switch
    fig.layout.shapes[0].update(x0=info["steps"][frame], x1=info["steps"][frame])
    return fig, info, info["n"] - 1, frame, labels, {"playing": False}


@app.callback(Output("status", "children"), Input("status-tick", "n_intervals"))
def update_status(_):
    lines = []
    for g in STORE.status():
        pre = f"{g['preload_s']:.1f} s" if g["preload_s"] is not None else "loading…"
        lines.append(f"{g['task']} s{g['state']}: {g['loaded']}/{g['total']} decoded, preload {pre}, "
                     f"{g['gb']:.2f} GB")
    if FRAME_TIMES:
        lines.append(f"server frame time: median {statistics.median(FRAME_TIMES):.1f} ms "
                     f"(n={len(FRAME_TIMES)}) · RSS {data.rss_gb():.1f} GB")
    return "\n".join(lines)


app.clientside_callback(
    ("""
    function(nFirst, nPrev, nPlay, nNext, nLast, tick, fps, frame, maxv, play) {
        const nu = window.dash_clientside.no_update;
        const trig = (window.dash_clientside.callback_context.triggered[0] || {}).prop_id || "";
        fps = Math.max(1, Math.min(60, fps || %d));
        const N = maxv || 0;
        let f = frame || 0;
        play = play || {playing: false};
        const stop = {playing: false};
        if (trig.startsWith("btn-first")) { f = 0; play = stop; }
        else if (trig.startsWith("btn-prev")) { f = Math.max(0, f - 1); play = stop; }
        else if (trig.startsWith("btn-next")) { f = Math.min(N, f + 1); play = stop; }
        else if (trig.startsWith("btn-last")) { f = N; play = stop; }
        else if (trig.startsWith("btn-play")) {
            if (play.playing) { play = stop; }
            else { if (f >= N) f = 0; play = {playing: true, t0: Date.now(), f0: f}; }
        } else if (trig.startsWith("fps")) {
            if (play.playing) play = {playing: true, t0: Date.now(), f0: f};
        } else if (trig.startsWith("play-tick")) {
            if (!play.playing) return [nu, nu, true, nu, "▶"];
            // frame from wall clock: if rendering is slower than fps, frames are dropped
            const target = play.f0 + Math.floor((Date.now() - play.t0) * fps / 1000);
            if (target >= N) { f = N; play = stop; }
            else if (target === f) return [nu, nu, false, nu, "❚❚"];
            else f = target;
        }
        return [f, play, !play.playing, Math.round(1000 / fps), play.playing ? "❚❚" : "▶"];
    }
    """ % config.DEFAULT_FPS),
    Output("frame", "value", allow_duplicate=True), Output("play", "data", allow_duplicate=True),
    Output("play-tick", "disabled"), Output("play-tick", "interval"), Output("btn-play", "children"),
    Input("btn-first", "n_clicks"), Input("btn-prev", "n_clicks"), Input("btn-play", "n_clicks"),
    Input("btn-next", "n_clicks"), Input("btn-last", "n_clicks"), Input("play-tick", "n_intervals"),
    Input("fps", "value"),
    State("frame", "value"), State("frame", "max"), State("play", "data"),
    prevent_initial_call=True,
)

app.clientside_callback(
    ("""
    function(frame, info) {
        if (!info) { window.gapViewer.clear(); return ""; }
        const i = Math.max(0, Math.min(frame || 0, info.n - 1));
        window.gapViewer.show(`${info.base}/${i}.jpg`);
        const ahead = [];
        for (let k = i + 1; k < Math.min(info.n, i + 1 + %d); k++) ahead.push(`${info.base}/${k}.jpg`);
        window.gapViewer.prefetch(ahead);
        window.gapViewer.moveMarker(info.steps[i]);
        return `frame ${i + 1}/${info.n} · step ${info.steps[i]} · chunk ${info.chunks[i]}` +
               (info.success ? " · success" : " · failure");
    }
    """ % config.PREFETCH_FRAMES),
    Output("counter", "children"), Input("frame", "value"), Input("ep-info", "data"),
)

app.clientside_callback(
    """
    function(n, fps) {
        const trig = window.dash_clientside.callback_context.triggered_id;
        const preset = trig && trig.type === "speed";
        if (preset) fps = Math.round(%(rt)d * trig.factor);
        const x = (fps || 0) / %(rt)d;
        return [preset ? fps : window.dash_clientside.no_update,
                `= ${x.toFixed(2).replace(/\\.?0+$/, "")}× real time (sim: %(rt)d steps/s, 1 frame per step)`];
    }
    """ % {"rt": config.REALTIME_FPS},
    Output("fps", "value"), Output("speed", "children"),
    Input({"type": "speed", "factor": ALL}, "n_clicks"), Input("fps", "value"),
)

app.clientside_callback(
    """
    function(drag, value, maxv, xmode) {
        const x = drag != null ? drag : value;
        if (xmode !== "abs") return "(absolute steps only)";
        if (x == null || !maxv || maxv <= 1) return "";
        const gd = document.querySelector("#group-plot .js-plotly-plot");
        const sh = gd && gd.layout && gd.layout.shapes;
        if (window.Plotly && sh && sh.length >= 2 && sh[0].type === "line" && sh[1].type === "rect") {
            window.Plotly.relayout(gd, {"shapes[0].x0": x, "shapes[0].x1": x, "shapes[1].x0": x});
        }
        return `${x} / ${maxv}` + (x >= maxv ? " (whole episodes)" : "");
    }
    """,
    Output("cutoff-label", "children"),
    Input("cutoff", "drag_value"), Input("cutoff", "value"), State("cutoff", "max"), State("xmode", "value"),
)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gap Analyzer Dash app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    print(f"{len(INDEX)} episodes in {config.DEFAULT_RUN}, cached metrics: {cached_metrics()}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
