"""The timeline (Gantt) view: what every thread was doing, when."""

from __future__ import annotations

import os

import numpy as np
from matplotlib.collections import PolyCollection

from ..colors import ColorMap, resolve
from ..model import Trace
from ..theme import get_theme
from ._common import (
    add_grid,
    add_legend,
    ink_on,
    new_axes,
    save_figure,
    style_axes,
    time_scale,
    truncate,
)

__all__ = ["timeline"]

#: Fraction of a row occupied by its bars; the rest is the gap between rows.
_ROW_HEIGHT = 0.78
#: Below these sizes a bar cannot carry a legible label.
_MIN_LABEL_PX = 26.0
_MIN_LANE_PX = 9.0
#: Breathing room kept between a label and the edges of its bar.
_LABEL_PADDING_PX = 6.0
#: Cap on direct labels; beyond this the legend is the faster read.
_MAX_LABELS = 400


def timeline(
    trace: Trace,
    *,
    ax=None,
    colors: ColorMap | None = None,
    color_by: str = "region",
    max_colors: int = 8,
    nesting: str = "overlay",
    theme=None,
    figsize: tuple[float, float] | None = None,
    time_unit: str | None = None,
    legend: str | bool = "right",
    bar_labels: bool | str = "auto",
    markers: bool = False,
    show_idle: bool = True,
    title: str | None = None,
    save: str | os.PathLike | None = None,
    max_intervals: int | None = 200_000,
):
    """Draw one row per location, one bar per region interval.

    Parameters
    ----------
    trace
        The trace to draw. Filter it first (:meth:`~otf2viz.Trace.filter`,
        :meth:`~otf2viz.Trace.select_time`) rather than drawing everything —
        a timeline of a long run is unreadable at any size.
    colors, color_by, max_colors
        Pass an explicit :class:`~otf2viz.ColorMap` to keep colours stable
        across several plots, or let one be derived by ``"region"``,
        ``"paradigm"`` or ``"role"``.
    nesting
        ``"overlay"`` paints nested regions on top of their parent, the usual
        trace-viewer look. ``"lanes"`` gives each call-stack depth its own
        sub-lane (a flame graph per thread). ``"flat"`` keeps only the
        outermost regions.
    bar_labels
        Name regions directly inside bars that are wide enough. ``"auto"``
        (the default) does this whenever it fits, which is what keeps the plot
        readable without hunting through the legend.
    markers
        Mark ``ThreadFork`` / ``ThreadJoin`` events with a caret.
    save
        Also write the figure to this path, in the format named by the file
        extension (``.png``, ``.svg``, ``.pdf``, ...). With ``ax`` given this
        saves that axis' whole figure, panels and all.

    Returns
    -------
    matplotlib.axes.Axes
    """
    theme = get_theme(theme)
    if max_intervals is not None and len(trace) > max_intervals:
        raise ValueError(
            f"{len(trace):,} intervals is too many to draw legibly. Narrow it down "
            "first, e.g. trace.select_time(0, 0.01), trace.filter(max_depth=1) or "
            "trace.filter(min_duration=1e-5) — or pass max_intervals=None to "
            "override."
        )
    if nesting not in ("overlay", "lanes", "flat"):
        raise ValueError(f"unknown nesting {nesting!r}; expected overlay, lanes or flat")

    view = trace.filter(max_depth=0) if nesting == "flat" else trace
    colormap = resolve(colors, trace, by=color_by, max_colors=max_colors, theme=theme)

    order, row_of, boundaries = _row_positions(view)
    n_rows = len(order)
    if figsize is None:
        figsize = (11.0, max(2.0, 0.34 * n_rows + 1.4))
    fig, ax, owned = new_axes(ax, figsize, theme)
    style_axes(ax, theme)

    factor, unit = time_scale(view.duration, time_unit)
    x0 = view.start * factor
    width = np.maximum((view.stop - view.start) * factor, 0.0)
    y_row = np.array([row_of[i] for i in view.location], dtype=np.float64)

    depth = view.depth if nesting != "flat" else np.zeros(len(view), np.int32)
    lanes = int(depth.max()) + 1 if len(view) else 1
    if nesting == "lanes":
        height = np.full(len(view), _ROW_HEIGHT / lanes)
        y0 = y_row - _ROW_HEIGHT / 2 + depth * (_ROW_HEIGHT / lanes)
    else:
        height = np.full(len(view), _ROW_HEIGHT)
        y0 = y_row - _ROW_HEIGHT / 2

    if show_idle:
        _draw_idle_rows(ax, view, order, row_of, factor, theme)

    keys = np.asarray(colormap.interval_keys(view), dtype=object)
    labels_drawn: set[str] = set()
    for key in _unique_in_order(keys):
        style = colormap.style(key)
        label = colormap.label(key)
        labels_drawn.add(key)
        selection = keys == key
        # One collection per depth so that, in overlay mode, an inner region is
        # painted over the parent it interrupts.
        for d in np.unique(depth[selection]):
            rows = selection & (depth == d)
            ax.add_collection(
                PolyCollection(
                    _rectangles(x0[rows], width[rows], y0[rows], height[rows]),
                    facecolors=style["facecolor"],
                    edgecolors=theme.surface,
                    linewidths=0.7,
                    hatch=style["hatch"],
                    zorder=3 + int(d),
                    label=label,
                )
            )

    span = (view.span[0] * factor, view.span[1] * factor)
    ax.set_xlim(span[0], span[1] if span[1] > span[0] else span[0] + 1.0)
    ax.set_ylim(n_rows - 0.5 + 0.5 * len(boundaries), -0.6)
    row_labels = view.location_labels()
    ax.set_yticks([row_of[i] for i in order])
    ax.set_yticklabels([truncate(row_labels[i], 28) for i in order], fontsize=9)
    ax.set_xlabel(f"time ({unit})")
    ax.set_title(title if title is not None else f"{view.name} — timeline")
    add_grid(ax, theme, axis="x")

    for boundary in boundaries:
        ax.axhline(boundary, color=theme.axis, linewidth=0.8, zorder=2)

    if markers and len(view.markers):
        _draw_markers(ax, view, row_of, factor, theme)

    if bar_labels:
        _draw_bar_labels(
            ax, view, colormap, keys, x0, width, y0, height, theme, fig, nesting
        )

    add_legend(
        ax,
        colormap.entries(labels_drawn),
        theme,
        where=legend,
        title=color_by,
    )
    if owned:
        fig.tight_layout()
    if save is not None:
        save_figure(fig, save)
    return ax


# ----------------------------------------------------------------------
def _row_positions(trace: Trace):
    """Row centre per location, plus the y of each process boundary."""
    order = trace.location_order()
    row_of: dict[int, float] = {}
    boundaries: list[float] = []
    position = 0.0
    previous_group = None
    for index in order:
        group = trace.locations[index].group_ref
        if previous_group is not None and group != previous_group:
            boundaries.append(position - 0.25)
            position += 0.5
        row_of[index] = position
        position += 1.0
        previous_group = group
    return order, row_of, boundaries


def _rectangles(x, width, y, height) -> np.ndarray:
    """``(n, 4, 2)`` vertex array for ``PolyCollection``."""
    x1 = x + width
    y1 = y + height
    return np.stack(
        [
            np.column_stack([x, y]),
            np.column_stack([x1, y]),
            np.column_stack([x1, y1]),
            np.column_stack([x, y1]),
        ],
        axis=1,
    )


def _unique_in_order(keys: np.ndarray) -> list:
    seen, out = set(), []
    for key in keys.tolist():
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _draw_idle_rows(ax, trace, order, row_of, factor, theme) -> None:
    """A hairline band per row, so an idle thread is still a visible row."""
    lo, hi = trace.span[0] * factor, trace.span[1] * factor
    ax.add_collection(
        PolyCollection(
            _rectangles(
                np.full(len(order), lo),
                np.full(len(order), max(hi - lo, 0.0)),
                np.array([row_of[i] - _ROW_HEIGHT / 2 for i in order]),
                np.full(len(order), _ROW_HEIGHT),
            ),
            facecolors=theme.grid,
            edgecolors="none",
            alpha=0.55,
            zorder=1,
        )
    )


def _draw_markers(ax, trace, row_of, factor, theme) -> None:
    """Thread fork/join carets, drawn above the bars."""
    styles = {"ThreadFork": "v", "ThreadJoin": "^"}
    for kind, marker in styles.items():
        selection = trace.markers.kind == kind
        if not selection.any():
            continue
        ax.scatter(
            trace.markers.time[selection] * factor,
            [
                row_of[i] - _ROW_HEIGHT / 2 - 0.12
                for i in trace.markers.location[selection]
            ],
            marker=marker,
            s=22,
            color=theme.primary,
            edgecolors=theme.surface,
            linewidths=0.6,
            zorder=20,
            label=None,
        )


def _draw_bar_labels(
    ax, trace, colormap, keys, x0, width, y0, height, theme, fig, nesting
) -> None:
    """Name the regions that own enough room on screen to say so.

    A label is placed in the widest part of a bar that no nested region paints
    over, then measured against that gap and dropped if it does not fit. It is
    cheaper to leave a bar unlabelled than to let two names collide.
    """
    if not len(trace):
        return
    renderer = _renderer(fig)
    if renderer is None:
        return
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    extent = ax.get_window_extent()
    if extent.width <= 0 or x_hi <= x_lo or y_hi == y_lo:
        return
    px_per_x = extent.width / (x_hi - x_lo)
    px_per_y = extent.height / abs(y_hi - y_lo)

    # Rank by on-screen width; past a few hundred labels the text costs more
    # than it explains.
    visible = np.minimum(x0 + width, x_hi) - np.maximum(x0, x_lo)
    candidates = np.flatnonzero(visible * px_per_x >= _MIN_LABEL_PX)
    if candidates.size == 0:
        return
    candidates = candidates[np.argsort(-visible[candidates])][:_MAX_LABELS]

    for row in candidates:
        if nesting == "overlay":
            start, span = _widest_gap(trace, x0, width, int(row))
        else:
            start, span = x0[row], width[row]
        start, end = max(start, x_lo), min(start + span, x_hi)
        span_px = (end - start) * px_per_x
        lane_px = height[row] * px_per_y
        if span_px < _MIN_LABEL_PX or lane_px < _MIN_LANE_PX:
            continue
        size = min(8.0, max(6.0, lane_px * 0.5))
        chars = int((span_px - _LABEL_PADDING_PX) / (0.6 * size * fig.dpi / 72.0))
        if chars < 3:
            continue
        key = str(keys[row])
        label = ax.text(
            (start + end) / 2,
            y0[row] + height[row] / 2,
            truncate(key, chars),
            ha="center",
            va="center",
            fontsize=size,
            color=ink_on(colormap.style(key)["facecolor"], theme),
            zorder=15,
            clip_on=True,
        )
        if label.get_window_extent(renderer).width > span_px - _LABEL_PADDING_PX:
            label.remove()


def _widest_gap(trace, x0, width, row: int) -> tuple[float, float]:
    """The longest stretch of a bar that its own nested regions leave visible."""
    start, end = x0[row], x0[row] + width[row]
    inner = trace.rows_for_location(int(trace.location[row]))
    children = np.flatnonzero(
        (trace.depth[inner] == trace.depth[row] + 1)
        & (x0[inner] >= start)
        & (x0[inner] + width[inner] <= end)
    )
    if children.size == 0:
        return start, end - start
    children += inner.start
    order = np.argsort(x0[children], kind="stable")
    # Gaps run from the end of one child (or the bar start) to the next start.
    gap_starts = np.concatenate([[start], (x0[children] + width[children])[order]])
    gap_ends = np.concatenate([x0[children][order], [end]])
    gaps = gap_ends - gap_starts
    best = int(np.argmax(gaps))
    return float(gap_starts[best]), float(max(gaps[best], 0.0))


def _renderer(fig):
    """The figure's renderer, or ``None`` on a backend that cannot supply one."""
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        pass
    try:  # pragma: no cover - backend dependent
        return fig._get_renderer()
    except AttributeError:
        return None
