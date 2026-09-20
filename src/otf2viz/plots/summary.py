"""Aggregate views: where the time went, and how evenly it was spread."""

from __future__ import annotations

import os

import numpy as np

from ..colors import ColorMap, resolve
from ..model import Trace
from ..theme import get_theme
from ._common import (
    add_grid,
    add_legend,
    format_duration,
    new_axes,
    save_figure,
    style_axes,
    time_scale,
    truncate,
)

__all__ = ["region_summary", "location_breakdown", "concurrency"]


def region_summary(
    trace: Trace,
    *,
    ax=None,
    top: int = 12,
    metric: str = "exclusive",
    theme=None,
    figsize: tuple[float, float] | None = None,
    time_unit: str | None = None,
    title: str | None = None,
    save: str | os.PathLike | None = None,
):
    """Horizontal bars of total time per region, longest first.

    One measure, one colour: the bar *length* is the comparison, so colouring
    the bars by region would encode nothing that the axis does not already say.

    ``metric`` is ``"exclusive"`` (time in the region itself — the default, and
    the one that answers "what is slow"), ``"inclusive"`` (including nested
    regions) or ``"count"``.

    ``save`` also writes the figure to that path, in the format named by the
    file extension (``.png``, ``.svg``, ``.pdf``, ...).
    """
    theme = get_theme(theme)
    if metric not in ("exclusive", "inclusive", "count"):
        raise ValueError(
            f"unknown metric {metric!r}; expected exclusive, inclusive or count"
        )

    stats = trace.region_stats(sort_by=metric)[:top]
    if not stats:
        raise ValueError("trace contains no regions to summarise")
    values = np.array([getattr(s, metric) for s in stats], dtype=np.float64)

    if figsize is None:
        figsize = (8.5, max(2.0, 0.36 * len(stats) + 1.2))
    fig, ax, owned = new_axes(ax, figsize, theme)
    style_axes(ax, theme)

    if metric == "count":
        factor, unit = 1.0, ""
        axis_label = "calls"
    else:
        factor, unit = time_scale(float(values.max()), time_unit)
        axis_label = f"{metric} time ({unit})"

    positions = np.arange(len(stats))
    ax.barh(
        positions,
        values * factor,
        height=0.62,
        color=theme.accent,
        edgecolor=theme.surface,
        linewidth=0.8,
        zorder=3,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([truncate(s.name, 42) for s in stats], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(axis_label)
    ax.set_title(
        title if title is not None else f"{trace.name} — {metric} time per region"
    )
    add_grid(ax, theme, axis="x")

    # Direct value labels: the reason this chart needs no legend and no
    # gridline counting.
    headroom = float(values.max()) * factor
    ax.set_xlim(0, headroom * 1.16 if headroom > 0 else 1.0)
    for position, stat, value in zip(positions, stats, values):
        text = f"{stat.count:,}" if metric == "count" else format_duration(value)
        ax.text(
            value * factor + headroom * 0.015,
            position,
            text,
            va="center",
            ha="left",
            fontsize=8.5,
            color=theme.secondary,
            zorder=4,
        )
    if owned:
        fig.tight_layout()
    if save is not None:
        save_figure(fig, save)
    return ax


def location_breakdown(
    trace: Trace,
    *,
    ax=None,
    colors: ColorMap | None = None,
    color_by: str = "region",
    max_colors: int = 8,
    theme=None,
    figsize: tuple[float, float] | None = None,
    time_unit: str | None = None,
    legend: str | bool = "right",
    show_idle: bool = True,
    title: str | None = None,
    save: str | os.PathLike | None = None,
):
    """Stacked bars of exclusive time per location — the load-balance view.

    Ragged bar ends mean an imbalance; a wide ``idle`` segment on one thread
    means it spent the run waiting. Colours match :func:`~otf2viz.timeline`
    when both are given the same :class:`~otf2viz.ColorMap`.

    ``save`` also writes the figure to that path, in the format named by the
    file extension (``.png``, ``.svg``, ``.pdf``, ...).
    """
    theme = get_theme(theme)
    colormap = resolve(colors, trace, by=color_by, max_colors=max_colors, theme=theme)

    order = trace.location_order()
    labels = trace.location_labels()
    # Exclusive time per (row, colour slot), in one pass: everything folded
    # lands in a trailing "other" bucket, so one bincount cross-tabulates the
    # whole chart. Columns are indexed by *slot*, not by rank, so a filtered
    # trace keeps each series in its own column.
    n_series = colormap.max_colors + 1
    slot_of_region = np.array(
        [colormap.slot(colormap.key_of(region)) for region in trace.regions],
        dtype=np.int64,
    )
    slot_of_interval = np.where(
        slot_of_region[trace.region] < 0, n_series - 1, slot_of_region[trace.region]
    )
    row_of_location = np.empty(len(trace.locations), dtype=np.int64)
    row_of_location[order] = np.arange(len(order))
    flat = row_of_location[trace.location] * n_series + slot_of_interval
    totals = np.bincount(
        flat, weights=trace.exclusive, minlength=len(order) * n_series
    ).reshape(len(order), n_series)

    series: list[tuple[str, dict, np.ndarray]] = []
    for key in colormap.keys[: colormap.max_colors]:
        column = totals[:, colormap.slot(key)]
        if column.any():
            series.append((key, colormap.style(key), column))
    if totals[:, -1].any():
        series.append((colormap.other_label, colormap.other_style(), totals[:, -1]))
    idle = None
    if show_idle:
        # location_stats() is emitted in location_order(), i.e. the row order.
        busy = np.array([s.busy for s in trace.location_stats()])
        idle = np.maximum(trace.duration - busy, 0.0)
        if idle.any():
            series.append(("idle", {"facecolor": theme.grid, "hatch": None}, idle))
        else:
            idle = None

    if not series:
        raise ValueError("trace contains no time to break down")

    if figsize is None:
        figsize = (9.0, max(2.0, 0.4 * len(order) + 1.2))
    fig, ax, owned = new_axes(ax, figsize, theme)
    style_axes(ax, theme)

    stacked = np.sum([values for _, _, values in series], axis=0)
    factor, unit = time_scale(float(stacked.max()), time_unit)
    positions = np.arange(len(order))
    left = np.zeros(len(order))
    for label, style, values in series:
        ax.barh(
            positions,
            values * factor,
            left=left * factor,
            height=0.66,
            facecolor=style["facecolor"],
            hatch=style["hatch"],
            edgecolor=theme.surface,
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        left = left + values

    ax.set_yticks(positions)
    ax.set_yticklabels([truncate(labels[i], 28) for i in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"exclusive time ({unit})")
    ax.set_title(
        title if title is not None else f"{trace.name} — time per location"
    )
    add_grid(ax, theme, axis="x")

    if idle is not None:
        # Name the idle band once, in place, instead of spending a legend slot
        # on the one segment that means "nothing happened here".
        widest = int(np.argmax(idle))
        ax.text(
            (left[widest] - idle[widest] / 2) * factor,
            positions[widest],
            "idle",
            ha="center",
            va="center",
            fontsize=8,
            color=theme.muted,
            zorder=4,
        )
    add_legend(
        ax,
        [(label, style) for label, style, _ in series if label != "idle"],
        theme,
        where=legend,
        title=color_by,
    )
    if owned:
        fig.tight_layout()
    if save is not None:
        save_figure(fig, save)
    return ax


def concurrency(
    trace: Trace,
    *,
    ax=None,
    theme=None,
    figsize: tuple[float, float] | None = None,
    time_unit: str | None = None,
    show_peak: bool = True,
    title: str | None = None,
    save: str | os.PathLike | None = None,
):
    """How many locations were busy at once, over time.

    The flat stretches below the dashed line are the parallel efficiency you
    are actually losing: a serial section, a barrier, a straggler.

    ``save`` also writes the figure to that path, in the format named by the
    file extension (``.png``, ``.svg``, ``.pdf``, ...).
    """
    theme = get_theme(theme)
    times, counts = trace.concurrency()

    if figsize is None:
        figsize = (11.0, 2.4)
    fig, ax, owned = new_axes(ax, figsize, theme)
    style_axes(ax, theme)

    factor, unit = time_scale(trace.duration, time_unit)
    x = times * factor
    ax.fill_between(
        x, counts, step="post", color=theme.accent, alpha=0.18, linewidth=0, zorder=2
    )
    ax.step(x, counts, where="post", color=theme.accent, linewidth=2.0, zorder=3)

    peak = len(trace.locations)
    if show_peak and peak:
        ax.axhline(
            peak, color=theme.muted, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4
        )
        ax.annotate(
            f"all {peak} locations busy",
            xy=(0.997, peak),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            va="bottom",
            ha="right",
            fontsize=8.5,
            color=theme.muted,
        )

    span = (trace.span[0] * factor, trace.span[1] * factor)
    ax.set_xlim(span[0], span[1] if span[1] > span[0] else span[0] + 1.0)
    ax.set_ylim(0, max(peak, int(counts.max()) if counts.size else 0, 1) * 1.18)
    ax.set_xlabel(f"time ({unit})")
    ax.set_ylabel("busy locations")
    ax.set_title(title if title is not None else f"{trace.name} — concurrency")
    add_grid(ax, theme, axis="y")
    if owned:
        fig.tight_layout()
    if save is not None:
        save_figure(fig, save)
    return ax
