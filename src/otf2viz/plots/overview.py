"""A single figure that answers the three first questions about a run."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from ..colors import ColorMap, resolve
from ..model import Trace
from ..theme import get_theme, use
from ._common import save_figure
from .summary import concurrency, location_breakdown, region_summary
from .timeline import timeline

__all__ = ["overview"]


def overview(
    trace: Trace,
    *,
    colors: ColorMap | None = None,
    color_by: str = "region",
    max_colors: int = 8,
    theme=None,
    figsize: tuple[float, float] | None = None,
    time_unit: str | None = None,
    nesting: str = "overlay",
    title: str | None = None,
    save: str | os.PathLike | None = None,
    **timeline_kwargs,
):
    """Timeline, concurrency and per-region totals in one themed figure.

    The timeline and the concurrency panel share an x axis, so a dip in
    concurrency lines up with the bars that caused it. All panels share one
    :class:`~otf2viz.ColorMap`, so a colour means the same thing throughout.

    ``save`` also writes the figure to that path, in the format named by the
    file extension (``.png``, ``.svg``, ``.pdf``, ...).

    Returns the :class:`~matplotlib.figure.Figure`; the four axes are on
    ``fig.axes`` in the order timeline, concurrency, region summary,
    location breakdown.
    """
    theme = get_theme(theme)
    colormap = resolve(colors, trace, by=color_by, max_colors=max_colors, theme=theme)

    rows = max(len(trace.locations), 1)
    if figsize is None:
        figsize = (13.0, 0.34 * rows + 8.0)

    with use(theme):
        # Constrained layout, not tight_layout: it is the one that reserves
        # room for the timeline legend hanging off the right-hand edge, so the
        # figure survives a plain savefig() without bbox_inches="tight".
        fig = plt.figure(figsize=figsize, layout="constrained")
        fig.get_layout_engine().set(hspace=0.06, wspace=0.06)
        grid = fig.add_gridspec(3, 2, height_ratios=[0.34 * rows + 1.4, 1.9, 3.4])
        ax_timeline = fig.add_subplot(grid[0, :])
        ax_concurrency = fig.add_subplot(grid[1, :], sharex=ax_timeline)
        ax_regions = fig.add_subplot(grid[2, 0])
        ax_locations = fig.add_subplot(grid[2, 1])

    unit = time_unit or _shared_unit(trace)
    timeline(
        trace,
        ax=ax_timeline,
        colors=colormap,
        color_by=color_by,
        nesting=nesting,
        theme=theme,
        time_unit=unit,
        title="timeline",
        **timeline_kwargs,
    )
    concurrency(
        trace, ax=ax_concurrency, theme=theme, time_unit=unit, title="concurrency"
    )
    region_summary(
        trace, ax=ax_regions, theme=theme, top=10, title="exclusive time per region"
    )
    location_breakdown(
        trace,
        ax=ax_locations,
        colors=colormap,
        color_by=color_by,
        theme=theme,
        legend=False,
        title="time per location",
    )
    if title is not None:
        fig.suptitle(title, color=theme.primary, fontsize=13, x=0.01, ha="left")
    elif trace.name:
        fig.suptitle(
            f"{trace.name} — {len(trace):,} intervals on {len(trace.locations)} "
            "locations",
            color=theme.primary,
            fontsize=13,
            x=0.01,
            ha="left",
        )
    if save is not None:
        save_figure(fig, save)
    return fig


def _shared_unit(trace: Trace) -> str:
    """One unit for both time axes, so the panels stay comparable."""
    from ._common import time_scale

    return time_scale(trace.duration)[1]
