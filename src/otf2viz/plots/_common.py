"""Shared axis, legend and unit helpers for the plotting functions."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from ..theme import Theme, use

__all__ = [
    "time_scale",
    "new_axes",
    "style_axes",
    "add_grid",
    "add_legend",
    "truncate",
    "ink_on",
    "format_duration",
    "save_figure",
]

_UNITS = {"s": 1.0, "ms": 1e3, "us": 1e6, "µs": 1e6, "ns": 1e9}


def time_scale(duration: float, unit: str | None = None) -> tuple[float, str]:
    """Multiplier and axis label for the unit a duration reads best in."""
    if unit is not None:
        key = unit.lower()
        if key not in _UNITS:
            raise ValueError(f"unknown time unit {unit!r}; expected one of s, ms, µs, ns")
        return _UNITS[key], "µs" if key == "us" else key
    if duration <= 0:
        return 1.0, "s"
    if duration < 1e-3:
        return 1e6, "µs"
    if duration < 1.0:
        return 1e3, "ms"
    return 1.0, "s"


def new_axes(ax, figsize, theme: Theme):
    """Return ``(fig, ax, owned)``, creating a themed figure when ``ax`` is ``None``.

    ``owned`` says whether we made the figure. When we did not, the caller
    controls the layout — so a plot dropped into someone else's grid must not
    call ``tight_layout`` behind their back.
    """
    if ax is not None:
        return ax.get_figure(), ax, False
    with use(theme):
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, True


def save_figure(fig, path: str | os.PathLike) -> Path:
    """Write ``fig`` to ``path``, taking the file type from the extension.

    Any format Matplotlib can write is accepted — ``.png``, ``.svg``, ``.pdf``,
    ``.eps``, ... — and an extension it does not know is an error here rather
    than a silent PNG named ``plot.jpg``.

    ``bbox_inches="tight"`` is not optional: :func:`~otf2viz.timeline` hangs its
    legend off the right-hand edge of the axes, and a plain ``savefig`` crops it
    away. The saved file is what the figure looks like on screen.
    """
    path = Path(path).expanduser()
    suffix = path.suffix.lower().lstrip(".")
    supported = fig.canvas.get_supported_filetypes()
    if suffix not in supported:
        raise ValueError(
            f"cannot tell the file type of {path.name!r}: save= takes it from the "
            f"extension, one of {', '.join('.' + s for s in sorted(supported))}"
        )
    fig.savefig(path, bbox_inches="tight")
    return path


def style_axes(ax, theme: Theme) -> None:
    """Apply the theme explicitly, so a caller-supplied axis matches too."""
    ax.set_facecolor(theme.surface)
    ax.get_figure().set_facecolor(theme.page)
    for side, visible in (
        ("top", False),
        ("right", False),
        ("bottom", True),
        ("left", True),
    ):
        spine = ax.spines[side]
        spine.set_visible(visible)
        spine.set_color(theme.axis)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=theme.muted, labelcolor=theme.secondary, labelsize=9)
    ax.xaxis.label.set_color(theme.secondary)
    ax.yaxis.label.set_color(theme.secondary)
    ax.title.set_color(theme.primary)
    ax.set_axisbelow(True)


def add_grid(ax, theme: Theme, axis: str = "x") -> None:
    """Recessive hairline grid on one axis only."""
    ax.grid(True, axis=axis, color=theme.grid, linewidth=0.8, zorder=0)
    ax.grid(False, axis="y" if axis == "x" else "x")


def add_legend(
    ax,
    entries,
    theme: Theme,
    *,
    where: str | bool = "right",
    title: str | None = None,
    max_label: int = 38,
):
    """Attach a legend built from ``(label, style)`` pairs.

    ``where`` is ``"right"`` (best for the long region names OTF2 produces),
    ``"bottom"``, or ``False``. A legend is drawn for two or more series; a
    single series is named by the title instead.
    """
    if not where or len(entries) < 2:
        return None
    handles = [
        Patch(
            facecolor=style["facecolor"],
            hatch=style["hatch"],
            edgecolor=theme.surface,
            linewidth=0.8,
            label=truncate(label, max_label),
        )
        for label, style in entries
    ]
    if where == "bottom":
        legend = ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.16),
            ncols=max(1, min(4, len(handles))),
            frameon=False,
            fontsize=9,
            title=title,
            alignment="left",
        )
    else:
        legend = ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.015, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
            title=title,
            alignment="left",
        )
    for text in legend.get_texts():
        text.set_color(theme.secondary)
    if legend.get_title() is not None:
        legend.get_title().set_color(theme.muted)
        legend.get_title().set_fontsize(9)
    return legend


def truncate(text: str, limit: int) -> str:
    """Shorten a label from the left, keeping the informative tail."""
    text = str(text)
    if len(text) <= limit:
        return text
    return "…" + text[-(limit - 1) :]


def ink_on(color: str, theme: Theme) -> str:
    """Readable text colour for a label drawn on top of ``color``."""
    r, g, b = to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if luminance > 0.55 else "#ffffff"


def format_duration(seconds: float) -> str:
    """Re-exported from the model so plot code has one place to import from."""
    from ..model import _format_duration

    return _format_duration(seconds)
