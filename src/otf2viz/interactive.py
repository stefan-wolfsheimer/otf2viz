"""Optional ipywidgets controls for exploring a trace inside Jupyter.

Everything here is a convenience wrapper over :func:`otf2viz.timeline` and
:meth:`otf2viz.Trace.filter` — the static plots stay usable without ipywidgets,
and without a notebook at all.
"""

from __future__ import annotations

from .colors import ColorMap
from .model import Trace
from .plots import timeline
from .theme import get_theme

__all__ = ["interactive_timeline"]

_HINT = (
    "otf2viz.interactive_timeline() needs ipywidgets: "
    "pip install 'otf2viz[jupyter]'"
)


def interactive_timeline(
    trace: Trace,
    *,
    color_by: str = "region",
    max_colors: int = 8,
    theme=None,
    nesting: str = "overlay",
    figsize: tuple[float, float] | None = None,
    **timeline_kwargs,
):
    """A timeline with a time-window slider, depth limit and region filter.

    Controls sit in one row above the plot: drag the window to zoom, cap the
    call depth to strip nested detail, and pick regions to isolate a phase.
    Colours are assigned once from the full trace, so filtering never repaints
    the series that survive.

    >>> import otf2viz                                    # doctest: +SKIP
    >>> otf2viz.interactive_timeline(otf2viz.read_trace("scorep_run"))
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_HINT) from exc

    import matplotlib.pyplot as plt

    resolved_theme = get_theme(theme)
    colormap = ColorMap.from_trace(
        trace, by=color_by, max_colors=max_colors, theme=resolved_theme
    )
    lo, hi = trace.span
    step = (hi - lo) / 500 if hi > lo else 1.0

    window = widgets.FloatRangeSlider(
        value=(lo, hi),
        min=lo,
        max=hi if hi > lo else lo + 1.0,
        step=step,
        description="window",
        readout_format=".4g",
        continuous_update=False,
        layout=widgets.Layout(width="46%"),
    )
    depth = widgets.IntSlider(
        value=trace.max_depth,
        min=0,
        max=max(trace.max_depth, 1),
        description="max depth",
        continuous_update=False,
        layout=widgets.Layout(width="22%"),
    )
    regions = widgets.SelectMultiple(
        options=["(all)"] + list(colormap.keys),
        value=("(all)",),
        rows=4,
        description=color_by,
        layout=widgets.Layout(width="30%"),
    )
    controls = widgets.HBox([window, depth, regions])
    output = widgets.Output()

    def render(*_):
        with output:
            output.clear_output(wait=True)
            selected = [key for key in regions.value if key != "(all)"]
            view = trace.filter(
                time_range=window.value,
                max_depth=depth.value,
                regions=(
                    None
                    if not selected
                    else (lambda region: colormap.key_of(region) in set(selected))
                ),
            )
            if not len(view):
                print("no intervals match the current filters")
                return
            ax = timeline(
                view,
                colors=colormap,
                color_by=color_by,
                nesting=nesting,
                theme=resolved_theme,
                figsize=figsize,
                max_intervals=None,
                **timeline_kwargs,
            )
            # Show it once and drop it: a redraw per slider tick would
            # otherwise leak a figure each time.
            figure = ax.get_figure()
            display(figure)
            plt.close(figure)

    for control in (window, depth, regions):
        control.observe(render, names="value")
    render()
    display(widgets.VBox([controls, output]))
    return controls
