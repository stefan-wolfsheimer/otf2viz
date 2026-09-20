"""otf2viz — Matplotlib visualisation of OTF2 parallel traces.

Read a Score-P / OTF2 measurement into a filterable table of region intervals
and draw it: a timeline per thread, concurrency over time, and where the time
actually went. Every figure is a plain Matplotlib figure, so it renders inline
in Jupyter, saves to PNG/SVG/PDF, and composes into your own subplot grids.

    >>> import otf2viz                                    # doctest: +SKIP
    >>> trace = otf2viz.read_trace("scorep_run")
    >>> otf2viz.overview(trace)

Without a trace file at hand (or without the ``otf2`` bindings) you can build a
trace from plain tuples and use the same plots:

    >>> from otf2viz import Trace, timeline
    >>> trace = Trace.from_events([
    ...     ("thread 0", "compute", 0.0, 0.8),
    ...     ("thread 1", "compute", 0.1, 0.7),
    ...     ("thread 1", "barrier", 0.7, 0.8),
    ... ])
    >>> ax = timeline(trace)
"""

from . import plots, tables, theme
from .colors import ColorMap
from .interactive import interactive_timeline
from .model import Location, Markers, Region, Trace
from .plots import concurrency, location_breakdown, overview, region_summary, timeline
from .reader import find_traces, read_trace
from .tables import location_table, region_table, summary
from .theme import DARK, LIGHT, Theme, set_theme

__version__ = "0.1.0"

__all__ = [
    # reading
    "read_trace",
    "find_traces",
    # model
    "Trace",
    "Region",
    "Location",
    "Markers",
    # plots
    "timeline",
    "region_summary",
    "location_breakdown",
    "concurrency",
    "overview",
    "interactive_timeline",
    # tables
    "region_table",
    "location_table",
    "summary",
    # theming
    "ColorMap",
    "Theme",
    "LIGHT",
    "DARK",
    "set_theme",
    # submodules
    "plots",
    "tables",
    "theme",
    "__version__",
]
