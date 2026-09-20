"""Regenerate ``quickstart.ipynb``.

The notebook is kept as source here rather than as hand-edited JSON, so it
stays diffable and cannot drift into invalid nbformat. Run this after changing
the tutorial:

    python examples/build_quickstart.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# otf2viz quickstart

Reading an OTF2 (Score-P) trace and looking at it, in a handful of cells.

This notebook needs the `otf2` Python bindings for the first section only —
they ship with Score-P, not with pip. If you do not have them, skip to
**Without an OTF2 file** at the bottom; everything after that runs anywhere.""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt

import otf2viz

otf2viz.__version__""",
    ),
    (
        "markdown",
        """## 1. Read a trace

`read_trace` takes the `traces.otf2` anchor file or any directory containing
one — a Score-P experiment directory works directly. Point `TRACE` at yours.""",
    ),
    (
        "code",
        """TRACE = "../../parmix/openmp/02_pi/score-p"

trace = otf2viz.read_trace(TRACE)
trace""",
    ),
    (
        "markdown",
        """The `Trace` renders as a summary because it is the last expression in the
cell. For the whole measurement in one table:""",
    ),
    ("code", "otf2viz.summary(trace)"),
    (
        "markdown",
        """## 2. The overview

One figure, four panels, one shared palette and x axis: what each thread did,
how many were busy at once, which region cost the most, and whether the load
was balanced.""",
    ),
    ("code", "otf2viz.overview(trace);"),
    (
        "markdown",
        """## 3. The timeline on its own

`nesting="overlay"` (the default) paints nested regions over their parent.
`"lanes"` gives each call depth its own sub-lane, turning every thread's row
into a flame graph — useful when you care about the call structure.""",
    ),
    ("code", 'otf2viz.timeline(trace, nesting="lanes", markers=True);'),
    (
        "markdown",
        """## 4. Filtering is how you make a trace readable

`filter()` returns a new trace; the original is untouched. Regions that end up
unused are pruned from the legend, but locations are kept — a thread that goes
idle under a filter stays as an empty row, which is usually the point.""",
    ),
    (
        "code",
        """import re

# Only the OpenMP regions, without the implicit barriers, in the first 10 ms.
work = (
    trace
    .filter(paradigms="OPENMP")
    .filter(exclude_roles=["IMPLICIT_BARRIER"])
    .select_time(0.0, 0.010)
)
work""",
    ),
    ("code", "otf2viz.timeline(work);"),
    (
        "markdown",
        """## 5. Keeping colours stable across plots

Build one `ColorMap` from the full trace and pass it to every plot. Colour then
follows the region, not its rank, so a filter never repaints the series that
survive. Compare the two rows below — same colours, different windows.""",
    ),
    (
        "code",
        """colors = otf2viz.ColorMap.from_trace(trace)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 5))
otf2viz.timeline(trace, ax=top, colors=colors, legend=False, title="whole run")
otf2viz.timeline(
    trace.select_time(0.005, 0.012), ax=bottom, colors=colors, title="zoomed"
)
fig.tight_layout();""",
    ),
    (
        "markdown",
        """Grouping by paradigm or role is a good first look at a busy trace: it has few
enough categories to fit the palette without folding anything into `other`.""",
    ),
    ('code', 'otf2viz.timeline(trace, color_by="role");'),
    (
        "markdown",
        """## 6. The numbers behind the plots

Every chart has a table. These are the exact values, and they render as HTML in
Jupyter.""",
    ),
    ("code", "otf2viz.region_table(trace, top=8)"),
    ("code", "otf2viz.location_table(trace)"),
    (
        "code",
        """# ...or hand it to pandas, if you have it installed.
# trace.to_dataframe().head()""",
    ),
    (
        "markdown",
        """## 7. Interactive exploration

With `ipywidgets` installed (`pip install 'otf2viz[jupyter]'`), this gives you a
time-window slider, a call-depth cap and a region picker above the timeline.""",
    ),
    ("code", "# otf2viz.interactive_timeline(trace)"),
    (
        "markdown",
        """## 8. Dark theme

The dark theme is a selected palette — the same hues re-stepped for a dark
surface — not an inverted light one.""",
    ),
    ("code", 'otf2viz.overview(trace, theme="dark");'),
    (
        "markdown",
        """## Without an OTF2 file

The plots work on any start/stop data. `Trace.from_events` takes
`(location, region, start, stop)` tuples and recovers the call nesting itself,
so you can sketch a schedule by hand or feed in timings from another tool.""",
    ),
    (
        "code",
        """from otf2viz import Trace

sketch = Trace.from_events(
    [
        ((0, 0), "main", 0.0, 1.0),
        ((0, 0), "compute", 0.05, 0.55),
        ((0, 0), "MPI_Allreduce", 0.55, 0.95),
        ((0, 1), "compute", 0.10, 0.60),
        ((0, 1), "MPI_Allreduce", 0.60, 0.95),
        ((1, 0), "main", 0.0, 1.0),
        ((1, 0), "compute", 0.05, 0.80),
        ((1, 0), "MPI_Allreduce", 0.80, 0.95),
        ((1, 1), "compute", 0.10, 0.35),
        ((1, 1), "MPI_Allreduce", 0.35, 0.95),
    ],
    name="two ranks, two threads",
)
otf2viz.timeline(sketch);""",
    ),
    (
        "markdown",
        """Rank 1's first thread computes for 0.75 s while everyone else waits in the
allreduce — the load-balance view says the same thing in one glance.""",
    ),
    ("code", "otf2viz.location_breakdown(sketch);"),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.splitlines(keepends=True)
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines,
                }
            )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    target = Path(__file__).with_name("quickstart.ipynb")
    target.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {target}")
