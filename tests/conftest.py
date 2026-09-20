"""Shared fixtures.

The plotting tests run on synthetic traces so that the whole suite works
without the ``otf2`` bindings or a measurement on disk. The reader tests are
skipped unless both are available.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from otf2viz import Trace  # noqa: E402
from otf2viz.model import Location, Region  # noqa: E402

#: Point this at a Score-P experiment directory to exercise the OTF2 reader.
TRACE_ENV = "OTF2VIZ_TEST_TRACE"


@pytest.fixture
def simple_trace() -> Trace:
    """Two threads, nested regions, one idle stretch."""
    return Trace.from_events(
        [
            ("thread 0", "main", 0.0, 1.0),
            ("thread 0", "compute", 0.1, 0.6),
            ("thread 0", "inner", 0.2, 0.4),
            ("thread 0", "barrier", 0.6, 1.0),
            ("thread 1", "compute", 0.15, 0.55),
            ("thread 1", "barrier", 0.55, 1.0),
        ],
        name="simple",
    )


@pytest.fixture
def multiprocess_trace() -> Trace:
    """Two ranks with two threads each, for row grouping and labels."""
    locations = [
        Location(name="Master thread", group_ref=0, thread=0),
        Location(name="OMP thread 1", group_ref=0, thread=1),
        Location(name="Master thread", group_ref=1, thread=0),
        Location(name="OMP thread 1", group_ref=1, thread=1),
    ]
    regions = [
        Region(name="MPI_Allreduce", paradigm="MPI", role="COLLECTIVE"),
        Region(name="compute", paradigm="OPENMP", role="LOOP"),
    ]
    starts = [0.0, 0.1, 0.0, 0.2]
    stops = [0.5, 0.6, 0.4, 0.7]
    return Trace(
        locations,
        regions,
        starts,
        stops,
        [0, 1, 2, 3],
        [1, 1, 1, 0],
        span=(0.0, 1.0),
        name="ranks",
    )


@pytest.fixture(scope="session")
def otf2_trace_path() -> Path:
    """A real OTF2 measurement, or skip."""
    pytest.importorskip("otf2", reason="the otf2 bindings are not installed")
    path = os.environ.get(TRACE_ENV)
    if not path:
        pytest.skip(f"set {TRACE_ENV} to a Score-P experiment directory")
    resolved = Path(path)
    if not resolved.exists():
        pytest.skip(f"{TRACE_ENV}={path} does not exist")
    return resolved
