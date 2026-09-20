"""The OTF2 reader.

These need the ``otf2`` bindings and a measurement on disk; point
``OTF2VIZ_TEST_TRACE`` at a Score-P experiment directory to run them.
"""

from __future__ import annotations

import numpy as np
import pytest

import otf2viz
from otf2viz.reader import _enum_name, _resolve_anchor


def test_resolve_anchor_accepts_a_directory(otf2_trace_path):
    anchor = _resolve_anchor(otf2_trace_path)
    assert anchor.name == "traces.otf2"
    assert _resolve_anchor(anchor) == anchor


def test_resolve_anchor_reports_a_missing_trace(tmp_path):
    with pytest.raises(FileNotFoundError, match="traces.otf2"):
        _resolve_anchor(tmp_path)


def test_enum_name_strips_the_ctypes_enum_prefix():
    assert _enum_name("Paradigm.OPENMP") == "OPENMP"
    assert _enum_name(None) == "UNKNOWN"


def test_read_trace_produces_a_consistent_table(otf2_trace_path):
    trace = otf2viz.read_trace(otf2_trace_path)
    assert len(trace) > 0
    assert len(trace.locations) > 0
    assert (trace.stop >= trace.start).all()
    assert (trace.start >= 0).all()
    assert trace.duration > 0
    # Times are seconds relative to the start of the measurement.
    assert trace.start.min() == pytest.approx(0.0, abs=trace.duration)
    assert trace.timer_resolution > 0


def test_read_trace_recovers_nesting(otf2_trace_path):
    trace = otf2viz.read_trace(otf2_trace_path)
    assert trace.depth.min() == 0
    # A child never outlives its parent, so exclusive <= inclusive everywhere.
    exclusive = trace.durations - trace.child_time
    assert (exclusive >= -1e-12).all()


def test_read_trace_drops_measurement_artefacts(otf2_trace_path):
    trace = otf2viz.read_trace(otf2_trace_path)
    assert "TRACE BUFFER FLUSH" not in trace.region_names()
    kept = otf2viz.read_trace(otf2_trace_path, exclude_regions=None)
    assert len(kept) >= len(trace)


def test_parallel_read_matches_the_serial_read(otf2_trace_path):
    serial = otf2viz.read_trace(otf2_trace_path, workers=1)
    parallel = otf2viz.read_trace(otf2_trace_path, workers=2)
    assert len(serial) == len(parallel)
    assert np.allclose(np.sort(serial.start), np.sort(parallel.start))
    assert sorted(serial.region_names()) == sorted(parallel.region_names())


def test_find_traces(otf2_trace_path):
    found = otf2viz.find_traces(otf2_trace_path)
    assert found and all(p.name == "traces.otf2" for p in found)


def test_plots_work_on_a_real_trace(otf2_trace_path):
    trace = otf2viz.read_trace(otf2_trace_path)
    figure = otf2viz.overview(trace)
    assert len(figure.axes) == 4
    assert len(otf2viz.region_table(trace)) == len(trace.regions)
    assert len(otf2viz.location_table(trace)) == len(trace.locations)
