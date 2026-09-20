"""The interval table: nesting, filtering, statistics."""

from __future__ import annotations

import re

import numpy as np
import pytest

from otf2viz import Trace
from otf2viz.model import _derive_nesting, _merge_intervals


def test_from_events_recovers_nesting(simple_trace):
    depth = {
        (name, start): d
        for name, start, d in zip(
            [simple_trace.regions[i].name for i in simple_trace.region],
            simple_trace.start,
            simple_trace.depth,
        )
    }
    assert depth[("main", 0.0)] == 0
    assert depth[("compute", 0.1)] == 1
    assert depth[("inner", 0.2)] == 2
    assert depth[("compute", 0.15)] == 0  # first region on thread 1


def test_exclusive_time_excludes_children(simple_trace):
    stats = {s.name: s for s in simple_trace.region_stats()}
    # main spans 1.0s but hosts compute (0.5s) and barrier (0.4s).
    assert stats["main"].inclusive == pytest.approx(1.0)
    assert stats["main"].exclusive == pytest.approx(0.1)
    # compute runs twice: 0.5s nesting inner (0.2s), and 0.4s on thread 1.
    assert stats["compute"].count == 2
    assert stats["compute"].inclusive == pytest.approx(0.9)
    assert stats["compute"].exclusive == pytest.approx(0.7)


def test_region_stats_sorted_by_exclusive_time(simple_trace):
    stats = simple_trace.region_stats()
    assert [s.name for s in stats][:2] == ["barrier", "compute"]
    assert stats == sorted(stats, key=lambda s: s.exclusive, reverse=True)


def test_filter_by_name_regex_and_predicate(simple_trace):
    assert simple_trace.filter(regions="compute").region_names() == ["compute"]
    assert set(simple_trace.filter(regions=["main", "inner"]).region_names()) == {
        "main",
        "inner",
    }
    assert simple_trace.filter(regions=re.compile("^in")).region_names() == ["inner"]
    predicate = simple_trace.filter(regions=lambda r: r.name.endswith("er"))
    assert set(predicate.region_names()) == {"barrier", "inner"}


def test_filter_prunes_regions_but_keeps_locations(simple_trace):
    filtered = simple_trace.filter(regions="main")
    assert filtered.regions == tuple(
        r for r in simple_trace.regions if r.name == "main"
    )
    # thread 1 never runs main, but it is still a row on the timeline.
    assert len(filtered.locations) == len(simple_trace.locations)
    assert filtered.region.max() < len(filtered.regions)


def test_filter_by_depth_and_duration(simple_trace):
    assert simple_trace.filter(max_depth=0).depth.max() == 0
    assert len(simple_trace.filter(max_depth=0)) == 3
    long_only = simple_trace.filter(min_duration=0.45)
    assert (long_only.durations >= 0.45).all()


def test_time_range_keeps_overlapping_intervals(simple_trace):
    window = simple_trace.select_time(0.5, 0.7)
    # main, compute and barrier on thread 0; compute and barrier on thread 1.
    # "inner" (0.2-0.4) ends before the window opens.
    assert len(window) == 5
    assert "inner" not in window.region_names()
    assert window.span == (0.5, 0.7)
    # Without clipping the original bounds survive, so durations stay true.
    assert window.start.min() < 0.5

    clipped = simple_trace.select_time(0.5, 0.7, clip=True)
    assert clipped.start.min() >= 0.5
    assert clipped.stop.max() <= 0.7


def test_filter_locations_reindexes(multiprocess_trace):
    one_rank = multiprocess_trace.filter(locations=[0, 1])
    assert len(one_rank.locations) == 2
    assert one_rank.n_processes == 1
    assert one_rank.location.max() < 2

    by_name = multiprocess_trace.filter(locations=["P1 OMP thread 1"])
    assert len(by_name.locations) == 1
    with pytest.raises(KeyError):
        multiprocess_trace.filter(locations=["no such thread"])


def test_location_labels_disambiguate_only_when_needed(
    simple_trace, multiprocess_trace
):
    assert simple_trace.location_labels() == ["thread 0", "thread 1"]
    assert multiprocess_trace.location_labels() == [
        "P0 Master thread",
        "P0 OMP thread 1",
        "P1 Master thread",
        "P1 OMP thread 1",
    ]


def test_location_stats_merge_nested_intervals(simple_trace):
    stats = {s.label: s for s in simple_trace.location_stats()}
    # thread 0 is busy for the whole second even though its regions overlap.
    assert stats["thread 0"].busy == pytest.approx(1.0)
    assert stats["thread 0"].idle == pytest.approx(0.0)
    assert stats["thread 1"].busy == pytest.approx(0.85)
    assert stats["thread 1"].utilisation == pytest.approx(0.85)


def test_concurrency_counts_each_location_once(simple_trace):
    times, counts = simple_trace.concurrency()
    assert counts.max() == 2
    assert times[0] == simple_trace.span[0]
    assert times[-1] == simple_trace.span[1]
    # Between 0.0 and 0.15 only thread 0 has started.
    assert counts[np.searchsorted(times, 0.05, side="right") - 1] == 1


def test_merge_intervals_handles_nesting_and_gaps():
    start = np.array([0.0, 0.1, 0.5, 2.0])
    stop = np.array([1.0, 0.4, 0.9, 3.0])
    merged_start, merged_stop = _merge_intervals(start, stop)
    assert merged_start.tolist() == [0.0, 2.0]
    assert merged_stop.tolist() == [1.0, 3.0]


def test_merge_intervals_on_empty_input():
    merged_start, merged_stop = _merge_intervals(
        np.empty(0), np.empty(0)
    )
    assert merged_start.size == 0 and merged_stop.size == 0


def test_derive_nesting_is_per_location():
    start = np.array([0.0, 0.1, 0.0])
    stop = np.array([1.0, 0.5, 1.0])
    location = np.array([0, 0, 1])
    depth, child = _derive_nesting(start, stop, location)
    assert depth.tolist() == [0, 1, 0]
    assert child.tolist() == [0.4, 0.0, 0.0]


def test_empty_trace_is_usable():
    trace = Trace.from_events([], name="empty")
    assert len(trace) == 0
    assert trace.duration == 0.0
    assert trace.region_stats() == []
    assert trace.concurrency()[1].tolist() == [0, 0]


def test_rejects_inconsistent_arrays():
    with pytest.raises(ValueError, match="different length"):
        Trace([], [], [0.0], [0.0, 1.0], [0], [0])
    with pytest.raises(ValueError, match="location index"):
        Trace([], [], [0.0], [1.0], [0], [0])


def test_to_dict_columns_line_up(simple_trace):
    columns = simple_trace.to_dict()
    assert {len(v) for v in columns.values()} == {len(simple_trace)}
    assert set(columns) >= {"location", "region", "start", "stop", "exclusive"}


def test_repr_and_html_do_not_raise(simple_trace):
    assert "simple" in repr(simple_trace)
    assert "<table" in simple_trace._repr_html_()
