"""Table views, and the optional ipywidgets explorer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import otf2viz
from otf2viz.tables import Table


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_region_table_matches_the_stats(simple_trace):
    table = otf2viz.region_table(simple_trace)
    assert len(table) == len(simple_trace.regions)
    assert table.columns[0] == "region"
    assert [row[0] for row in table] == simple_trace.region_names()
    assert "<table" in table._repr_html_()
    assert "barrier" in repr(table)


def test_region_table_can_be_capped_and_resorted(simple_trace):
    assert len(otf2viz.region_table(simple_trace, top=2)) == 2
    by_count = otf2viz.region_table(simple_trace, sort_by="count")
    assert by_count.rows[0][2] >= by_count.rows[-1][2]


def test_location_table_reports_utilisation(simple_trace):
    table = otf2viz.location_table(simple_trace)
    assert [row[0] for row in table] == ["thread 0", "thread 1"]
    assert table.rows[0][-1] == "100.0%"


def test_summary_table_covers_the_measurement(simple_trace):
    facts = dict(otf2viz.summary(simple_trace).rows)
    assert facts["locations"] == "2"
    assert facts["intervals"] == "6"
    assert facts["max call depth"] == "2"
    assert facts["hottest region"] == "barrier"


def test_table_columns_round_trip(simple_trace):
    table = otf2viz.location_table(simple_trace)
    columns = table.to_dict()
    assert list(columns) == list(table.columns)
    assert len(columns["location"]) == len(table)


def test_empty_table_renders():
    table = Table(columns=("a", "b"), rows=[], title="nothing")
    assert "<table" in table._repr_html_()
    assert "nothing" in repr(table)


def test_interactive_timeline_builds_and_responds(simple_trace):
    pytest.importorskip("ipywidgets")
    controls = otf2viz.interactive_timeline(simple_trace)
    window, depth, regions = controls.children
    assert window.min == simple_trace.span[0]
    assert window.max == simple_trace.span[1]
    assert depth.max == simple_trace.max_depth
    assert regions.options[0] == "(all)"

    # Each of these fires a redraw; none of them may raise.
    window.value = (0.2, 0.5)
    depth.value = 0
    regions.value = ("compute",)
    window.value = (0.0, 1e-9)  # a window with nothing in it
