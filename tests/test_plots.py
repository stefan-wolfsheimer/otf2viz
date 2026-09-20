"""The figures. Rendered on Agg, so these run headless."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import otf2viz
from otf2viz import ColorMap, Trace


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("nesting", ["overlay", "lanes", "flat"])
def test_timeline_draws_every_nesting_mode(simple_trace, nesting):
    ax = otf2viz.timeline(simple_trace, nesting=nesting)
    assert ax.collections  # the bars
    assert [t.get_text() for t in ax.get_yticklabels()] == ["thread 0", "thread 1"]
    assert ax.get_xlabel() == "time (s)"


def test_timeline_rejects_an_unknown_nesting_mode(simple_trace):
    with pytest.raises(ValueError, match="unknown nesting"):
        otf2viz.timeline(simple_trace, nesting="spiral")


def test_timeline_refuses_to_draw_an_unreadable_number_of_bars():
    trace = Trace.from_events(
        [("t0", "r", i * 1e-6, i * 1e-6 + 5e-7) for i in range(50)]
    )
    with pytest.raises(ValueError, match="too many to draw"):
        otf2viz.timeline(trace, max_intervals=10)
    assert otf2viz.timeline(trace, max_intervals=None) is not None


def test_timeline_labels_stay_inside_their_bars(simple_trace):
    ax = otf2viz.timeline(simple_trace, figsize=(12, 3))
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        t.get_window_extent(renderer)
        for t in ax.texts
        if t.get_text() and t not in (ax.title,)
    ]
    for first, second in zip(boxes, boxes[1:]):
        assert not first.overlaps(second), "bar labels must not collide"


def test_timeline_rows_are_grouped_by_process(multiprocess_trace):
    ax = otf2viz.timeline(multiprocess_trace)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [
        "P0 Master thread",
        "P0 OMP thread 1",
        "P1 Master thread",
        "P1 OMP thread 1",
    ]
    # A hairline separates the two ranks.
    assert any(line.get_linestyle() == "-" for line in ax.lines)


def test_a_shared_colormap_keeps_colours_stable(simple_trace):
    colormap = ColorMap.from_trace(simple_trace)
    full = otf2viz.timeline(simple_trace, colors=colormap)
    partial = otf2viz.timeline(
        simple_trace.filter(exclude_regions="barrier"), colors=colormap
    )

    def colour_of(ax, label):
        for collection in ax.collections:
            if collection.get_label() == label:
                return tuple(collection.get_facecolor()[0])
        raise AssertionError(f"{label!r} was not drawn")

    assert colour_of(full, "compute") == colour_of(partial, "compute")


def test_location_breakdown_keeps_slots_under_a_shared_colormap():
    # Ten regions, so the colour map folds; then keep two that are far apart in
    # the ranking. The surviving series must land in their own colour slots.
    trace = Trace.from_events(
        # Disjoint, and shorter as the index grows, so the ranking is region0
        # first down to region9 last.
        [(f"t{i % 2}", f"region{i}", i * 10.0, i * 10.0 + (10 - i)) for i in range(10)]
    )
    colors = ColorMap.from_trace(trace, max_colors=8)
    assert colors.keys[:2] == ("region0", "region1")
    kept = [colors.keys[0], colors.keys[5]]
    ax = otf2viz.location_breakdown(
        trace.filter(regions=kept), colors=colors, show_idle=False
    )
    drawn = {
        container.get_label(): container.patches[0].get_facecolor()
        for container in ax.containers
    }
    assert set(drawn) == set(kept)
    for key in kept:
        assert (
            plt.matplotlib.colors.to_hex(drawn[key]) == colors.style(key)["facecolor"]
        )


def test_region_summary_labels_every_bar(simple_trace):
    ax = otf2viz.region_summary(simple_trace, top=3)
    assert len(ax.patches) == 3
    values = [t.get_text() for t in ax.texts]
    assert len(values) == 3
    assert all("s" in v for v in values)  # a formatted duration, e.g. "400 ms"


def test_region_summary_metrics(simple_trace):
    assert otf2viz.region_summary(simple_trace, metric="count").get_xlabel() == "calls"
    assert "inclusive" in otf2viz.region_summary(
        simple_trace, metric="inclusive"
    ).get_xlabel()
    with pytest.raises(ValueError, match="unknown metric"):
        otf2viz.region_summary(simple_trace, metric="median")


def test_location_breakdown_stacks_to_the_full_window(simple_trace):
    ax = otf2viz.location_breakdown(simple_trace)
    per_row: dict[float, float] = {}
    for patch in ax.patches:
        per_row[patch.get_y()] = per_row.get(patch.get_y(), 0.0) + patch.get_width()
    # Every bar reaches the same total: busy time plus idle time.
    assert len(set(round(v, 6) for v in per_row.values())) == 1
    assert "idle" in [t.get_text() for t in ax.texts]


def test_concurrency_marks_the_peak(simple_trace):
    ax = otf2viz.concurrency(simple_trace)
    assert ax.get_ylabel() == "busy locations"
    assert "all 2 locations busy" in [t.get_text() for t in ax.texts]


def test_overview_shares_one_colormap_and_one_x_axis(simple_trace):
    fig = otf2viz.overview(simple_trace)
    timeline_ax, concurrency_ax = fig.axes[0], fig.axes[1]
    assert timeline_ax.get_xlim() == concurrency_ax.get_xlim()
    assert len(fig.axes) == 4


def test_plots_accept_a_caller_supplied_axis(simple_trace):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5))
    otf2viz.timeline(simple_trace, ax=top)
    otf2viz.concurrency(simple_trace, ax=bottom)
    assert top.collections and bottom.lines


#: Every plot, and whether it hands back an Axes or the whole Figure.
PLOTS = [
    otf2viz.timeline,
    otf2viz.region_summary,
    otf2viz.location_breakdown,
    otf2viz.concurrency,
    otf2viz.overview,
]

#: (extension, magic number) for the formats worth naming in the docs.
FILE_TYPES = [("png", b"\x89PNG"), ("svg", b"<?xm"), ("pdf", b"%PDF")]


@pytest.mark.parametrize("plot", PLOTS, ids=lambda p: p.__name__)
@pytest.mark.parametrize("extension, magic", FILE_TYPES, ids=[e for e, _ in FILE_TYPES])
def test_save_writes_the_format_named_by_the_extension(
    simple_trace, tmp_path, plot, extension, magic
):
    path = tmp_path / f"plot.{extension}"
    assert plot(simple_trace, save=path) is not None  # still returns the plot
    assert path.read_bytes()[: len(magic)] == magic


@pytest.mark.parametrize("plot", PLOTS, ids=lambda p: p.__name__)
def test_save_rejects_an_extension_it_cannot_map_to_a_format(
    simple_trace, tmp_path, plot
):
    with pytest.raises(ValueError, match="cannot tell the file type"):
        plot(simple_trace, save=tmp_path / "plot.bogus")
    with pytest.raises(ValueError, match="cannot tell the file type"):
        plot(simple_trace, save=tmp_path / "no_extension")
    assert list(tmp_path.iterdir()) == []


def test_save_keeps_a_legend_that_overflows_the_figure(tmp_path):
    """The timeline hangs its legend off the right edge of the axes.

    On a figure too narrow to hold it — a caller's grid, where we do not get to
    run tight_layout — a plain savefig() would crop the legend away. The saved
    file has to be what the figure actually looks like, so it grows instead.
    """
    trace = Trace.from_events(
        [
            (f"t{t}", f"region {i}", i * 0.1, i * 0.1 + 0.08)
            for t in range(2)
            for i in range(4)
        ]
    )
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    path = tmp_path / "wide.png"
    otf2viz.timeline(trace, ax=ax, save=path)
    assert ax.get_legend() is not None
    assert plt.matplotlib.image.imread(path).shape[1] > 4 * 100


def test_save_with_a_caller_supplied_axis_writes_the_whole_figure(
    simple_trace, tmp_path
):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5))
    otf2viz.timeline(simple_trace, ax=top)
    path = tmp_path / "grid.png"
    otf2viz.concurrency(simple_trace, ax=bottom, save=path)
    assert plt.matplotlib.image.imread(path).size > 0


def test_plots_render_in_both_themes(simple_trace):
    for theme in ("light", "dark"):
        ax = otf2viz.timeline(simple_trace, theme=theme)
        expected = otf2viz.LIGHT if theme == "light" else otf2viz.DARK
        assert ax.get_facecolor() == plt.matplotlib.colors.to_rgba(expected.surface)


def test_legend_appears_for_two_or_more_series(simple_trace):
    assert otf2viz.timeline(simple_trace).get_legend() is not None
    single = simple_trace.filter(regions="main")
    assert otf2viz.timeline(single).get_legend() is None
    assert otf2viz.timeline(simple_trace, legend=False).get_legend() is None


def test_time_unit_is_chosen_and_can_be_forced():
    microseconds = Trace.from_events([("t0", "r", 0.0, 5e-6)])
    assert "µs" in otf2viz.timeline(microseconds).get_xlabel()
    seconds = Trace.from_events([("t0", "r", 0.0, 12.0)])
    assert "(s)" in otf2viz.timeline(seconds).get_xlabel()
    assert "(ms)" in otf2viz.timeline(seconds, time_unit="ms").get_xlabel()


def test_empty_trace_draws_an_empty_frame_or_says_why():
    empty = Trace.from_events([], name="empty")
    assert otf2viz.timeline(empty) is not None
    assert otf2viz.concurrency(empty) is not None
    with pytest.raises(ValueError, match="no regions to summarise"):
        otf2viz.region_summary(empty)
    with pytest.raises(ValueError, match="no time to break down"):
        otf2viz.location_breakdown(empty)


def test_markers_are_optional(multiprocess_trace):
    ax = otf2viz.timeline(multiprocess_trace, markers=True)
    assert ax is not None  # no markers in this trace, and no crash either
