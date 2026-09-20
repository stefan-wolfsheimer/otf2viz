"""Colour assignment and theming rules."""

from __future__ import annotations

import pytest

import matplotlib as mpl

from otf2viz import DARK, LIGHT, ColorMap, Trace
from otf2viz.theme import get_theme, set_theme, use


def test_slots_follow_the_ranking_of_exclusive_time(simple_trace):
    colormap = ColorMap.from_trace(simple_trace)
    assert colormap.keys[0] == "barrier"  # the longest exclusive time
    assert colormap.slot("barrier") == 0
    assert colormap.style("barrier")["facecolor"] == LIGHT.categorical[0]


def test_colour_follows_the_entity_not_the_filtered_rank(simple_trace):
    colormap = ColorMap.from_trace(simple_trace)
    before = colormap.style("compute")
    # Drop the series that outranked it; the survivor keeps its colour.
    filtered = simple_trace.filter(exclude_regions="barrier")
    assert colormap.style("compute") == before
    assert ColorMap.from_trace(filtered).style("compute") != before


def test_extra_series_fold_into_other():
    trace = Trace.from_events(
        [(f"t{i % 2}", f"region{i}", i, i + 1) for i in range(12)]
    )
    colormap = ColorMap.from_trace(trace, max_colors=3)
    assert len(colormap.slots) == 3
    assert len(colormap.folded) == 9
    folded_key = colormap.folded[0]
    assert colormap.style(folded_key)["facecolor"] == LIGHT.other
    assert colormap.label(folded_key) == colormap.other_label
    assert "9" in colormap.other_label


def test_entries_include_other_only_when_folded_keys_are_present():
    trace = Trace.from_events(
        [(f"t{i % 2}", f"region{i}", i, i + 1) for i in range(12)]
    )
    colormap = ColorMap.from_trace(trace, max_colors=3)
    labels = [label for label, _ in colormap.entries()]
    assert labels[-1] == colormap.other_label

    top_only = [label for label, _ in colormap.entries(keys=colormap.keys[:2])]
    assert colormap.other_label not in top_only
    assert len(top_only) == 2


def test_hues_repeat_only_with_a_texture():
    plain, hatch = LIGHT.slot(0)
    repeat_colour, repeat_hatch = LIGHT.slot(len(LIGHT.categorical))
    assert hatch is None
    assert repeat_colour == plain  # the hue is reused, not invented
    assert repeat_hatch is not None  # ...but never colour alone


def test_slot_beyond_the_palette_is_an_error():
    with pytest.raises(IndexError, match="fold the rest"):
        LIGHT.slot(LIGHT.max_series)
    with pytest.raises(ValueError, match="exceeds"):
        ColorMap(["a"], max_colors=LIGHT.max_series + 1)


def test_grouping_by_paradigm_and_role(multiprocess_trace):
    by_paradigm = ColorMap.from_trace(multiprocess_trace, by="paradigm")
    assert set(by_paradigm.keys) == {"MPI", "OPENMP"}
    by_role = ColorMap.from_trace(multiprocess_trace, by="role")
    assert set(by_role.keys) == {"COLLECTIVE", "LOOP"}
    with pytest.raises(ValueError, match="unknown grouping"):
        ColorMap.from_trace(multiprocess_trace, by="nonsense")


def test_dark_theme_is_selected_not_flipped():
    assert DARK.categorical != LIGHT.categorical
    assert len(DARK.categorical) == len(LIGHT.categorical)
    assert DARK.surface != LIGHT.surface


def test_set_theme_switches_the_default():
    previous = get_theme()
    try:
        assert set_theme("dark") is DARK
        assert get_theme() is DARK
        assert get_theme("light") is LIGHT
        assert get_theme(DARK) is DARK
    finally:
        set_theme(previous)
    with pytest.raises(ValueError, match="unknown theme"):
        get_theme("solarized")


def test_use_restores_only_its_own_rcparams():
    """A theme block puts back what it set, and leaves everything else alone."""
    saved = {k: mpl.rcParams[k] for k in ("interactive", "axes.facecolor")}
    try:
        mpl.rcParams["interactive"] = False
        with use("dark") as theme:
            assert theme is DARK
            assert mpl.rcParams["axes.facecolor"] == DARK.surface
            mpl.rcParams["interactive"] = True
        assert mpl.rcParams["axes.facecolor"] == saved["axes.facecolor"]
        assert mpl.rcParams["interactive"] is True
    finally:
        mpl.rcParams.update(saved)


def test_plotting_keeps_interactive_mode_enabled_by_the_kernel(
    simple_trace, monkeypatch
):
    """Jupyter turns ``interactive`` on while the first figure is being made.

    Reverting it there leaves the inline backend unable to flush any later
    figure, so every plot after the first comes out blank until the kernel is
    restarted. Stand in for the kernel by flipping the flag from inside
    ``plt.subplots``, which is where ``new_axes`` opens its theme block.
    """
    import matplotlib.pyplot as plt

    from otf2viz import timeline

    real_subplots = plt.subplots

    def subplots_enabling_interactive(*args, **kwargs):
        mpl.rcParams["interactive"] = True
        return real_subplots(*args, **kwargs)

    saved = mpl.rcParams["interactive"]
    try:
        monkeypatch.setattr(plt, "subplots", subplots_enabling_interactive)
        mpl.rcParams["interactive"] = False
        timeline(simple_trace)
        assert mpl.rcParams["interactive"] is True
    finally:
        mpl.rcParams["interactive"] = saved


def test_interval_keys_align_with_the_table(simple_trace):
    colormap = ColorMap.from_trace(simple_trace)
    keys = colormap.interval_keys(simple_trace)
    assert len(keys) == len(simple_trace)
    assert keys == [simple_trace.regions[i].name for i in simple_trace.region]
