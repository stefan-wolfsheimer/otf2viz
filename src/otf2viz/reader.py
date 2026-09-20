"""Read an OTF2 trace (Score-P, Extrae, ...) into a :class:`~otf2viz.model.Trace`.

The heavy lifting is one pass per location: Enter events are pushed on a stack
and popped by the matching Leave, which yields the interval, its call-stack
depth and the time spent in nested regions. Locations are independent, so the
pass can be spread over a process pool for large traces.
"""

from __future__ import annotations

import multiprocessing
import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

from .model import Location, Markers, Region, Trace

__all__ = ["read_trace", "find_traces"]

#: Events kept as point markers on the timeline.
_MARKER_EVENTS = ("ThreadFork", "ThreadJoin")

#: Regions that only exist because the measurement system is running.
DEFAULT_EXCLUDED_REGIONS = ("TRACE BUFFER FLUSH", "MEASUREMENT OFF")


def find_traces(root: str | os.PathLike) -> list[Path]:
    """All ``traces.otf2`` anchor files under ``root``, sorted by path."""
    return sorted(Path(root).rglob("traces.otf2"))


def _resolve_anchor(path: str | os.PathLike) -> Path:
    """Accept the anchor file itself or any directory that contains one."""
    path = Path(path).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in (
            path / "traces.otf2",
            path / "scorep_measurements" / "traces.otf2",
        ):
            if candidate.is_file():
                return candidate
        found = find_traces(path)
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError(
                f"{path} contains {len(found)} traces; pass one of them explicitly, "
                f"e.g. {found[0]}"
            )
    raise FileNotFoundError(f"no OTF2 anchor file (traces.otf2) at {path}")


def read_trace(
    path: str | os.PathLike,
    *,
    workers: int | None = None,
    name: str | None = None,
    exclude_regions: Iterable[str] | None = DEFAULT_EXCLUDED_REGIONS,
    markers: bool = True,
) -> Trace:
    """Read an OTF2 trace into a :class:`~otf2viz.model.Trace`.

    Parameters
    ----------
    path
        The ``traces.otf2`` anchor file, or a directory containing one (a
        Score-P experiment directory works directly).
    workers
        Number of reader processes. ``None`` picks one process per location up
        to the CPU count, but only once the trace has enough locations to make
        that worthwhile; ``1`` forces a serial read.
    name
        Display name; defaults to the experiment directory name.
    exclude_regions
        Region names to drop, by default the measurement-system artefacts that
        would otherwise dominate a timeline.
    markers
        Keep ``ThreadFork`` / ``ThreadJoin`` events as point markers.

    Notes
    -----
    Requires the ``otf2`` Python bindings that ship with Score-P / OTF2; they
    are not installable from PyPI. If ``import otf2`` fails, source the same
    environment you used to run ``scorep``.
    """
    otf2 = _import_otf2()
    anchor = _resolve_anchor(path)

    with otf2.reader.open(str(anchor)) as trace:
        definitions = trace.definitions
        clock = definitions.clock_properties
        raw_locations = list(definitions.locations)
        locations, location_refs = _build_locations(raw_locations)

    if not raw_locations:
        raise ValueError(f"{anchor} contains no locations")

    n_workers = _pick_workers(workers, len(location_refs))
    args = [(str(anchor), ref, markers) for ref in location_refs]
    if n_workers > 1:
        try:
            with multiprocessing.Pool(n_workers) as pool:
                chunks = pool.starmap(_read_location, args)
        except (OSError, ValueError, RuntimeError) as exc:  # pragma: no cover
            warnings.warn(
                f"parallel read failed ({exc}); falling back to a serial read",
                RuntimeWarning,
                stacklevel=2,
            )
            chunks = [_read_location(*a) for a in args]
    else:
        chunks = [_read_location(*a) for a in args]

    ticks = float(clock.timer_resolution)
    offset = float(clock.global_offset)
    excluded = set(exclude_regions or ())

    regions: list[Region] = []
    region_index: dict[int, int] = {}
    starts, stops, locs, regs, depths, child_times = [], [], [], [], [], []
    marker_time, marker_loc, marker_kind = [], [], []
    unmatched = 0
    unclosed = 0

    for loc_idx, chunk in enumerate(chunks):
        unmatched += chunk["unmatched"]
        unclosed += chunk["unclosed"]
        for region_def in chunk["regions"]:
            if region_def.ref not in region_index:
                region_index[region_def.ref] = len(regions)
                regions.append(region_def)
        for ref, t0, t1, depth, child in zip(
            chunk["region_ref"],
            chunk["start"],
            chunk["stop"],
            chunk["depth"],
            chunk["child"],
        ):
            if regions[region_index[ref]].name in excluded:
                continue
            starts.append((t0 - offset) / ticks)
            stops.append((t1 - offset) / ticks)
            locs.append(loc_idx)
            regs.append(region_index[ref])
            depths.append(depth)
            child_times.append(child / ticks)
        for t, kind in zip(chunk["marker_time"], chunk["marker_kind"]):
            marker_time.append((t - offset) / ticks)
            marker_loc.append(loc_idx)
            marker_kind.append(kind)

    if unmatched:
        warnings.warn(
            f"{unmatched} Leave event(s) had no matching Enter and were skipped",
            RuntimeWarning,
            stacklevel=2,
        )
    if unclosed:
        warnings.warn(
            f"{unclosed} region(s) were still open at the end of the trace and "
            "were closed at the last recorded timestamp",
            RuntimeWarning,
            stacklevel=2,
        )

    span_end = float(clock.trace_length) / ticks if clock.trace_length else None
    if span_end is None or (stops and max(stops) > span_end):
        span_end = max(stops) if stops else 0.0

    return Trace(
        locations,
        regions,
        np.asarray(starts, np.float64),
        np.asarray(stops, np.float64),
        np.asarray(locs, np.int32),
        np.asarray(regs, np.int32),
        np.asarray(depths, np.int32),
        np.asarray(child_times, np.float64),
        markers=Markers(
            np.asarray(marker_time, np.float64),
            np.asarray(marker_loc, np.int32),
            np.asarray(marker_kind, dtype=object),
        ),
        span=(0.0, span_end),
        name=name or _default_name(anchor),
        source=str(anchor),
        timer_resolution=int(clock.timer_resolution),
    )


# ----------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------
def _import_otf2():
    try:
        import otf2
        import otf2.reader  # noqa: F401  (registers reader.open)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "otf2viz.read_trace() needs the 'otf2' Python bindings that ship with "
            "Score-P/OTF2. Activate the environment you ran 'scorep' in, or build "
            "OTF2 with --enable-python. Everything else in otf2viz works without it "
            "(see Trace.from_events)."
        ) from exc
    return otf2


def _default_name(anchor: Path) -> str:
    parent = anchor.parent.name
    return parent if parent not in ("", ".") else anchor.name


def _pick_workers(workers: int | None, n_locations: int) -> int:
    if workers is not None:
        return max(1, min(int(workers), n_locations))
    if n_locations < 8:
        return 1
    return min(n_locations, os.cpu_count() or 1)


def _build_locations(raw_locations) -> tuple[list[Location], list[int]]:
    """Map OTF2 location definitions onto our dense, ordered rows."""
    threads_per_group: dict[int, int] = {}
    locations, refs = [], []
    for raw in raw_locations:
        group = raw.group
        group_ref = int(getattr(group, "_ref", 0))
        thread = threads_per_group.get(group_ref, 0)
        threads_per_group[group_ref] = thread + 1
        locations.append(
            Location(
                name=str(raw.name),
                group_ref=group_ref,
                group_name=str(getattr(group, "name", "Process")),
                thread=thread,
                ref=int(getattr(raw, "_ref", thread)),
            )
        )
        refs.append(int(raw._ref))
    return locations, refs


def _read_location(anchor: str, location_ref: int, keep_markers: bool) -> dict:
    """Extract every interval on one location. Runs in a worker process.

    Returns plain lists (and picklable :class:`Region` values) with timestamps
    still in raw clock ticks; the caller converts to seconds once.
    """
    import otf2

    region_ref: list[int] = []
    start: list[int] = []
    stop: list[int] = []
    depth: list[int] = []
    child: list[int] = []
    regions: dict[int, Region] = {}
    marker_time: list[int] = []
    marker_kind: list[str] = []
    unmatched = 0
    last_time = 0

    with otf2.reader.open(anchor) as trace:
        location = next(
            loc for loc in trace.definitions.locations if int(loc._ref) == location_ref
        )
        # (enter time, row index) for every region currently on the stack.
        stack: list[tuple[int, int]] = []
        for _, event in trace.events([location]):
            kind = type(event).__name__
            time = getattr(event, "time", None)
            if time is not None:
                last_time = max(last_time, int(time))
            if kind == "Enter":
                ref = int(event.region._ref)
                if ref not in regions:
                    regions[ref] = _convert_region(event.region)
                region_ref.append(ref)
                start.append(int(event.time))
                stop.append(int(event.time))
                depth.append(len(stack))
                child.append(0)
                stack.append((int(event.time), len(region_ref) - 1))
            elif kind == "Leave":
                if not stack:
                    unmatched += 1
                    continue
                enter_time, row = stack.pop()
                stop[row] = int(event.time)
                if stack:
                    child[stack[-1][1]] += int(event.time) - enter_time
            elif keep_markers and kind in _MARKER_EVENTS:
                marker_time.append(int(event.time))
                marker_kind.append(kind)

    # Regions left open (truncated trace, buffer flush) end at the last event.
    for _, row in stack:
        stop[row] = last_time

    return {
        "regions": list(regions.values()),
        "region_ref": region_ref,
        "start": start,
        "stop": stop,
        "depth": depth,
        "child": child,
        "marker_time": marker_time,
        "marker_kind": marker_kind,
        "unmatched": unmatched,
        "unclosed": len(stack),
    }


def _convert_region(raw) -> Region:
    """OTF2 region definition -> our plain, picklable :class:`Region`."""
    source_file = str(getattr(raw, "source_file", "") or "") or None
    begin_line = int(getattr(raw, "begin_line_number", 0) or 0) or None
    return Region(
        name=str(raw.name),
        paradigm=_enum_name(getattr(raw, "paradigm", None)),
        role=_enum_name(getattr(raw, "region_role", None)),
        source_file=source_file,
        begin_line=begin_line,
        ref=int(raw._ref),
    )


def _enum_name(value) -> str:
    """``Paradigm.OPENMP`` (a ctypes enum, not :mod:`enum`) -> ``"OPENMP"``."""
    if value is None:
        return "UNKNOWN"
    return str(value).rsplit(".", 1)[-1]
