"""Core data model.

A :class:`Trace` is a flat, array-backed table of *region intervals*: one row per
matched Enter/Leave pair, with the enclosing location and the nesting depth.
Everything else in the package (filtering, statistics, plotting) reads that
table, so a trace can come from an OTF2 file, from a hand-written list of
events, or from any other tool that can produce start/stop pairs.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = ["Region", "Location", "Markers", "Trace", "RegionStat", "LocationStat"]


@dataclass(frozen=True)
class Region:
    """A code region (function, loop, barrier, ...) as defined by the trace."""

    name: str
    paradigm: str = "UNKNOWN"
    role: str = "UNKNOWN"
    source_file: str | None = None
    begin_line: int | None = None
    ref: int = -1

    @property
    def source(self) -> str | None:
        """``file:line`` if the trace recorded a source location."""
        if not self.source_file:
            return None
        if self.begin_line:
            return f"{self.source_file}:{self.begin_line}"
        return self.source_file


@dataclass(frozen=True)
class Location:
    """A stream of events: an OpenMP thread, an MPI rank's main thread, ..."""

    name: str
    group_ref: int = 0
    group_name: str = "Process"
    thread: int = 0
    ref: int = -1

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (self.group_ref, self.thread, self.name)


@dataclass(frozen=True)
class Markers:
    """Point-in-time events (thread fork / join), kept alongside the intervals."""

    time: np.ndarray
    location: np.ndarray
    kind: np.ndarray

    def __len__(self) -> int:
        return int(self.time.size)

    @classmethod
    def empty(cls) -> "Markers":
        return cls(
            time=np.empty(0, np.float64),
            location=np.empty(0, np.int32),
            kind=np.empty(0, object),
        )

    def _take(self, mask: np.ndarray) -> "Markers":
        return Markers(self.time[mask], self.location[mask], self.kind[mask])


@dataclass(frozen=True)
class RegionStat:
    """Aggregated timings for one region."""

    region: Region
    count: int
    inclusive: float
    exclusive: float
    mean: float
    minimum: float
    maximum: float

    @property
    def name(self) -> str:
        return self.region.name


@dataclass(frozen=True)
class LocationStat:
    """Aggregated timings for one location."""

    location: Location
    label: str
    count: int
    busy: float
    idle: float

    @property
    def utilisation(self) -> float:
        total = self.busy + self.idle
        return self.busy / total if total > 0 else 0.0


RegionSelector = str | Iterable[str] | re.Pattern | Callable[[Region], bool]


class Trace:
    """A set of region intervals plus the definitions they refer to.

    All times are in **seconds**, relative to the start of the measurement.
    The interval table is exposed as parallel NumPy arrays so that plotting and
    statistics stay vectorised:

    ``start``, ``stop``
        Interval bounds, ``float64``.
    ``location``, ``region``
        Indices into :attr:`locations` / :attr:`regions`, ``int32``.
    ``depth``
        Call-stack depth of the interval on its location, 0 for outermost.
    ``child_time``
        Total time spent inside nested regions, used for exclusive timings.
    """

    __slots__ = (
        "name",
        "source",
        "locations",
        "regions",
        "start",
        "stop",
        "location",
        "region",
        "depth",
        "child_time",
        "markers",
        "span",
        "timer_resolution",
    )

    def __init__(
        self,
        locations: Sequence[Location],
        regions: Sequence[Region],
        start,
        stop,
        location,
        region,
        depth=None,
        child_time=None,
        *,
        markers: Markers | None = None,
        span: tuple[float, float] | None = None,
        name: str = "trace",
        source: str | None = None,
        timer_resolution: int | None = None,
    ) -> None:
        self.locations = tuple(locations)
        self.regions = tuple(regions)
        self.start = np.asarray(start, dtype=np.float64)
        self.stop = np.asarray(stop, dtype=np.float64)
        self.location = np.asarray(location, dtype=np.int32)
        self.region = np.asarray(region, dtype=np.int32)
        n = self.start.size
        self.depth = (
            np.zeros(n, np.int32) if depth is None else np.asarray(depth, np.int32)
        )
        self.child_time = (
            np.zeros(n, np.float64)
            if child_time is None
            else np.asarray(child_time, np.float64)
        )
        self.markers = markers if markers is not None else Markers.empty()
        self.name = name
        self.source = source
        self.timer_resolution = timer_resolution

        for field in ("stop", "location", "region", "depth", "child_time"):
            if getattr(self, field).size != n:
                raise ValueError(f"{field!r} has a different length than 'start'")
        if n and (self.location.max() >= len(self.locations) or self.location.min() < 0):
            raise ValueError("location index out of range")
        if n and (self.region.max() >= len(self.regions) or self.region.min() < 0):
            raise ValueError("region index out of range")

        if span is not None:
            self.span = (float(span[0]), float(span[1]))
        elif n:
            self.span = (float(self.start.min()), float(self.stop.max()))
        else:
            self.span = (0.0, 0.0)

        self._sort()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_events(
        cls,
        events: Iterable[tuple],
        *,
        name: str = "trace",
        derive_nesting: bool = True,
        **kwargs,
    ) -> "Trace":
        """Build a trace from ``(location, region, start, stop)`` tuples.

        Useful for tests, for sketching a schedule by hand, and for feeding in
        timings from a tool that is not OTF2. ``location`` and ``region`` are
        plain labels; a ``location`` may also be a ``(process, thread)`` pair.

        >>> t = Trace.from_events([("thread 0", "work", 0.0, 1.0)])
        >>> len(t)
        1
        """
        loc_index: dict[object, int] = {}
        reg_index: dict[str, int] = {}
        locations: list[Location] = []
        regions: list[Region] = []
        starts, stops, locs, regs = [], [], [], []

        for event in events:
            loc_key, reg_name, t0, t1 = event
            if loc_key not in loc_index:
                if isinstance(loc_key, tuple) and len(loc_key) == 2:
                    group, thread = loc_key
                    group_ref = int(group) if str(group).isdigit() else len(loc_index)
                    locations.append(
                        Location(
                            name=f"thread {thread}",
                            group_ref=group_ref,
                            group_name=str(group),
                            thread=int(thread),
                        )
                    )
                else:
                    locations.append(
                        Location(name=str(loc_key), thread=len(loc_index))
                    )
                loc_index[loc_key] = len(locations) - 1
            if reg_name not in reg_index:
                regions.append(Region(name=str(reg_name)))
                reg_index[reg_name] = len(regions) - 1
            starts.append(float(t0))
            stops.append(float(t1))
            locs.append(loc_index[loc_key])
            regs.append(reg_index[reg_name])

        start = np.asarray(starts, np.float64)
        stop = np.asarray(stops, np.float64)
        loc = np.asarray(locs, np.int32)
        depth, child_time = (
            _derive_nesting(start, stop, loc)
            if derive_nesting
            else (None, None)
        )
        return cls(
            locations,
            regions,
            start,
            stop,
            loc,
            np.asarray(regs, np.int32),
            depth,
            child_time,
            name=name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # basics
    # ------------------------------------------------------------------
    def _sort(self) -> None:
        """Order rows by (location, start, depth) for cache-friendly plotting."""
        if self.start.size < 2:
            return
        order = np.lexsort((self.depth, self.start, self.location))
        if np.array_equal(order, np.arange(order.size)):
            return
        self.start = self.start[order]
        self.stop = self.stop[order]
        self.location = self.location[order]
        self.region = self.region[order]
        self.depth = self.depth[order]
        self.child_time = self.child_time[order]

    def __len__(self) -> int:
        return int(self.start.size)

    def __repr__(self) -> str:
        lo, hi = self.span
        return (
            f"<Trace {self.name!r}: {len(self)} intervals, "
            f"{len(self.locations)} locations, {len(self.regions)} regions, "
            f"{_format_duration(hi - lo)}>"
        )

    @property
    def duration(self) -> float:
        """Wall-clock length of the trace window, in seconds."""
        return self.span[1] - self.span[0]

    @property
    def durations(self) -> np.ndarray:
        """Inclusive duration of every interval, in seconds."""
        return self.stop - self.start

    @property
    def max_depth(self) -> int:
        return int(self.depth.max()) if len(self) else 0

    @property
    def n_processes(self) -> int:
        return len({loc.group_ref for loc in self.locations})

    def region_names(self) -> list[str]:
        """Region names ordered by exclusive time, longest first."""
        return [stat.name for stat in self.region_stats()]

    def location_labels(self) -> list[str]:
        """Row labels, disambiguated by process only when there is more than one."""
        if self.n_processes > 1:
            return [f"P{loc.group_ref} {loc.name}" for loc in self.locations]
        return [loc.name for loc in self.locations]

    def location_order(self) -> list[int]:
        """Location indices sorted by (process, thread) — the plotting row order."""
        return sorted(
            range(len(self.locations)), key=lambda i: self.locations[i].sort_key
        )

    # ------------------------------------------------------------------
    # filtering
    # ------------------------------------------------------------------
    def filter(
        self,
        *,
        regions: RegionSelector | None = None,
        exclude_regions: RegionSelector | None = None,
        paradigms: str | Iterable[str] | None = None,
        roles: str | Iterable[str] | None = None,
        exclude_roles: str | Iterable[str] | None = None,
        locations: Iterable[int] | Iterable[str] | None = None,
        max_depth: int | None = None,
        min_duration: float | None = None,
        time_range: tuple[float, float] | None = None,
        clip: bool = False,
        name: str | None = None,
    ) -> "Trace":
        """Return a new trace holding the subset of intervals that match.

        ``regions`` / ``exclude_regions`` accept an exact name, an iterable of
        names, a compiled regular expression, or a predicate on :class:`Region`.
        Locations are kept even when they end up empty, so that a filtered
        timeline still shows an idle thread as an idle row; pass ``locations``
        to drop rows instead.

        With ``clip=True`` intervals that straddle ``time_range`` are cut to the
        window, which is what you want when the filtered trace feeds statistics
        rather than a plot.
        """
        keep = np.ones(len(self), dtype=bool)

        region_match = _match_regions(self.regions, regions)
        if region_match is not None:
            keep &= region_match[self.region]
        exclude_match = _match_regions(self.regions, exclude_regions)
        if exclude_match is not None:
            keep &= ~exclude_match[self.region]
        if paradigms is not None:
            wanted = _as_upper_set(paradigms)
            sel = np.array([r.paradigm.upper() in wanted for r in self.regions])
            keep &= sel[self.region]
        if roles is not None:
            wanted = _as_upper_set(roles)
            sel = np.array([r.role.upper() in wanted for r in self.regions])
            keep &= sel[self.region]
        if exclude_roles is not None:
            wanted = _as_upper_set(exclude_roles)
            sel = np.array([r.role.upper() in wanted for r in self.regions])
            keep &= ~sel[self.region]
        if max_depth is not None:
            keep &= self.depth <= max_depth
        if min_duration is not None:
            keep &= self.durations >= min_duration

        start, stop = self.start, self.stop
        span = self.span
        if time_range is not None:
            t0, t1 = float(time_range[0]), float(time_range[1])
            keep &= (stop > t0) & (start < t1)
            span = (t0, t1)
            if clip:
                start = np.clip(start, t0, t1)
                stop = np.clip(stop, t0, t1)

        kept_locations = self.locations
        loc_idx = self.location
        markers = self.markers
        if locations is not None:
            wanted_idx = self._resolve_locations(locations)
            loc_keep = np.zeros(len(self.locations), dtype=bool)
            loc_keep[list(wanted_idx)] = True
            keep &= loc_keep[self.location]
            remap = -np.ones(len(self.locations), np.int32)
            ordered = [i for i in range(len(self.locations)) if loc_keep[i]]
            remap[ordered] = np.arange(len(ordered), dtype=np.int32)
            kept_locations = tuple(self.locations[i] for i in ordered)
            loc_idx = remap[self.location]
            markers = markers._take(loc_keep[markers.location])
            markers = Markers(
                markers.time, remap[markers.location], markers.kind
            )

        if time_range is not None:
            m = (markers.time >= span[0]) & (markers.time <= span[1])
            markers = markers._take(m)

        # Regions are pruned (an unused region would only pad the legend);
        # locations are not (an idle row is information).
        used = np.zeros(len(self.regions), dtype=bool)
        used[self.region[keep]] = True
        reg_remap = -np.ones(len(self.regions), np.int32)
        kept_regions = tuple(r for r, u in zip(self.regions, used) if u)
        reg_remap[np.flatnonzero(used)] = np.arange(len(kept_regions), dtype=np.int32)

        return Trace(
            kept_locations,
            kept_regions,
            start[keep],
            stop[keep],
            loc_idx[keep],
            reg_remap[self.region[keep]],
            self.depth[keep],
            self.child_time[keep],
            markers=markers,
            span=span,
            name=name if name is not None else self.name,
            source=self.source,
            timer_resolution=self.timer_resolution,
        )

    def select_time(self, t0: float, t1: float, *, clip: bool = False) -> "Trace":
        """Shorthand for ``filter(time_range=(t0, t1))``."""
        return self.filter(time_range=(t0, t1), clip=clip)

    def _resolve_locations(self, locations) -> list[int]:
        labels = self.location_labels()
        resolved: list[int] = []
        for item in locations:
            if isinstance(item, (int, np.integer)):
                resolved.append(int(item))
            else:
                text = str(item)
                matches = [
                    i
                    for i, (label, loc) in enumerate(zip(labels, self.locations))
                    if text in (label, loc.name)
                ]
                if not matches:
                    raise KeyError(f"no location named {text!r}")
                resolved.extend(matches)
        return sorted(set(resolved))

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    @property
    def exclusive(self) -> np.ndarray:
        """Duration of every interval minus the time spent in nested regions."""
        return np.maximum(self.durations - self.child_time, 0.0)

    def region_stats(self, *, sort_by: str = "exclusive") -> list[RegionStat]:
        """Per-region timings, sorted by ``exclusive``, ``inclusive`` or ``count``.

        Aggregated with ``bincount`` over one sort of the region column, so the
        cost stays linear-ish in the number of intervals rather than growing
        with the number of regions as well.
        """
        n_regions = len(self.regions)
        if not len(self) or not n_regions:
            return []
        durations = self.durations
        counts = np.bincount(self.region, minlength=n_regions)
        inclusive = np.bincount(self.region, weights=durations, minlength=n_regions)
        exclusive = np.bincount(
            self.region, weights=self.exclusive, minlength=n_regions
        )
        order = np.argsort(self.region, kind="stable")
        grouped = self.region[order]
        heads = np.flatnonzero(np.append(True, grouped[1:] != grouped[:-1]))
        present = grouped[heads]
        sorted_durations = durations[order]
        minima = np.minimum.reduceat(sorted_durations, heads)
        maxima = np.maximum.reduceat(sorted_durations, heads)

        stats = [
            RegionStat(
                region=self.regions[idx],
                count=int(counts[idx]),
                inclusive=float(inclusive[idx]),
                exclusive=float(exclusive[idx]),
                mean=float(inclusive[idx] / counts[idx]),
                minimum=float(minima[group]),
                maximum=float(maxima[group]),
            )
            for group, idx in enumerate(present)
        ]
        key = {
            "exclusive": lambda s: s.exclusive,
            "inclusive": lambda s: s.inclusive,
            "count": lambda s: s.count,
            "name": lambda s: s.name,
        }[sort_by]
        reverse = sort_by != "name"
        return sorted(stats, key=key, reverse=reverse)

    def rows_for_location(self, index: int) -> slice:
        """Row range of one location — the table is sorted by location."""
        lo = int(np.searchsorted(self.location, index, side="left"))
        hi = int(np.searchsorted(self.location, index, side="right"))
        return slice(lo, hi)

    def location_stats(self) -> list[LocationStat]:
        """Per-location busy/idle time over the trace window."""
        window = self.duration
        labels = self.location_labels()
        stats = []
        for idx in self.location_order():
            rows = self.rows_for_location(idx)
            merged_start, merged_stop = _merge_intervals(
                self.start[rows], self.stop[rows]
            )
            busy = float((merged_stop - merged_start).sum())
            stats.append(
                LocationStat(
                    location=self.locations[idx],
                    label=labels[idx],
                    count=rows.stop - rows.start,
                    busy=busy,
                    idle=max(window - busy, 0.0),
                )
            )
        return stats

    def concurrency(self) -> tuple[np.ndarray, np.ndarray]:
        """Number of simultaneously busy locations as a step function.

        Returns ``(times, counts)`` for ``ax.step(..., where="post")``. Nested
        intervals on one location are merged first, so a location is counted
        once no matter how deep its call stack is.
        """
        starts, stops = [], []
        for idx in range(len(self.locations)):
            rows = self.rows_for_location(idx)
            if rows.start == rows.stop:
                continue
            s, e = _merge_intervals(self.start[rows], self.stop[rows])
            starts.append(s)
            stops.append(e)
        if not starts:
            return (
                np.array(self.span, np.float64),
                np.zeros(2, np.int32),
            )
        s = np.concatenate(starts)
        e = np.concatenate(stops)
        times = np.concatenate([s, e])
        deltas = np.concatenate([np.ones(s.size, np.int32), -np.ones(e.size, np.int32)])
        # At an identical timestamp, apply the -1 before the +1 so that a
        # hand-off between regions does not show up as a spurious peak.
        order = np.lexsort((deltas, times))
        times, counts = times[order], np.cumsum(deltas[order])
        # Keep the last value per distinct timestamp.
        last = np.append(times[1:] != times[:-1], True)
        times, counts = times[last], counts[last]
        times = np.concatenate([[self.span[0]], times, [self.span[1]]])
        counts = np.concatenate([[0], counts, [counts[-1]]])
        return times, counts

    # ------------------------------------------------------------------
    # interop
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, list]:
        """Column-oriented plain-Python view of the interval table."""
        labels = self.location_labels()
        return {
            "location": [labels[i] for i in self.location],
            "process": [self.locations[i].group_ref for i in self.location],
            "thread": [self.locations[i].thread for i in self.location],
            "region": [self.regions[i].name for i in self.region],
            "paradigm": [self.regions[i].paradigm for i in self.region],
            "role": [self.regions[i].role for i in self.region],
            "depth": self.depth.tolist(),
            "start": self.start.tolist(),
            "stop": self.stop.tolist(),
            "duration": self.durations.tolist(),
            "exclusive": np.maximum(self.durations - self.child_time, 0.0).tolist(),
        }

    def to_dataframe(self):
        """The interval table as a :mod:`pandas` DataFrame (pandas is optional)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Trace.to_dataframe() needs pandas: pip install 'otf2viz[pandas]'"
            ) from exc
        return pd.DataFrame(self.to_dict())

    def _repr_html_(self) -> str:
        """Rich summary shown when a Trace is the last expression in a cell."""
        rows = self.region_stats()[:10]
        total = self.duration or 1.0
        body = "".join(
            "<tr>"
            f"<td style='text-align:left'>{html.escape(s.name)}</td>"
            f"<td>{s.count}</td>"
            f"<td>{_format_duration(s.exclusive)}</td>"
            f"<td>{_format_duration(s.inclusive)}</td>"
            f"<td>{100 * s.exclusive / (total * max(len(self.locations), 1)):.1f}%</td>"
            "</tr>"
            for s in rows
        )
        more = len(self.region_stats()) - len(rows)
        footer = (
            f"<tr><td colspan='5' style='text-align:left;opacity:.6'>"
            f"+{more} more region(s)</td></tr>"
            if more > 0
            else ""
        )
        return (
            "<div style='font-family:system-ui,-apple-system,sans-serif;font-size:13px'>"
            f"<b>{html.escape(self.name)}</b>"
            f" &mdash; {len(self):,} intervals, {len(self.locations)} locations, "
            f"{len(self.regions)} regions, {_format_duration(self.duration)}"
            "<table style='border-collapse:collapse;margin-top:6px;font-size:12px'>"
            "<thead><tr style='text-align:right;opacity:.7'>"
            "<th style='text-align:left'>region</th><th>calls</th>"
            "<th>exclusive</th><th>inclusive</th><th>share</th>"
            "</tr></thead>"
            f"<tbody style='text-align:right'>{body}{footer}</tbody></table></div>"
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _as_upper_set(value: str | Iterable[str]) -> set[str]:
    if isinstance(value, str):
        return {value.upper()}
    return {str(v).upper() for v in value}


def _match_regions(
    regions: Sequence[Region], selector: RegionSelector | None
) -> np.ndarray | None:
    """Boolean mask over ``regions`` for a name / names / regex / predicate."""
    if selector is None:
        return None
    if isinstance(selector, re.Pattern):
        return np.array([bool(selector.search(r.name)) for r in regions])
    if callable(selector):
        return np.array([bool(selector(r)) for r in regions])
    names = {selector} if isinstance(selector, str) else set(selector)
    return np.array([r.name in names for r in regions])


def _merge_intervals(
    start: np.ndarray, stop: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Union of possibly overlapping/nested intervals, as disjoint (s, e) pairs."""
    if start.size == 0:
        return np.empty(0, np.float64), np.empty(0, np.float64)
    order = np.argsort(start, kind="stable")
    s, e = start[order], stop[order]
    running_max = np.maximum.accumulate(e)
    is_new = np.empty(s.size, dtype=bool)
    is_new[0] = True
    is_new[1:] = s[1:] > running_max[:-1]
    heads = np.flatnonzero(is_new)
    return s[heads], np.maximum.reduceat(e, heads)


def _derive_nesting(
    start: np.ndarray, stop: np.ndarray, location: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recover call-stack depth and nested-child time from bare intervals."""
    depth = np.zeros(start.size, np.int32)
    child_time = np.zeros(start.size, np.float64)
    for loc in np.unique(location):
        rows = np.flatnonzero(location == loc)
        # Outer intervals first: earlier start wins, longer wins on a tie.
        rows = rows[np.lexsort((-stop[rows], start[rows]))]
        stack: list[int] = []
        for row in rows:
            while stack and stop[stack[-1]] <= start[row]:
                stack.pop()
            if stack:
                depth[row] = depth[stack[-1]] + 1
                child_time[stack[-1]] += stop[row] - start[row]
            stack.append(int(row))
    return depth, child_time


def _format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. ``1.23 ms``."""
    if seconds is None or not np.isfinite(seconds):
        return "?"
    for limit, factor, unit in (
        (1e-6, 1e9, "ns"),
        (1e-3, 1e6, "µs"),
        (1.0, 1e3, "ms"),
    ):
        if abs(seconds) < limit:
            return f"{seconds * factor:.3g} {unit}"
    return f"{seconds:.3g} s"
