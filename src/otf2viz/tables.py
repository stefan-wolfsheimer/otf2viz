"""Tabular views of a trace.

Every chart in this package has a table behind it. That is not decoration: a
table is how someone reads an exact value, how a screen reader gets at the
data, and how you check a plot that looks surprising.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Sequence

from .model import Trace, _format_duration

__all__ = ["Table", "region_table", "location_table"]


@dataclass
class Table:
    """A small result table that renders in Jupyter and converts to pandas."""

    columns: tuple[str, ...]
    rows: list[tuple]
    title: str = ""
    #: Column indices to right-align (numbers).
    numeric: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def to_dict(self) -> dict[str, list]:
        return {
            name: [row[i] for row in self.rows] for i, name in enumerate(self.columns)
        }

    def to_dataframe(self):
        """Requires pandas (``pip install 'otf2viz[pandas]'``)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Table.to_dataframe() needs pandas: pip install 'otf2viz[pandas]'"
            ) from exc
        return pd.DataFrame(self.rows, columns=list(self.columns))

    def __repr__(self) -> str:
        widths = [
            max(len(str(name)), *(len(str(row[i])) for row in self.rows or [self.columns]))
            for i, name in enumerate(self.columns)
        ]
        lines = [self.title] if self.title else []
        lines.append("  ".join(n.ljust(w) for n, w in zip(self.columns, widths)))
        lines.append("  ".join("-" * w for w in widths))
        lines += [
            "  ".join(str(v).ljust(w) for v, w in zip(row, widths)) for row in self.rows
        ]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        head = "".join(
            f"<th style='text-align:{'right' if i in self.numeric else 'left'};"
            f"padding:2px 10px 2px 0;opacity:.7;font-weight:600'>"
            f"{html.escape(str(name))}</th>"
            for i, name in enumerate(self.columns)
        )
        body = "".join(
            "<tr>"
            + "".join(
                f"<td style='text-align:{'right' if i in self.numeric else 'left'};"
                f"padding:2px 10px 2px 0'>{html.escape(str(value))}</td>"
                for i, value in enumerate(row)
            )
            + "</tr>"
            for row in self.rows
        )
        caption = (
            f"<div style='margin-bottom:4px;font-weight:600'>"
            f"{html.escape(self.title)}</div>"
            if self.title
            else ""
        )
        return (
            "<div style='font-family:system-ui,-apple-system,sans-serif;"
            "font-size:12px;font-variant-numeric:tabular-nums'>"
            f"{caption}<table style='border-collapse:collapse'>"
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        )


def region_table(
    trace: Trace, *, top: int | None = None, sort_by: str = "exclusive"
) -> Table:
    """Per-region call counts and timings — the table behind ``region_summary``."""
    stats = trace.region_stats(sort_by=sort_by)
    if top is not None:
        stats = stats[:top]
    total = sum(s.exclusive for s in stats) or 1.0
    rows = [
        (
            stat.name,
            stat.region.paradigm,
            stat.count,
            _format_duration(stat.exclusive),
            _format_duration(stat.inclusive),
            _format_duration(stat.mean),
            _format_duration(stat.maximum),
            f"{100 * stat.exclusive / total:.1f}%",
        )
        for stat in stats
    ]
    return Table(
        columns=(
            "region",
            "paradigm",
            "calls",
            "exclusive",
            "inclusive",
            "mean",
            "max",
            "share",
        ),
        rows=rows,
        title=f"{trace.name} — regions by {sort_by} time",
        numeric=(2, 3, 4, 5, 6, 7),
    )


def location_table(trace: Trace) -> Table:
    """Per-location busy/idle time — the table behind ``location_breakdown``."""
    rows = [
        (
            stat.label,
            stat.location.group_ref,
            stat.count,
            _format_duration(stat.busy),
            _format_duration(stat.idle),
            f"{100 * stat.utilisation:.1f}%",
        )
        for stat in trace.location_stats()
    ]
    return Table(
        columns=("location", "process", "intervals", "busy", "idle", "utilisation"),
        rows=rows,
        title=f"{trace.name} — locations",
        numeric=(1, 2, 3, 4, 5),
    )


def summary(trace: Trace) -> Table:
    """One-line-per-fact overview of the whole measurement."""
    stats = trace.region_stats()
    facts: Sequence[tuple[str, str]] = (
        ("source", trace.source or "—"),
        ("duration", _format_duration(trace.duration)),
        ("locations", f"{len(trace.locations)}"),
        ("processes", f"{trace.n_processes}"),
        ("regions", f"{len(trace.regions)}"),
        ("intervals", f"{len(trace):,}"),
        ("max call depth", f"{trace.max_depth}"),
        ("hottest region", stats[0].name if stats else "—"),
    )
    return Table(
        columns=("property", "value"),
        rows=[tuple(fact) for fact in facts],
        title=trace.name,
    )
