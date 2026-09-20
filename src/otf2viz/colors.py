"""Categorical colour assignment.

One :class:`ColorMap` maps trace keys (region names, paradigms, roles) onto the
theme's fixed categorical slots. Two rules are worth stating, because they are
what makes a multi-panel figure readable:

* **Colour follows the entity, not its rank.** Build the map once from the full
  trace and pass it to every plot; filtering then dims or drops series without
  repainting the survivors.
* **Hues are never generated.** Past the theme's slots, series fold into a
  single neutral "other" bucket. Raise ``max_colors`` and the extra series
  reuse a hue *plus* a texture, so identity is still not colour alone.
"""

from __future__ import annotations

from typing import Iterable

from .model import Trace
from .theme import Theme, get_theme

__all__ = ["ColorMap", "OTHER"]

#: Bucket label for series past ``max_colors``.
OTHER = "other"

_KEY_FUNCS = {
    "region": lambda region: region.name,
    "paradigm": lambda region: region.paradigm,
    "role": lambda region: region.role,
}


class ColorMap:
    """A stable key -> (colour, hatch) assignment with an "other" fold."""

    def __init__(
        self,
        keys: Iterable[str],
        *,
        theme: Theme | str | None = None,
        max_colors: int = 8,
        by: str = "region",
        other_label: str | None = None,
    ) -> None:
        self.theme = get_theme(theme)
        self.by = by
        if max_colors < 1:
            raise ValueError("max_colors must be >= 1")
        if max_colors > self.theme.max_series:
            raise ValueError(
                f"max_colors={max_colors} exceeds the {self.theme.max_series} "
                f"distinguishable series of the {self.theme.name} theme"
            )
        self.max_colors = max_colors

        ordered: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        self.keys = tuple(ordered)
        self.slots = {key: i for i, key in enumerate(self.keys[:max_colors])}
        self.folded = tuple(self.keys[max_colors:])
        self.other_label = other_label or (
            f"other ({len(self.folded)} {by}s)" if self.folded else OTHER
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        *,
        by: str = "region",
        max_colors: int = 8,
        theme: Theme | str | None = None,
    ) -> "ColorMap":
        """Rank the trace's keys by exclusive time and assign slots in that order.

        ``by`` is ``"region"`` (the default), ``"paradigm"`` (OPENMP / MPI /
        USER / ...) or ``"role"`` (FUNCTION / LOOP / IMPLICIT_BARRIER / ...).
        The two coarse groupings usually fit in the palette without folding,
        which makes them a good first look at a busy trace.
        """
        if by not in _KEY_FUNCS:
            raise ValueError(
                f"unknown grouping {by!r}; expected one of {sorted(_KEY_FUNCS)}"
            )
        key_of = _KEY_FUNCS[by]
        weights: dict[str, float] = {}
        for stat in trace.region_stats():
            key = key_of(stat.region)
            weights[key] = weights.get(key, 0.0) + stat.exclusive
        # Ties break on the name, so the assignment is reproducible.
        ranked = sorted(weights, key=lambda k: (-weights[k], k))
        return cls(ranked, theme=theme, max_colors=max_colors, by=by)

    # ------------------------------------------------------------------
    def key_of(self, region) -> str:
        """The grouping key of a :class:`~otf2viz.model.Region`."""
        return _KEY_FUNCS[self.by](region)

    def slot(self, key: str) -> int:
        """Slot index of ``key``, or ``-1`` if it folds into "other"."""
        return self.slots.get(key, -1)

    def style(self, key: str) -> dict:
        """Matplotlib face colour and hatch for ``key``."""
        index = self.slots.get(key)
        if index is None:
            return {"facecolor": self.theme.other, "hatch": None}
        color, hatch = self.theme.slot(index)
        return {"facecolor": color, "hatch": hatch}

    def other_style(self) -> dict:
        """Style of the "other" fold bucket."""
        return {"facecolor": self.theme.other, "hatch": None}

    def label(self, key: str) -> str:
        """Legend label for ``key`` — the key itself, or the "other" bucket."""
        return key if key in self.slots else self.other_label

    def entries(self, keys: Iterable[str] | None = None) -> list[tuple[str, dict]]:
        """``(label, style)`` pairs in slot order, for building a legend.

        Pass ``keys`` (typically the keys actually drawn) to omit series that
        are not on screen without disturbing anyone else's colour.
        """
        present = None if keys is None else set(keys)
        out = []
        for key in self.keys[: self.max_colors]:
            if present is None or key in present:
                out.append((key, self.style(key)))
        has_folded = bool(self.folded) and (
            present is None or bool(present & set(self.folded))
        )
        if has_folded:
            out.append((self.other_label, self.other_style()))
        return out

    def interval_keys(self, trace: Trace) -> list[str]:
        """Grouping key per interval of ``trace``."""
        per_region = [self.key_of(region) for region in trace.regions]
        return [per_region[i] for i in trace.region]

    def __len__(self) -> int:
        return len(self.keys)

    def __repr__(self) -> str:
        return (
            f"<ColorMap by={self.by!r} theme={self.theme.name!r} "
            f"{len(self.slots)} coloured, {len(self.folded)} folded>"
        )


def resolve(
    colors: "ColorMap | None",
    trace: Trace,
    *,
    by: str = "region",
    max_colors: int = 8,
    theme: Theme | str | None = None,
) -> ColorMap:
    """Use the caller's colour map, or derive one from ``trace``."""
    if colors is not None:
        return colors
    return ColorMap.from_trace(trace, by=by, max_colors=max_colors, theme=theme)
