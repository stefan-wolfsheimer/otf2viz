"""Colours and Matplotlib styling.

Two themes, ``LIGHT`` and ``DARK``. The dark one is *selected* rather than
derived: its categorical hues are the same eight hues re-stepped for a dark
surface, not an automatic inversion of the light values.

The categorical order matters. It is the mechanism that keeps neighbouring
series distinguishable under colour-vision deficiency, so hues are assigned in
this fixed order and never cycled or reshuffled by rank.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import matplotlib as mpl

__all__ = ["Theme", "LIGHT", "DARK", "get_theme", "set_theme", "use"]

_LIGHT_CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

_DARK_CATEGORICAL = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)


@dataclass(frozen=True)
class Theme:
    """Every colour a plot in this package is allowed to use."""

    name: str
    surface: str
    page: str
    primary: str
    secondary: str
    muted: str
    grid: str
    axis: str
    categorical: tuple[str, ...]
    #: Fold-colour for series past the categorical slots, and for "idle".
    other: str
    #: Single hue for magnitude-only charts (bar lengths, one-series areas).
    accent: str
    #: Hatches paired with a repeated hue when a plot needs more than 8 series.
    hatches: tuple[str | None, ...] = field(
        default=(None, "///", "...", "xxx"), repr=False
    )

    @property
    def max_series(self) -> int:
        """Distinguishable series before folding is unavoidable."""
        return len(self.categorical) * len(self.hatches)

    def slot(self, index: int) -> tuple[str, str | None]:
        """Colour and hatch for categorical slot ``index``.

        The first eight slots are plain hues. Beyond that the hues repeat but
        carry a texture, so identity never rests on a hue that is already in
        use elsewhere in the same chart.
        """
        if index < 0:
            raise IndexError("categorical slot index must be >= 0")
        n = len(self.categorical)
        cycle = index // n
        if cycle >= len(self.hatches):
            raise IndexError(
                f"{self.name} theme offers {self.max_series} distinguishable series; "
                "fold the rest into 'Other' or facet the chart"
            )
        return self.categorical[index % n], self.hatches[cycle]

    def rc(self) -> dict:
        """Matplotlib rcParams implementing this theme."""
        return {
            "figure.facecolor": self.page,
            "figure.edgecolor": self.page,
            "savefig.facecolor": self.page,
            "savefig.edgecolor": self.page,
            "axes.facecolor": self.surface,
            "axes.edgecolor": self.axis,
            "axes.labelcolor": self.secondary,
            "axes.titlecolor": self.primary,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "grid.color": self.grid,
            "grid.linewidth": 0.8,
            "text.color": self.primary,
            "xtick.color": self.muted,
            "ytick.color": self.muted,
            "xtick.labelcolor": self.secondary,
            "ytick.labelcolor": self.secondary,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": self.secondary,
            "font.size": 10,
            "font.family": "sans-serif",
            "hatch.linewidth": 0.6,
        }


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    primary="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    categorical=_LIGHT_CATEGORICAL,
    other="#898781",
    accent="#2a78d6",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    primary="#ffffff",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    categorical=_DARK_CATEGORICAL,
    other="#898781",
    accent="#3987e5",
)

_THEMES = {"light": LIGHT, "dark": DARK}
_active = LIGHT


def get_theme(theme: "Theme | str | None" = None) -> Theme:
    """Resolve a theme name, a theme, or ``None`` (the active default)."""
    if theme is None:
        return _active
    if isinstance(theme, Theme):
        return theme
    try:
        return _THEMES[str(theme).lower()]
    except KeyError:
        raise ValueError(
            f"unknown theme {theme!r}; expected 'light', 'dark' or a Theme"
        ) from None


def set_theme(theme: "Theme | str") -> Theme:
    """Set the default theme for every subsequent plot in this session."""
    global _active
    _active = get_theme(theme)
    return _active


@contextlib.contextmanager
def use(theme: "Theme | str | None" = None):
    """Apply a theme's rcParams for the duration of a ``with`` block.

    Only the keys the theme sets are saved and put back — deliberately *not*
    :func:`matplotlib.rc_context`, which snapshots every rcParam on entry and
    restores the lot on exit, discarding unrelated changes made inside the
    block. Jupyter makes that fatal rather than untidy: the kernel sets up its
    matplotlib integration lazily, when the first figure of the session is
    created, and part of that setup is turning ``rcParams["interactive"]`` on.
    That happens inside our block, so a blanket restore switches it straight
    back off, and the inline backend then never flushes another figure — every
    later plot in the session comes out blank until the kernel is restarted.

    >>> import otf2viz
    >>> with otf2viz.theme.use("dark"):
    ...     pass
    """
    resolved = get_theme(theme)
    overrides = resolved.rc()
    saved = {key: mpl.rcParams[key] for key in overrides}
    mpl.rcParams.update(overrides)
    try:
        yield resolved
    finally:
        mpl.rcParams.update(saved)
