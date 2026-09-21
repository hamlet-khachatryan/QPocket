"""Colour ramps for pocket features, and the maths a viewer legend must mirror."""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np


class RampSpec(NamedTuple):
    """How one feature is coloured.

    Attributes:
        kind: ``"diverging"`` for signed features, ``"sequential"`` otherwise.
        scale: Key into :data:`RAMP_TABLES`.
        ends: Labels for the low and high ends, for the legend.
    """

    kind: str
    scale: str
    ends: Tuple[str, str]


class ColourDomain(NamedTuple):
    """The value range a ramp spans.

    Attributes:
        lo: Value at the start of the ramp.
        hi: Value at the end of the ramp.
        zero_frac: Where zero falls in ``[lo, hi]``, or ``None`` when the feature is unsigned and zero carries no special meaning.
    """

    lo: float
    hi: float
    zero_frac: Optional[float]


#: Number of stops handed to the viewer. More stops shrink the flat last bin.
N_STOPS = 32

#: Sequential ramps start this far along, keeping the palest step off a white page.
SEQ_FLOOR = 0.15

FEATURE_RAMP: Dict[str, RampSpec] = {
    "apbs": RampSpec("diverging", "RdBu", ("negative", "positive")),
    "cp": RampSpec("diverging", "RdBu", ("negative", "positive")),
    "lp": RampSpec("diverging", "BrBG_r", ("hydrophilic", "lipophilic")),
    "h_donor": RampSpec("sequential", "Blues", ("weak", "strong")),
    "h_acceptor": RampSpec("sequential", "Reds", ("weak", "strong")),
    "aromatic": RampSpec("sequential", "YlOrBr", ("weak", "strong")),
}

#: Ramp for a feature with no entry above.
DEFAULT_RAMP = RampSpec("sequential", "Viridis", ("low", "high"))

#: Perceptually uniform, colour-vision-deficiency-safe alternatives. Semantics
#: survive (negative still reads warm); per-feature hue identity does not, since
#: every diverging feature then shares one ramp.
PERCEPTUAL_RAMP: Dict[str, str] = {"diverging": "balance_r", "sequential": "Viridis"}

#: Name of the scheme that swaps in :data:`PERCEPTUAL_RAMP`.
PERCEPTUAL_SCHEME = "perceptual (CVD-safe)"

#: Scheme names, in the order a UI should offer them.
SCHEMES: Tuple[str, ...] = ("feature-aware", PERCEPTUAL_SCHEME)


# Dense stop tables sampled from plotly's colour-scale definitions, so that
# this module needs only numpy at runtime. Regenerate with:
#
#     from plotly.colors import sample_colorscale
#     sample_colorscale(<scale>, [i / 32 for i in range(33)], colortype="rgb")
#
# tests/test_viz_ramps.py checks these against plotly whenever plotly is
# installed, so drift cannot go unnoticed.
RAMP_TABLES: dict[str, tuple[str, ...]] = {
    # ColorBrewer RdBu (red low, blue high)
    "RdBu": (
        "#67001f", "#7e0823", "#960f27", "#ad162a", "#bb2a34", "#c6403e",
        "#d25749", "#dc6d57", "#e58267", "#ee9878", "#f5ac8b", "#f8bda0",
        "#fbcdb6", "#fdddca", "#fbe5d9", "#f9eee8", "#f7f7f7", "#ebf1f5",
        "#dfecf3", "#d3e6f0", "#c1ddec", "#aed3e6", "#9ac9e0", "#83bcd9",
        "#6bacd0", "#529cc8", "#3f8dc0", "#347fb9", "#2a71b2", "#1f63a7",
        "#175290", "#0e4178", "#053061",
    ),
    # ColorBrewer BrBG reversed (teal low, brown high)
    "BrBG_r": (
        "#003c30", "#00493e", "#01564d", "#01635b", "#0e726a", "#1e827a",
        "#2e9189", "#43a198", "#5ab2a8", "#72c3b8", "#89d1c5", "#9fdad1",
        "#b5e3dc", "#caebe6", "#d8eeeb", "#e7f2f0", "#f5f5f5", "#f5f1e5",
        "#f6edd6", "#f6e9c6", "#f0dfb2", "#e9d39c", "#e2c786", "#d9b66e",
        "#cfa255", "#c58d3c", "#b97b29", "#a96c1e", "#995d13", "#894f0a",
        "#774508", "#663a07", "#543005",
    ),
    # ColorBrewer Blues
    "Blues": (
        "#f7fbff", "#f1f7fd", "#ebf3fb", "#e4eff9", "#deebf7", "#d8e7f5",
        "#d2e3f3", "#ccdff1", "#c6dbef", "#bcd7ec", "#b2d2e8", "#a8cee4",
        "#9ecae1", "#91c3de", "#84bcdc", "#78b5d9", "#6baed6", "#61a7d2",
        "#56a0ce", "#4c99ca", "#4292c6", "#3a8ac2", "#3282be", "#2979b9",
        "#2171b5", "#1b69af", "#1461a9", "#0e59a2", "#08519c", "#084990",
        "#084084", "#083877", "#08306b",
    ),
    # ColorBrewer Reds
    "Reds": (
        "#fff5f0", "#fff0e8", "#feebe1", "#fee5da", "#fee0d2", "#fed7c6",
        "#fdceba", "#fcc4ad", "#fcbba1", "#fcb195", "#fca68a", "#fc9c7e",
        "#fc9272", "#fc8868", "#fc7e5e", "#fb7454", "#fb6a4a", "#f85e42",
        "#f5523b", "#f24734", "#ef3b2c", "#e63228", "#dd2a24", "#d42121",
        "#cb181d", "#c2161b", "#b81419", "#ae1117", "#a50f15", "#960b13",
        "#860811", "#76040f", "#67000d",
    ),
    # ColorBrewer YlOrBr
    "YlOrBr": (
        "#ffffe5", "#fffddb", "#fffbd0", "#fff9c6", "#fff7bc", "#fff2b1",
        "#feeda6", "#fee89c", "#fee391", "#fedb80", "#fed370", "#fecc60",
        "#fec44f", "#feb946", "#feae3c", "#fea433", "#fe9929", "#fa8f24",
        "#f5841e", "#f07a19", "#ec7014", "#e46710", "#dc5e0b", "#d45506",
        "#cc4c02", "#bf4602", "#b24003", "#a63a04", "#993404", "#8c3004",
        "#802c05", "#732906", "#662506",
    ),
    # cmocean balance reversed (red low, blue high)
    "balance_r": (
        "#3c0911", "#510b19", "#650c21", "#7a0f28", "#8c1926", "#9d2325",
        "#ae2f26", "#b64333", "#be563f", "#c66a4e", "#cb7e64", "#d1917b",
        "#d7a491", "#ddb6a8", "#e3c8bf", "#e3d3ce", "#dcd5d3", "#d5d7d9",
        "#c5d1d6", "#aec5ce", "#96b9c5", "#7eadc0", "#65a0bd", "#4c93ba",
        "#3786ba", "#2377bb", "#0f69bd", "#135ab0", "#1e4ba0", "#283b90",
        "#233177", "#1d265c", "#171c42",
    ),
    # matplotlib viridis
    "Viridis": (
        "#440154", "#450c5e", "#461768", "#472272", "#472c7a", "#44357f",
        "#413f84", "#3e4888", "#3b518a", "#37598c", "#33628d", "#306a8e",
        "#2d728e", "#2a798e", "#27808e", "#24888d", "#22908c", "#21988a",
        "#20a088", "#27a784", "#2dae7f", "#33b57a", "#40bb73", "#50c26a",
        "#60c860", "#70cf57", "#84d34a", "#98d83d", "#acdc31", "#c0df2a",
        "#d5e228", "#e9e427", "#fde725",
    ),
}


def feature_ramp(feature: str, scheme: str = "feature-aware") -> RampSpec:
    """Pick the ramp for a feature under a colour scheme.

    Args:
        feature: Feature column name, e.g. ``"apbs"``.
        scheme: Either ``"feature-aware"`` or :data:`PERCEPTUAL_SCHEME`.

    Returns:
        The ramp to use.
    """
    spec = FEATURE_RAMP.get(feature, DEFAULT_RAMP)
    if scheme == PERCEPTUAL_SCHEME:
        return spec._replace(scale=PERCEPTUAL_RAMP[spec.kind])
    return spec


def colour_domain(
    values: Sequence[float], clip_pct: float, kind: str
) -> ColourDomain:
    """Value range for a ramp, with zero handled the way the feature needs.

    Args:
        values: Every value the ramp has to cover.
        clip_pct: Percentile trimmed from each end, ``0`` for none.
        kind: ``"diverging"`` or ``"sequential"``.

    Returns:
        The domain, and where zero sits inside it.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return ColourDomain(0.0, 1.0, 0.5 if kind == "diverging" else None)

    lo = float(np.percentile(arr, clip_pct))
    hi = float(np.percentile(arr, 100.0 - clip_pct))

    if kind == "diverging":
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        if lo < 0.0 < hi:
            limit = max(-lo, hi)
            lo, hi = -limit, limit
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        return ColourDomain(lo, hi, (0.0 - lo) / (hi - lo))

    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    return ColourDomain(lo, hi, None)


def scale_to_percent(values: Sequence[float], domain: ColourDomain) -> np.ndarray:
    """Rescale values onto ``0-100`` for the B-factor column.

    Args:
        values: Raw feature values.
        domain: Range they should be mapped onto.

    Returns:
        Values clipped into ``[0, 100]``.
    """
    arr = np.asarray(values, dtype=float)
    return np.clip(100.0 * (arr - domain.lo) / (domain.hi - domain.lo), 0.0, 100.0)


def _sample_table(name: str, positions: np.ndarray) -> Tuple[str, ...]:
    """Interpolate a stop table at fractional positions in ``[0, 1]``."""
    try:
        table = RAMP_TABLES[name]
    except KeyError:
        raise KeyError(f"unknown ramp {name!r}; have {sorted(RAMP_TABLES)}") from None
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in table], dtype=float)
    idx = np.clip(np.asarray(positions, dtype=float), 0.0, 1.0) * (len(table) - 1)
    lower = np.floor(idx).astype(int)
    upper = np.minimum(lower + 1, len(table) - 1)
    frac = (idx - lower)[:, None]
    mixed = rgb[lower] * (1.0 - frac) + rgb[upper] * frac
    return tuple("#" + "".join(f"{int(round(v)):02x}" for v in row) for row in mixed)


def ramp_stops(
    scale: str, zero_frac: Optional[float] = None, n: int = N_STOPS
) -> Tuple[str, ...]:
    """Hex stops for a viewer gradient.

    Args:
        scale: Key into :data:`RAMP_TABLES`.
        zero_frac: Where zero falls in the domain, which is where the ramp's neutral midpoint is pinned.
        n: Number of stops.

    Returns:
        ``n`` hex colours, low value first.
    """
    t = np.linspace(0.0, 1.0, n)
    if zero_frac is None:
        u = SEQ_FLOOR + (1.0 - SEQ_FLOOR) * t
    else:
        z = float(np.clip(zero_frac, 0.0, 1.0))
        if z <= 0.0:  # nothing negative: upper arm only
            u = 0.5 + 0.5 * t
        elif z >= 1.0:  # nothing positive: lower arm only
            u = 0.5 * t
        else:
            u = np.where(t <= z, 0.5 * t / z, 0.5 + 0.5 * (t - z) / (1.0 - z))
    return _sample_table(scale, np.clip(u, 0.0, 1.0))


def gradient_spec(stops: Sequence[str]) -> Dict:
    """3Dmol style ``colorscheme`` that colours by the B-factor column.

    Args:
        stops: Hex colours from :func:`ramp_stops`.

    Returns:
        A ``colorscheme`` dictionary for a 3Dmol style spec.
    """
    return {"prop": "b", "gradient": "linear_" + "_".join(stops), "min": 0, "max": 100}


def linear_hex(
    value: float, stops: Sequence[str], lo: float = 0.0, hi: float = 100.0
) -> str:
    """Colour a value the way 3Dmol's ``CustomLinear`` gradient would.

    Args:
        value: Value to colour.
        stops: The same stops handed to the viewer.
        lo: Low end of the scaled range.
        hi: High end of the scaled range.

    Returns:
        A hex colour.
    """
    value = min(max(value, lo), hi)
    n = len(stops)
    step = (hi - lo) / n
    i = min(int((value - lo) / step), n - 1) if step else 0
    j = min(i + 1, n - 1)
    frac = ((value - lo) - i * step) / step if step else 0.0
    c1 = tuple(int(stops[i][k:k + 2], 16) for k in (1, 3, 5))
    c2 = tuple(int(stops[j][k:k + 2], 16) for k in (1, 3, 5))
    return "#" + "".join(f"{int(a + frac * (b - a)):02x}" for a, b in zip(c1, c2))


def legend_gradient(stops: Sequence[str], n: int = 48) -> Tuple[str, ...]:
    """Sample a CSS gradient from the stops the viewer is using.

    Args:
        stops: Hex colours from :func:`ramp_stops`.
        n: Number of samples across the bar.

    Returns:
        Hex colours, low value first.
    """
    return tuple(linear_hex(v, stops) for v in np.linspace(0.0, 100.0, n))
