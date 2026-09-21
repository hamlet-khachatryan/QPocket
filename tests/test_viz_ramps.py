"""Tests for qpocket_pipeline.viz.ramps.

Two things matter here. A signed feature must read warm-for-negative whatever
values happen to be in view, and the legend must be coloured by the same maths
the viewer uses, or it will describe colours that are not on screen.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpocket_pipeline.viz.ramps import (
    FEATURE_RAMP,
    PERCEPTUAL_SCHEME,
    RAMP_TABLES,
    SCHEMES,
    colour_domain,
    feature_ramp,
    gradient_spec,
    legend_gradient,
    linear_hex,
    ramp_stops,
    scale_to_percent,
)

SIGNED_FEATURES = ("apbs", "cp", "lp")
POSITIVE_FEATURES = ("h_donor", "h_acceptor", "aromatic")


def rgb(hex_colour):
    return tuple(int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))


def warmth(hex_colour):
    """Positive when a colour is red-leaning, negative when blue-leaning."""
    r, _, b = rgb(hex_colour)
    return r - b


def chroma(hex_colour):
    channels = rgb(hex_colour)
    return max(channels) - min(channels)


# ---------------------------------------------------------------------------
# feature_ramp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("feature", SIGNED_FEATURES)
def test_signed_features_are_diverging(feature):
    assert feature_ramp(feature).kind == "diverging"


@pytest.mark.parametrize("feature", POSITIVE_FEATURES)
def test_strictly_positive_features_are_sequential(feature):
    assert feature_ramp(feature).kind == "sequential"


def test_unknown_feature_falls_back_to_a_sequential_ramp():
    spec = feature_ramp("something_new")
    assert spec.kind == "sequential"
    assert spec.scale in RAMP_TABLES


@pytest.mark.parametrize("feature", list(FEATURE_RAMP))
@pytest.mark.parametrize("scheme", SCHEMES)
def test_every_scheme_resolves_to_a_known_table(feature, scheme):
    assert feature_ramp(feature, scheme).scale in RAMP_TABLES


@pytest.mark.parametrize("feature", list(FEATURE_RAMP))
def test_scheme_changes_the_ramp_but_never_the_semantics(feature):
    plain, perceptual = feature_ramp(feature), feature_ramp(feature, PERCEPTUAL_SCHEME)
    assert plain.kind == perceptual.kind
    assert plain.ends == perceptual.ends


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("feature", ("apbs", "cp"))
def test_negative_potential_reads_warm_in_every_scheme(scheme, feature):
    """The convention APBS and the molecular viewers established."""
    stops = ramp_stops(feature_ramp(feature, scheme).scale, zero_frac=0.5)
    assert warmth(stops[0]) > 40, stops[0]
    assert warmth(stops[-1]) < -40, stops[-1]


def test_lipophilicity_runs_teal_to_brown():
    stops = ramp_stops(feature_ramp("lp").scale, zero_frac=0.5)
    r_lo, g_lo, b_lo = rgb(stops[0])
    r_hi, _, b_hi = rgb(stops[-1])
    assert g_lo > r_lo and b_lo > r_lo  # hydrophilic end is teal
    assert r_hi > b_hi  # lipophilic end is brown


# ---------------------------------------------------------------------------
# colour_domain
# ---------------------------------------------------------------------------


def test_two_sided_signed_data_gets_a_symmetric_domain():
    """Otherwise +5 would look as strongly positive as -20 looks negative."""
    domain = colour_domain([-20.0, -1.0, 5.0], clip_pct=0, kind="diverging")
    assert domain.lo == pytest.approx(-20.0)
    assert domain.hi == pytest.approx(20.0)
    assert domain.zero_frac == pytest.approx(0.5)


def test_all_negative_data_keeps_zero_in_the_domain():
    """The case that used to be misread as a magnitude ramp."""
    domain = colour_domain([-20.0, -10.0, -3.0], clip_pct=0, kind="diverging")
    assert domain.lo == pytest.approx(-20.0)
    assert domain.hi == pytest.approx(0.0)
    assert domain.zero_frac == pytest.approx(1.0)


def test_all_negative_data_never_produces_a_cool_colour():
    domain = colour_domain([-20.0, -10.0, -3.0], clip_pct=0, kind="diverging")
    stops = ramp_stops("RdBu", domain.zero_frac)
    assert min(warmth(c) for c in stops) >= -2


def test_all_positive_signed_data_uses_the_upper_arm_only():
    domain = colour_domain([1.0, 5.0, 9.0], clip_pct=0, kind="diverging")
    assert domain.lo == pytest.approx(0.0)
    assert domain.zero_frac == pytest.approx(0.0)
    stops = ramp_stops("RdBu", domain.zero_frac)
    assert max(warmth(c) for c in stops) <= 2


def test_sequential_domain_tracks_the_data_and_has_no_zero_point():
    domain = colour_domain([0.2, 0.5, 0.9], clip_pct=0, kind="sequential")
    assert (domain.lo, domain.hi) == pytest.approx((0.2, 0.9))
    assert domain.zero_frac is None


def test_clipping_trims_the_heavy_tail():
    """One run's `cp` peaks at 1679 against a 99th percentile of 66."""
    values = np.concatenate([np.linspace(-50, 50, 999), [1679.0]])
    wide = colour_domain(values, clip_pct=0, kind="diverging")
    clipped = colour_domain(values, clip_pct=2, kind="diverging")
    assert wide.hi > 1000
    assert clipped.hi < 60
    assert clipped.zero_frac == pytest.approx(0.5)


def test_constant_data_gives_a_usable_domain():
    for kind in ("diverging", "sequential"):
        domain = colour_domain([3.0, 3.0, 3.0], clip_pct=0, kind=kind)
        assert domain.hi > domain.lo


def test_empty_data_does_not_raise():
    assert colour_domain([], clip_pct=2, kind="diverging").hi > 0


def test_non_finite_values_are_ignored():
    domain = colour_domain([-1.0, 1.0, np.nan, np.inf], clip_pct=0, kind="diverging")
    assert domain.hi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# scale_to_percent
# ---------------------------------------------------------------------------


def test_scaling_maps_the_domain_onto_0_100():
    domain = colour_domain([-4.0, 4.0], clip_pct=0, kind="diverging")
    scaled = scale_to_percent([-4.0, 0.0, 4.0], domain)
    assert scaled == pytest.approx([0.0, 50.0, 100.0])


def test_scaling_clips_values_outside_the_clipped_domain():
    domain = colour_domain([-1.0, 1.0], clip_pct=0, kind="diverging")
    assert scale_to_percent([-99.0, 99.0], domain) == pytest.approx([0.0, 100.0])


# ---------------------------------------------------------------------------
# ramp_stops: zero pinning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zero_frac", [0.0, 0.15, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_neutral_midpoint_lands_where_zero_is(zero_frac):
    stops = ramp_stops("RdBu", zero_frac)
    least_chromatic = min(range(len(stops)), key=lambda i: chroma(stops[i]))
    assert abs(least_chromatic / (len(stops) - 1) - zero_frac) <= 0.05


@pytest.mark.parametrize("n", [4, 12, 32, 64])
def test_stop_count_is_respected(n):
    assert len(ramp_stops("RdBu", 0.5, n=n)) == n


def test_sequential_ramp_starts_off_white():
    """A near-white first stop disappears against the page."""
    for feature in POSITIVE_FEATURES:
        first = ramp_stops(feature_ramp(feature).scale)[0]
        assert sum(rgb(first)) / 3 < 235, (feature, first)


def test_sequential_ramp_ends_at_the_darkest_stop():
    stops = ramp_stops("Blues")
    assert sum(rgb(stops[-1])) < sum(rgb(stops[0]))


def test_unknown_ramp_name_is_reported():
    with pytest.raises(KeyError, match="unknown ramp"):
        ramp_stops("NotAScale")


# ---------------------------------------------------------------------------
# legend agreement
# ---------------------------------------------------------------------------


def test_legend_endpoints_match_the_stops_handed_to_the_viewer():
    stops = ramp_stops("RdBu", 0.5)
    assert linear_hex(0.0, stops) == stops[0]
    assert linear_hex(100.0, stops) == stops[-1]


def test_legend_is_monotonic_through_the_ramp():
    stops = ramp_stops("Blues")
    brightness = [sum(rgb(c)) for c in legend_gradient(stops, n=40)]
    assert brightness[0] > brightness[-1]


def test_linear_hex_reproduces_the_flat_last_bin():
    """CustomLinear's last bin has no successor, so it renders flat."""
    stops = ramp_stops("RdBu", 0.5, n=8)
    top_of_penultimate = 100.0 * 7 / 8
    assert linear_hex(top_of_penultimate, stops) == stops[-1]
    assert linear_hex(100.0, stops) == stops[-1]


def test_linear_hex_clamps_outside_the_range():
    stops = ramp_stops("Blues")
    assert linear_hex(-50.0, stops) == stops[0]
    assert linear_hex(150.0, stops) == stops[-1]


# ---------------------------------------------------------------------------
# gradient_spec
# ---------------------------------------------------------------------------


def test_gradient_spec_is_a_3dmol_colorscheme():
    stops = ramp_stops("RdBu", 0.5)
    spec = gradient_spec(stops)
    assert spec["prop"] == "b"
    assert (spec["min"], spec["max"]) == (0, 100)
    assert spec["gradient"].startswith("linear_#")
    # 3Dmol splits this string on "_", so hex colours must not contain one.
    assert spec["gradient"].split("_")[1:] == list(stops)


# ---------------------------------------------------------------------------
# the embedded tables must still match plotly's definitions
# ---------------------------------------------------------------------------


def test_tables_match_plotly():
    """Guards the generated RAMP_TABLES against upstream drift."""
    plotly_colors = pytest.importorskip("plotly.colors")
    from plotly.colors import sample_colorscale

    for name, table in RAMP_TABLES.items():
        scale = None
        for module in (plotly_colors.diverging, plotly_colors.sequential, plotly_colors.cyclical):
            scale = getattr(module, name, None)
            if scale:
                break
        assert scale, f"{name} is no longer a plotly scale"
        wanted = sample_colorscale(
            scale, [i / (len(table) - 1) for i in range(len(table))], colortype="rgb"
        )
        for got, want in zip(table, wanted):
            want_rgb = tuple(int(round(float(v))) for v in want[want.index("(") + 1 : -1].split(","))
            assert rgb(got) == pytest.approx(want_rgb, abs=1), (name, got, want)
