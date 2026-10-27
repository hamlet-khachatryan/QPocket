"""One figure style, and cluster colours that agree across every view."""
import colorsys

from qpocket.viz.style import GREY, HUES, TEMPLATE, apply, cluster_colours


def _lightness(hex_colour):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[1]


def _hue(hex_colour):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return round(colorsys.rgb_to_hls(r, g, b)[0], 2)


def test_largest_cluster_gets_the_first_hue():
    shape = [1] * 3 + [0] * 8 + [2] * 5
    colours = cluster_colours(shape, None, min_size=2)
    assert colours[(0, 0)] == HUES[0] and colours[(2, 0)] == HUES[1] and colours[(1, 0)] == HUES[2]


def test_subclusters_are_tints_of_their_shape_hue():
    shape = [0] * 9
    chem = [0] * 5 + [1] * 3 + [2]
    colours = cluster_colours(shape, chem, min_size=1)
    tints = [colours[(0, c)] for c in (0, 1, 2)]
    assert len({_hue(t) for t in tints}) == 1
    assert _lightness(tints[0]) < _lightness(tints[1]) < _lightness(tints[2])


def test_small_clusters_are_grey():
    colours = cluster_colours([0] * 6 + [1] * 3, None, min_size=4)
    assert colours[(1, 0)] == GREY and colours[(0, 0)] != GREY


def test_colours_cycle_stably():
    shape = [i for i in range(14) for _ in range(20 - i)]
    a, b = cluster_colours(shape, None, min_size=1), cluster_colours(list(shape), None, min_size=1)
    assert a == b and len(HUES) >= 10
    assert a[(10, 0)] == HUES[0]                       # the 11th cluster reuses the first hue


def test_template_has_no_gridlines():
    assert TEMPLATE.layout.xaxis.showgrid is False and TEMPLATE.layout.yaxis.showgrid is False
    assert TEMPLATE.layout.plot_bgcolor == "white"


def test_apply_sets_the_template():
    import plotly.graph_objects as go

    fig = apply(go.Figure())
    assert fig.layout.template.layout.xaxis.showgrid is False
