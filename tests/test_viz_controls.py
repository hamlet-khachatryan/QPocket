"""Control panels: a primary row, and the rest grouped out of the way."""
import pytest

pytest.importorskip("marimo", reason="needs the viz extra")

import marimo as mo  # noqa: E402

from qpocket.viz import controls  # noqa: E402


def _widgets(n):
    return [mo.ui.slider(start=0, stop=1, label=f"w{i}") for i in range(n)]


def test_a_panel_renders_with_only_primary_controls():
    assert controls.panel(primary=_widgets(2)) is not None


def test_groups_are_offered_under_one_heading():
    panel = controls.panel(primary=_widgets(1),
                           groups={"Colour": _widgets(2), "Layout": _widgets(3)})
    assert panel is not None


def test_empty_groups_are_dropped():
    """A group whose controls do not apply should not leave an empty heading."""
    assert controls.group_names({"Colour": _widgets(1), "Layout": []}) == ["Colour"]


def test_group_order_is_kept():
    names = controls.group_names({"Colour": _widgets(1), "Layout": _widgets(1),
                                  "Protein": _widgets(1)})
    assert names == ["Colour", "Layout", "Protein"]


def test_a_panel_with_nothing_to_show_is_still_renderable():
    assert controls.panel(primary=[], groups={}) is not None


def test_the_heading_can_be_named():
    assert controls.panel(primary=_widgets(1), groups={"Colour": _widgets(1)},
                          label="Viewer options") is not None
