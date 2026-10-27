"""The Site page's notebook-free helpers: the 3D overview, the override table, the loss summary."""
import pandas as pd

from qpocket.arms import sites as steps
from qpocket.viz.pages import site as page
from tests.test_site_steps import FAR, NEAR, _finished_run


def _boxes(fig):
    return {t.name: t for t in fig.data if t.name in (NEAR, FAR)}


def test_overview_has_one_box_per_site_and_the_selected_is_thick(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    fig = page.overview_figure(root, steps.site_table(root), selected=FAR)
    boxes = _boxes(fig)
    assert set(boxes) == {NEAR, FAR}
    assert boxes[FAR].line.width == 4 and boxes[NEAR].line.width == 1.5
    assert boxes[NEAR].line.color != boxes[FAR].line.color
    assert any(t.name == "protein" for t in fig.data)


def test_overview_caps_ligands(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    fig = page.overview_figure(root, steps.site_table(root), max_ligands=1)
    shown = {h.split(" · ")[0] for t in fig.data if t.name == "ligands" for h in t.hovertext}
    assert len(shown) == 1
    assert "1 of 3 ligands" in fig.layout.title.text


def test_loss_markdown_says_nothing_is_lost():
    assert page.loss_markdown({}) == "Nothing is lost."
    assert page.loss_markdown({"S1-Y1": []}) == "Nothing is lost."
    text = page.loss_markdown({"S2-G127": ["clustering", "featurised pockets (412)"], "S1-Y1": []})
    assert text == "- **S2-G127**: clustering, featurised pockets (412)"


def test_override_table_shows_unmatched(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    assert list(page.override_table(root).columns) == ["#", "operation", "sites", "status"]
    steps.apply_site_overrides(root, [{"op": "drop", "cavities": [["X", 9]], "label": "gone"}])
    table = page.override_table(root)
    assert table.to_dict("records") == [{"#": 0, "operation": "drop", "sites": "gone", "status": "unmatched"}]
