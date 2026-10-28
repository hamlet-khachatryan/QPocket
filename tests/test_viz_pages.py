"""Tabs are a nav bar plus CSS: each section is wrapped with the page it belongs to."""
import marimo as mo
import pytest

from qpocket.viz import pages


def test_a_section_is_wrapped_with_its_pages():
    html = pages.on("Exploration", mo.md("hello")).text
    assert html.startswith('<div class="qp-page qp-on-Exploration">') and "hello" in html
    both = pages.on(("Exploration", "Screening"), "guard").text
    assert 'class="qp-page qp-on-Exploration qp-on-Screening"' in both


def test_unknown_page_is_refused():
    with pytest.raises(ValueError, match="Site, Exploration, Screening"):
        pages.on("Docking", "x")


def test_css_hides_every_other_page():
    assert pages.page_css("Site") == "<style>.qp-page:not(.qp-on-Site) { display: none; }</style>"
    assert pages.PAGES == ("Site", "Exploration", "Screening")


def test_a_message_shows_only_while_its_stamp_holds(tmp_path):
    site = tmp_path / "S1"
    (site / "metadata").mkdir(parents=True)
    csv = site / "metadata" / "clustering_metadata.csv"
    csv.write_text("pdb_id,cluster_id\nA,0\n")
    stamp = pages.clustering_stamp(site)
    message = pages.stamped("_A is now the representative._", stamp)
    assert pages.shown(message, pages.clustering_stamp(site)) == "_A is now the representative._"
    csv.write_text("pdb_id,cluster_id\nA,1\n")             # Apply wrote a new clustering
    assert pages.shown(message, pages.clustering_stamp(site)) == ""
    other = tmp_path / "S2"
    assert pages.clustering_stamp(other) != stamp          # another site never shows this site's message
    assert pages.shown("", stamp) == "" and pages.shown(None, stamp) == ""
