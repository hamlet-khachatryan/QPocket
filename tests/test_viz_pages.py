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
