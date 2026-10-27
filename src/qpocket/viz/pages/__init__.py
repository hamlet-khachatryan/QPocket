"""The explorer's pages (Site, Exploration, Screening): a tab bar, CSS that shows one page, and page logic.

Switching tabs reruns only the CSS cell, so no section recomputes and every control keeps its value.
"""
from __future__ import annotations

from typing import Iterable, Union

#: Tab labels, in order.
PAGES = ("Site", "Exploration", "Screening")


def on(page: Union[str, Iterable[str]], content):
    """Wrap a cell's output so it shows only on ``page`` (one label, or several).

    Args:
        page: A label from PAGES, or several.
        content: Anything marimo can show.

    Returns:
        An ``mo.Html`` block carrying the pages' classes.

    Raises:
        ValueError: On a label not in PAGES.
    """
    import marimo as mo

    names = [page] if isinstance(page, str) else list(page)
    unknown = [p for p in names if p not in PAGES]
    if unknown:
        raise ValueError(f"unknown page {', '.join(unknown)}; known: {', '.join(PAGES)}")
    classes = " ".join(f"qp-on-{p}" for p in names)
    return mo.Html(f'<div class="qp-page {classes}">{mo.as_html(content).text}</div>')


def page_css(active: str) -> str:
    """The style block that hides every page section except ``active``'s."""
    return f"<style>.qp-page:not(.qp-on-{active}) {{ display: none; }}</style>"
