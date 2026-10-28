"""The explorer's pages (Site, Exploration, Screening): a tab bar, CSS that shows one page, and page logic.

Switching tabs reruns only the CSS cell, so no section recomputes and every control keeps its value.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

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
    """The style block that hides every page section except ``active``'s, and every cell holding only those."""
    # Hiding the sections alone leaves each cell's own box and spacing, which stack up above later pages.
    return (f"<style>.qp-page:not(.qp-on-{active}) {{ display: none; }}\n"
            f".marimo-cell:has(.qp-page):not(:has(.qp-on-{active})) {{ display: none; }}</style>")


def clustering_stamp(site_dir: Path) -> str:
    """What a site's clustering is now: its name and its applied clustering's hash ("none" before one)."""
    from qpocket.io.tables import file_hash

    applied = Path(site_dir) / "metadata" / "clustering_metadata.csv"
    return f"{Path(site_dir).name}:{file_hash(applied) if applied.exists() else 'none'}"


def stamped(text: str, stamp: str) -> Tuple[str, str]:
    """A button's outcome, kept with the state it was about (:func:`shown` drops it once that changes)."""
    return (stamp, text)


def shown(message: Optional[Tuple[str, str]], stamp: str) -> str:
    """The message's text while ``stamp`` still holds, else ""."""
    return message[1] if message and message[0] == stamp else ""
