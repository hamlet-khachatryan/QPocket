"""One tidy control panel per notebook section."""

from __future__ import annotations

from typing import List, Mapping, Sequence

#: Heading the secondary groups sit behind.
DEFAULT_LABEL = "Display options"


def group_names(groups: Mapping[str, Sequence]) -> List[str]:
    """Groups that have something to show, in the order they were given.

    Args:
        groups: Controls keyed by the question they answer.

    Returns:
        Names of the non-empty groups.
    """
    return [name for name, widgets in groups.items() if list(widgets)]


def panel(primary: Sequence, groups: Mapping[str, Sequence] = None,
          label: str = DEFAULT_LABEL):
    """A row of primary controls, with the rest folded away behind one heading.

    Args:
        primary: Controls worth seeing at all times.
        groups: The rest, keyed by the question they answer.
        label: Heading the groups sit behind.

    Returns:
        A marimo object ready to be the value of a cell.
    """
    # marimo is a viz extra, so it is imported here rather than at module load.
    import marimo as mo

    groups = dict(groups or {})
    rows = []
    if list(primary):
        rows.append(mo.hstack(list(primary), justify="start", gap=1.5, wrap=True))

    present = group_names(groups)
    if present:
        folded = mo.vstack([
            mo.hstack(
                [mo.md(f"<span style='color:#6b6a67;font-size:0.85em'>{name}</span>"),
                 mo.hstack(list(groups[name]), justify="start", gap=1.25, wrap=True)],
                justify="start", gap=1.0, align="center", wrap=True,
            )
            for name in present
        ], gap=0.75)
        rows.append(mo.accordion({label: folded}))

    return mo.vstack(rows, gap=0.6) if rows else mo.md("")


def summary(**facts) -> str:
    """One quiet line of what the controls currently amount to.

    Args:
        **facts: Label to value; a value of None is left out.

    Returns:
        Markdown, or "" when there is nothing to say.
    """
    parts = [f"{name.replace('_', ' ')} **{value}**"
             for name, value in facts.items() if value is not None]
    return " · ".join(parts)
