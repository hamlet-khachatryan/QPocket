"""Failures as values, and composing steps that may fail (pure)."""
from __future__ import annotations

from typing import Any, Callable, Iterable, List, NamedTuple, Tuple


class Failure(NamedTuple):
    """Why one structure dropped out of a run.

    Attributes:
        pdb_id: The structure.
        stage: Step that failed.
        reason: What went wrong.
    """

    pdb_id: str
    stage: str
    reason: str

    def to_json(self) -> dict:
        """Plain form, for a sidecar.

        Returns:
            ``{"failure": {...}}``.
        """
        return {"failure": self._asdict()}

    @classmethod
    def from_json(cls, payload: dict) -> "Failure":
        """Rebuild from :meth:`to_json`.

        Args:
            payload: Output of to_json.

        Returns:
            The failure.
        """
        return cls(**payload["failure"])


def pipe(value: Any, *steps: Callable[[Any], Any]) -> Any:
    """Apply steps in turn, stopping at the first Failure.

    Args:
        value: Input to the first step.
        *steps: One-argument functions.

    Returns:
        The last step's result, or the first Failure.
    """
    for step in steps:
        if isinstance(value, Failure):
            return value
        value = step(value)
    return value


def partition(results: Iterable[Any]) -> Tuple[List[Any], List[Failure]]:
    """Split results into successes and failures, keeping order.

    Args:
        results: Values and Failures.

    Returns:
        (successes, failures).
    """
    ok: List[Any] = []
    failed: List[Failure] = []
    for result in results:
        (failed if isinstance(result, Failure) else ok).append(result)
    return ok, failed


class guarded:
    """``fn`` with any exception turned into a Failure for that item.

    Args:
        stage: Stage name recorded on a Failure.
        fn: One-argument function.
        pdb_id_of: Names the item in a Failure.
    """

    # A class rather than a closure, so it pickles for process pools.
    def __init__(self, stage: str, fn: Callable[[Any], Any], pdb_id_of: Callable[[Any], str]):
        self.stage, self.fn, self.pdb_id_of = stage, fn, pdb_id_of

    def __call__(self, item: Any) -> Any:
        try:
            return self.fn(item)
        except Exception as exc:  # one structure must not end the run
            return Failure(self.pdb_id_of(item), self.stage, repr(exc))
