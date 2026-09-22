"""Running per-structure work on many cores, or in a plain loop."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, TypeVar

from eliot import log_message
from joblib import Parallel, delayed

T = TypeVar("T")
R = TypeVar("R")

#: Worker logs live under <run>/logs/<stage>/<label>.log.
WORKER_LOG_DIR = "logs"


def resolve_workers(workers: int) -> int:
    """Turn a configured worker count into a concrete one.

    Args:
        workers: 0 for half the cores, -1 for all, or an exact count.

    Returns:
        At least 1.
    """
    cores = os.cpu_count() or 1
    if workers == 0:
        return max(1, cores // 2)
    if workers < 0:
        return cores
    return max(1, workers)


def _in_worker(fn, item, stage: str, log_dir: Optional[Path], label: str):
    """Run one item, sending its eliot messages to its own file."""
    if log_dir is not None:
        from eliot import to_file

        target = Path(log_dir) / stage
        target.mkdir(parents=True, exist_ok=True)
        to_file((target / f"{label}.log").open("w", encoding="utf-8"))
    return fn(item)


def run_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int,
    backend: str = "process",
    stage: str = "",
    log_dir: Optional[Path] = None,
    label: Callable[[T], str] = str,
) -> List[R]:
    """Apply fn to every item, in input order, on workers processes or threads.

    Args:
        fn: Work for one item; must be picklable by cloudpickle.
        items: The items.
        workers: Resolved by resolve_workers.
        backend: "process" or "thread".
        stage: Name used for the worker log directory.
        log_dir: Run log root; None keeps the parent's logging.
        label: Item to log-file name.

    Returns:
        One result per item, in order.
    """
    items = list(items)
    n = resolve_workers(workers)
    started = time.time()
    if n == 1 or len(items) <= 1:
        results = [fn(item) for item in items]
    else:
        results = Parallel(n_jobs=n, backend="loky" if backend == "process" else "threading")(
            delayed(_in_worker)(fn, item, stage or "stage", log_dir, label(item)) for item in items
        )
    log_message(message_type="parallel_map", stage=stage, n_items=len(items),
                workers=n, seconds=round(time.time() - started, 2))
    return list(results)
