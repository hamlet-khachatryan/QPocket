"""run_map: order, worker resolution, per-worker logs, serial reference path."""
import os

import pytest

from qpocket.pipelines import parallel


def square(x):
    return x * x


@pytest.mark.parametrize("requested,expected", [
    (1, 1), (4, 4), (-1, os.cpu_count()), (0, max(1, (os.cpu_count() or 2) // 2)),
])
def test_resolve_workers(requested, expected):
    assert parallel.resolve_workers(requested) == expected


@pytest.mark.parametrize("workers", [1, 2])
def test_results_keep_input_order(workers):
    assert parallel.run_map(square, list(range(10)), workers=workers) == [x * x for x in range(10)]


def test_serial_path_uses_no_joblib(monkeypatch):
    """workers=1 must be a plain loop: the reference behaviour tests compare against."""
    def explode(*a, **k):
        raise AssertionError("joblib used for a serial run")

    monkeypatch.setattr(parallel, "Parallel", explode)
    assert parallel.run_map(square, [1, 2, 3], workers=1) == [1, 4, 9]


def test_each_item_logs_to_its_own_file(tmp_path):
    from eliot import log_message

    def work(name):
        log_message(message_type="did_work", item=name)
        return name

    parallel.run_map(work, ["A1", "B2"], workers=2, backend="thread",
                     stage="featurize", log_dir=tmp_path)
    written = sorted(p.name for p in (tmp_path / "featurize").iterdir())
    assert written == ["A1.log", "B2.log"]
    assert "did_work" in (tmp_path / "featurize" / "A1.log").read_text(encoding="utf-8")
