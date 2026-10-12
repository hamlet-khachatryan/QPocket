"""Failures are values: pipe stops at one, partition splits them out."""
import pickle
from functools import partial

from qpocket.core.result import Failure, guarded, partition, pipe


def test_pipe_threads_a_value_through_every_step():
    assert pipe(1, lambda x: x + 1, lambda x: x * 10) == 20


def test_pipe_stops_at_the_first_failure():
    seen = []
    failure = Failure("1ABC", "align", "no overlap")
    out = pipe(1, lambda x: failure, lambda x: seen.append(x))
    assert out is failure and seen == []


def test_partition_keeps_order():
    a, b = Failure("A", "s", "r"), Failure("B", "s", "r")
    assert partition([1, a, 2, b]) == ([1, 2], [a, b])


def test_guarded_turns_an_exception_into_a_failure():
    def boom(item):
        raise RuntimeError("bad file")

    out = guarded("structure", boom, str)("7KG6")
    assert out == Failure("7KG6", "structure", "RuntimeError('bad file')")


def test_guarded_passes_results_through():
    assert guarded("s", lambda x: x * 2, str)(4) == 8


def test_failure_round_trips_through_json():
    f = Failure("7KG6", "clean", "empty")
    assert Failure.from_json(f.to_json()) == f


def test_guarded_partials_pickle():
    pickle.dumps(guarded("s", partial(max, 3), str))
