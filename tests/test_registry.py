"""The generic plug-in registry: registration, lookup, errors, lazy loading."""

import pytest

from qpocket.core.registry import Registry


def _registry():
    return Registry("widgets")


def test_unknown_name_lists_the_available_ones():
    reg = _registry()
    reg.register("a")(lambda: 1)
    reg.register("b")(lambda: 2)
    with pytest.raises(ValueError, match=r"unknown widgets 'z'; available: a, b"):
        reg.get("z")


def test_concurrent_first_lookups_all_see_the_loaded_plugins():
    """A thread arriving while another loads the built-ins must wait, not find it empty."""
    import threading
    import time

    reg = Registry("slow")

    def loader():
        time.sleep(0.2)
        reg.register("late")(lambda: 1)

    reg._loader = loader
    seen, start = [], threading.Barrier(8)

    def lookup():
        start.wait()
        seen.append("late" in reg.available())

    threads = [threading.Thread(target=lookup) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == [True] * 8
