"""The generic plug-in registry: registration, lookup, errors, lazy loading."""
import dataclasses
import subprocess
import sys

import pytest

from qpocket.core.registry import NoParams, Plugin, Registry, Resources


def _registry():
    return Registry("widgets")


def test_register_and_get():
    reg = _registry()

    @reg.register("double", resources=Resources(cpus=2), tools=("echo",))
    def double(x):
        return 2 * x

    plugin = reg.get("double")
    assert isinstance(plugin, Plugin)
    assert plugin.fn(3) == 6 and plugin.params is NoParams
    assert plugin.resources.cpus == 2 and plugin.tools == ("echo",)
    assert reg.available() == ("double",)


def test_register_returns_the_function_unchanged():
    reg = _registry()

    def f():
        return 1

    assert reg.register("f")(f) is f


def test_unknown_name_lists_the_available_ones():
    reg = _registry()
    reg.register("a")(lambda: 1)
    reg.register("b")(lambda: 2)
    with pytest.raises(ValueError, match=r"unknown widgets 'z'; available: a, b"):
        reg.get("z")


def test_duplicate_registration_is_an_error():
    reg = _registry()
    reg.register("a")(lambda: 1)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("a")(lambda: 2)


def test_params_must_be_a_frozen_dataclass():
    @dataclasses.dataclass
    class Loose:
        x: int = 1

    with pytest.raises(TypeError, match="frozen dataclass"):
        _registry().register("a", params=Loose)


def test_resolve_keeps_order():
    reg = _registry()
    for name in ("a", "b", "c"):
        reg.register(name)(lambda: None)
    assert [p.name for p in reg.resolve(["c", "a"])] == ["c", "a"]


def test_loader_runs_once_on_first_lookup():
    calls = []
    reg = Registry("lazy", loader=lambda: calls.append(1))
    reg.available()
    reg.available()
    assert calls == [1]


def test_importing_qpocket_loads_no_plugin():
    code = ("import sys, qpocket; "
            "loaded = [m for m in sys.modules if m.startswith('qpocket.plugins')]; "
            "assert not loaded, loaded")
    subprocess.run([sys.executable, "-c", code], check=True)
