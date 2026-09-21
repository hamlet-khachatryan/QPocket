"""core/ is pure: no I/O, no logging, no config, no pipeline imports."""
import ast
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "src/qpocket/core"
FORBIDDEN_IMPORTS = {"qpocket.io", "qpocket.pipelines", "eliot", "subprocess",
                     "requests", "h5py", "hydra", "omegaconf"}
FORBIDDEN_CALLS = {"open"}
FORBIDDEN_PARAMS = {"cfg", "dirs"}

# Each Phase-B task appends the modules it has made pure.
CORE_MODULES_DONE = ["clustering.py", "spectral.py"]


def _tree(name):
    return ast.parse((CORE / name).read_text(encoding="utf-8"))


def _imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


@pytest.mark.parametrize("name", CORE_MODULES_DONE)
def test_core_imports_nothing_impure(name):
    for mod in _imported_modules(_tree(name)):
        assert not any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN_IMPORTS), (name, mod)


@pytest.mark.parametrize("name", CORE_MODULES_DONE)
def test_core_opens_no_files(name):
    calls = {n.func.id for n in ast.walk(_tree(name))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls & FORBIDDEN_CALLS, name


@pytest.mark.parametrize("name", CORE_MODULES_DONE)
def test_core_takes_no_config_or_dirs(name):
    for fn in ast.walk(_tree(name)):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            assert not params & FORBIDDEN_PARAMS, (name, fn.name)


def test_rules_catch_an_impure_module(tmp_path, monkeypatch):
    """Guard against the checks silently passing everything."""
    bad = tmp_path / "bad.py"
    bad.write_text("import subprocess\ndef f(cfg):\n    return open('x')\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "CORE", tmp_path)
    with pytest.raises(AssertionError):
        test_core_imports_nothing_impure("bad.py")
    with pytest.raises(AssertionError):
        test_core_opens_no_files("bad.py")
    with pytest.raises(AssertionError):
        test_core_takes_no_config_or_dirs("bad.py")
