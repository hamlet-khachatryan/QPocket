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
CORE_MODULES_DONE = ["clustering.py", "spectral.py", "features.py", "structure.py", "sequence.py",
                     "geometry.py", "pocket.py", "interactions.py", "screening.py",
                     "perception.py", "detect.py",
                     "protonation.py", "hydroxyl.py", "registry.py", "ports.py", "result.py", "types.py", "receptor.py", "crossdock.py", "sites.py"]


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


PKG = CORE.parent


def _modules_under(sub):
    root = PKG / sub
    return sorted(root.rglob("*.py")) if root.exists() else []


def _imports_of(path):
    return list(_imported_modules(ast.parse(path.read_text(encoding="utf-8"))))


@pytest.mark.parametrize("path", _modules_under("plugins"), ids=lambda p: str(p.relative_to(PKG)))
def test_plugins_import_no_skeleton_or_apps(path):
    banned = ("qpocket.pipelines", "qpocket.arms", "qpocket.cli", "qpocket.viz")
    for mod in _imports_of(path):
        assert not mod.startswith(banned), (path.name, mod)


@pytest.mark.parametrize(
    "path",
    [p for sub in ("pipelines", "arms", "cli") for p in _modules_under(sub)] + [PKG / "config.py"],
    ids=lambda p: str(p.relative_to(PKG)),
)
def test_only_registries_reach_plugins(path):
    for mod in _imports_of(path):
        assert not mod.startswith("qpocket.plugins"), (path.name, mod)


@pytest.mark.parametrize("path", _modules_under("pipelines") + _modules_under("core"),
                         ids=lambda p: str(p.relative_to(PKG)))
def test_the_core_run_imports_no_arm(path):
    for mod in _imports_of(path):
        assert not mod.startswith("qpocket.arms"), (path.name, mod)


def _arm_of(path):
    rel = path.relative_to(PKG / "arms").parts
    return rel[0].removesuffix(".py")


@pytest.mark.parametrize("path", _modules_under("arms"), ids=lambda p: str(p.relative_to(PKG)))
def test_arms_do_not_import_each_other(path):
    own = _arm_of(path)
    for mod in _imports_of(path):
        if mod.startswith("qpocket.arms."):
            assert mod.split(".")[2] == own, (path.name, mod)


def test_the_arms_exist():
    assert {p.stem for p in (PKG / "arms").iterdir()} >= {"screen", "compare", "recluster"}
