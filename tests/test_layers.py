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


def _check_core_imports_nothing_impure(name):
    for mod in _imported_modules(_tree(name)):
        assert not any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN_IMPORTS), (name, mod)


def test_core_imports_nothing_impure():
    failures = []
    for name in CORE_MODULES_DONE:
        try:
            _check_core_imports_nothing_impure(name)
        except AssertionError as e:
            failures.append(f"{name}: {e}")
    assert not failures, "\n".join(failures)


PKG = CORE.parent


def _modules_under(sub):
    root = PKG / sub
    return sorted(root.rglob("*.py")) if root.exists() else []


def _imports_of(path):
    return list(_imported_modules(ast.parse(path.read_text(encoding="utf-8"))))


def _check_plugins_import_no_skeleton_or_apps(path):
    banned = ("qpocket.pipelines", "qpocket.arms", "qpocket.cli", "qpocket.viz")
    for mod in _imports_of(path):
        assert not mod.startswith(banned), (path.name, mod)


def test_plugins_import_no_skeleton_or_apps():
    failures = []
    for path in _modules_under("plugins"):
        try:
            _check_plugins_import_no_skeleton_or_apps(path)
        except AssertionError as e:
            failures.append(f"{path}: {e}")
    assert not failures, "\n".join(failures)


def _check_only_registries_reach_plugins(path):
    for mod in _imports_of(path):
        assert not mod.startswith("qpocket.plugins"), (path.name, mod)


def test_only_registries_reach_plugins():
    failures = []
    for path in [p for sub in ("pipelines", "arms", "cli") for p in _modules_under(sub)] + [PKG / "config.py"]:
        try:
            _check_only_registries_reach_plugins(path)
        except AssertionError as e:
            failures.append(f"{path}: {e}")
    assert not failures, "\n".join(failures)


def _check_the_core_run_imports_no_arm(path):
    for mod in _imports_of(path):
        assert not mod.startswith("qpocket.arms"), (path.name, mod)


def test_the_core_run_imports_no_arm():
    failures = []
    for path in _modules_under("pipelines") + _modules_under("core"):
        try:
            _check_the_core_run_imports_no_arm(path)
        except AssertionError as e:
            failures.append(f"{path}: {e}")
    assert not failures, "\n".join(failures)


def _arm_of(path):
    rel = path.relative_to(PKG / "arms").parts
    return rel[0].removesuffix(".py")


def _check_arms_do_not_import_each_other(path):
    own = _arm_of(path)
    for mod in _imports_of(path):
        if mod.startswith("qpocket.arms."):
            assert mod.split(".")[2] == own, (path.name, mod)


def test_arms_do_not_import_each_other():
    failures = []
    for path in _modules_under("arms"):
        try:
            _check_arms_do_not_import_each_other(path)
        except AssertionError as e:
            failures.append(f"{path}: {e}")
    assert not failures, "\n".join(failures)


