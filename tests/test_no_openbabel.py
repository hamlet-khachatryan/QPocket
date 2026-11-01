"""Nothing in the package depends on OpenBabel or PLIP."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src/qpocket"
FILES = [p for p in SRC.rglob("*.py") if "vendor" not in p.parts]
BANNED = {"openbabel", "pybel", "plip"}


def _check_no_banned_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                   else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for module in modules:
            assert module.split(".")[0] not in BANNED, f"{path.name} imports {module}"


def test_no_banned_imports():
    failures = []
    for path in FILES:
        try:
            _check_no_banned_imports(path)
        except AssertionError as e:
            failures.append(f"{path}: {e}")
    assert not failures, "\n".join(failures)


