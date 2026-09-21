"""Every public function and class has a short Google-style docstring."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src/qpocket"
FILES = sorted(p for p in SRC.rglob("*.py") if "vendor" not in p.parts)
SECTIONS = ("Args:", "Returns:", "Raises:", "Yields:", "Attributes:")


def _is_click_command(node):
    # A click command's docstring is its --help text, written for users.
    return any("command" in ast.unparse(d) or "group" in ast.unparse(d)
               for d in getattr(node, "decorator_list", []))


def _public(tree):
    # Module-level functions and classes, and methods; nested helpers are private.
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            yield from (m for m in node.body if isinstance(m, ast.FunctionDef) and not m.name.startswith("_"))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            if not _is_click_command(node):
                yield node


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_docstrings_are_short(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert len((ast.get_docstring(tree) or "").splitlines()) <= 3, "module docstring over 3 lines"
    for node in _public(tree):
        doc = ast.get_docstring(node)
        assert doc, f"{node.name} has no docstring"
        first, *rest = doc.split("\n\n")
        assert "\n" not in first.strip(), f"{node.name}: summary must be one line"
        assert all(block.lstrip().startswith(SECTIONS) for block in rest), \
            f"{node.name}: only Args/Returns/Raises/Yields/Attributes after the summary"
        for block in rest:
            items = block.splitlines()[1:]
            assert all(line.startswith("    ") and not line.startswith("        ") for line in items), \
                f"{node.name}: one line per item in {block.splitlines()[0]}"
