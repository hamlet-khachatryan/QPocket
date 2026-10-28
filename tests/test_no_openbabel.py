"""Nothing in the package depends on OpenBabel or PLIP."""
import ast
import os
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src/qpocket"
FILES = [p for p in SRC.rglob("*.py") if "vendor" not in p.parts]
BANNED = {"openbabel", "pybel", "plip"}


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_banned_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                   else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for module in modules:
            assert module.split(".")[0] not in BANNED, f"{path.name} imports {module}"


def test_the_vendored_plip_is_gone():
    assert not (SRC.parent / "plip").exists()


def test_the_lockfile_has_no_openbabel():
    lock = (SRC.parents[1] / "uv.lock").read_text(encoding="utf-8")
    assert "openbabel" not in lock


def test_plip_is_preserved_on_a_branch():
    """The vendored copy is deleted here, not lost."""
    import subprocess
    branches = subprocess.run(["git", "branch", "--list", "legacy/plip"],
                              capture_output=True, text=True,
                              cwd=SRC.parents[1]).stdout
    if "legacy/plip" not in branches and os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.skip("CI checks out only the commit under test, never the archive branch")
    assert "legacy/plip" in branches
