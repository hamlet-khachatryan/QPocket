"""No POSIX-only constructs outside the SLURM renderers."""
import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src/qpocket"
FILES = sorted(p for p in SRC.rglob("*.py") if "vendor" not in p.parts)
RULES = {
    "hard-coded /tmp": re.compile(r"""["']/tmp\b"""),
    "shell=True": re.compile(r"shell\s*=\s*True"),
    "bash invocation": re.compile(r"""\[\s*["']bash["']"""),
}


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_posix_only_constructs(path):
    text = path.read_text(encoding="utf-8")
    for label, rule in RULES.items():
        assert not rule.search(text), f"{label} in {path.name}"


def _text_io_without_encoding(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or any(k.arg == "encoding" for k in node.keywords):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("read_text", "write_text"):
            yield node
        elif isinstance(f, ast.Name) and f.id == "open" or isinstance(f, ast.Attribute) and f.attr == "open" \
                and not (isinstance(f.value, ast.Name) and f.value.id in ("h5py", "gzip", "zipfile", "tarfile")):
            pos = 0 if isinstance(f, ast.Attribute) else 1
            mode = node.args[pos] if len(node.args) > pos else next((k.value for k in node.keywords if k.arg == "mode"), None)
            if isinstance(mode, ast.Constant) and "b" in str(mode.value):
                continue
            yield node


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_text_files_are_opened_as_utf8(path):
    """Windows defaults to cp1252; every text read and write names its encoding."""
    missing = [n.lineno for n in _text_io_without_encoding(ast.parse(path.read_text(encoding="utf-8")))]
    assert not missing, f"text I/O without encoding at lines {missing}"
