"""The explorer notebook ships with the package and runs in the installed env."""
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from qpocket.viz import explorer_path

REPO = Path(__file__).resolve().parents[1]


def test_the_notebook_lives_in_the_package():
    notebook = explorer_path()
    assert notebook.exists()
    assert notebook.is_relative_to(REPO / "src/qpocket")


def test_it_declares_no_inline_dependencies():
    """Inline metadata makes marimo offer a sandboxed venv, defaulting to yes in a
    terminal. That venv cannot see qpocket, so the kernel dies on import and the
    page reports 'kernel not found'."""
    assert "# /// script" not in explorer_path().read_text(encoding="utf-8")


def test_marimo_would_not_sandbox_it():
    reader = pytest.importorskip(
        "marimo._utils.inline_script_metadata").PyProjectReader
    assert not reader.from_filename(str(explorer_path())).dependencies


def test_it_imports_qpocket_directly():
    """Shipped inside the package, it needs no sys.path juggling."""
    text = explorer_path().read_text(encoding="utf-8")
    assert "import qpocket.viz as qviz" in text
    assert "sys.path.insert" not in text


def test_explore_is_offered_on_the_command_line():
    from qpocket.cli import cli

    result = CliRunner().invoke(cli, ["explore", "--help"])
    assert result.exit_code == 0
    assert "run directory" in result.output.lower()


@pytest.mark.golden
def test_the_wheel_carries_the_notebook(tmp_path):
    import shutil
    import subprocess

    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    subprocess.run(["uv", "build", "--wheel", "-o", str(tmp_path)], cwd=REPO, check=True)
    names = zipfile.ZipFile(next(tmp_path.glob("*.whl"))).namelist()
    assert "qpocket/viz/explorer.py" in names


def test_regroup_controls_put_the_config_defaults_on_their_step_grid():
    import ast

    from qpocket.config import SitesConfig

    tree = ast.parse((Path(__file__).parents[1] / "src/qpocket/viz/explorer.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_number"]
    assert len(calls) == 5
    for call in calls:
        key, lo, _, step = (ast.literal_eval(a) for a in call.args[:4])
        default = getattr(SitesConfig(), key)
        assert abs(round((default - lo) / step) * step - (default - lo)) < 1e-9, key
