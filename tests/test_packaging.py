"""The wheel built by uv ships every package and data file the pipeline needs."""
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REQUIRED = [
    "plip/structure/preparation.py",
    "cavitomix/cavfind.py",
    "qpocket/conf/config.yaml",
    "qpocket/viz/vendor/3Dmol-2.4.2.min.js",
]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_wheel_contents(tmp_path):
    subprocess.run(["uv", "build", "--wheel", "-o", str(tmp_path)], cwd=REPO, check=True)
    names = zipfile.ZipFile(next(tmp_path.glob("*.whl"))).namelist()
    for required in REQUIRED:
        assert required in names, required
    assert not any(n.startswith("plip/plip/") for n in names)
    assert not any("__pycache__" in n for n in names)
