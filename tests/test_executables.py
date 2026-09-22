"""External tools are found the same way on every OS."""
import sys
from pathlib import Path

import pytest

from qpocket.io import tools


def _fake_exe(tmp_path, name):
    exe = tmp_path / (f"{name}.exe" if sys.platform == "win32" else name)
    exe.write_text("", encoding="utf-8")
    exe.chmod(0o755)
    return exe


def test_env_override_wins(monkeypatch, tmp_path):
    exe = _fake_exe(tmp_path, "apbs")
    monkeypatch.setenv("QPOCKET_APBS", str(exe))
    assert tools.apbs_command() == ("env", str(exe))


def test_bad_env_override_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("QPOCKET_APBS", str(tmp_path / "missing"))
    with pytest.raises(tools.ToolNotFoundError, match="QPOCKET_APBS"):
        tools.apbs_command()


def test_path_lookup_and_a_helpful_miss(monkeypatch, tmp_path):
    exe = _fake_exe(tmp_path, "vina")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert Path(tools.executable("vina")).resolve() == exe.resolve()
    monkeypatch.setenv("PATH", "")
    with pytest.raises(tools.ToolNotFoundError, match="engine_path"):
        tools.executable("vina")


def test_apbs_falls_back_to_path_without_the_wheel(monkeypatch, tmp_path):
    exe = _fake_exe(tmp_path, "apbs")
    monkeypatch.delenv("QPOCKET_APBS", raising=False)
    monkeypatch.setitem(sys.modules, "apbs_binary", None)   # import fails, as on Windows
    monkeypatch.setenv("PATH", str(tmp_path))
    assert tools.apbs_command()[0] == "path"
