"""Deciding whether a machine can run Q-Pocket, before anything is installed.

Two dependencies ship as platform wheels with no source fallback, so the install
either can or cannot succeed and pip is the wrong thing to find that out with:

    apbs-binary        x86_64, glibc >= 2.28
    pymol-open-source  x86_64, glibc >= 2.35  (3.2.0a0)
                       x86_64, glibc >= 2.34, CPython 3.11 only  (3.1.0a0)

Rocky/RHEL 8 is glibc 2.28 and Rocky 9 is 2.34, so most HPC misses the 2.35
wheel by a hair. These tests pin the verdict for each of those machines, because
the machine that matters cannot be reached from here to try it on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import qpocket_doctor as probe  # noqa: E402

FAIL, OK, WARN = probe.FAIL, probe.OK, probe.WARN


def verdict(monkeypatch, system="Linux", machine="x86_64", glibc=(2, 35),
            python=(3, 11, 7)):
    """Run the platform check against a described machine."""
    monkeypatch.setattr(probe.platform, "system", lambda: system)
    monkeypatch.setattr(probe.platform, "machine", lambda: machine)
    monkeypatch.setattr(probe.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(probe, "glibc_version", lambda: glibc)
    monkeypatch.setattr(probe.sys, "version_info", python)
    rows, advice = probe.check_platform()
    return {label: (status, detail) for label, status, detail in rows}, advice


def status_of(rows, label):
    return rows[label][0]


# --- machines that work ----------------------------------------------------


def test_a_modern_ubuntu_node_is_fine(monkeypatch):
    rows, _ = verdict(monkeypatch, glibc=(2, 35))
    assert all(s != FAIL for s, _ in rows.values())


def test_macos_needs_no_glibc_checks(monkeypatch):
    rows, _ = verdict(monkeypatch, system="Darwin", machine="arm64", glibc=None)
    assert "pymol wheel" not in rows and status_of(rows, "platform") == OK


# --- machines that do not ---------------------------------------------------


def test_rocky_9_misses_the_pymol_wheel_unless_python_is_311(monkeypatch):
    """glibc 2.34: only the 3.1.0a0 wheel exists, and only for CPython 3.11."""
    rows, _ = verdict(monkeypatch, glibc=(2, 34), python=(3, 12, 0))
    assert status_of(rows, "pymol wheel") == FAIL


def test_rocky_9_works_on_python_311(monkeypatch):
    rows, advice = verdict(monkeypatch, glibc=(2, 34), python=(3, 11, 7))
    assert status_of(rows, "pymol wheel") == OK


def test_rocky_9_on_python_311_says_which_version_to_pin(monkeypatch):
    _, advice = verdict(monkeypatch, glibc=(2, 34), python=(3, 11, 7))
    assert any("3.1.0a0" in a for a in advice)


def test_rocky_8_can_still_run_apbs(monkeypatch):
    """glibc 2.28 is exactly the apbs-binary floor."""
    rows, _ = verdict(monkeypatch, glibc=(2, 28))
    assert status_of(rows, "apbs-binary wheel") == OK


def test_centos_7_cannot_install_apbs(monkeypatch):
    rows, _ = verdict(monkeypatch, glibc=(2, 17))
    assert status_of(rows, "apbs-binary wheel") == FAIL


def test_centos_7_is_told_where_else_to_get_apbs(monkeypatch):
    _, advice = verdict(monkeypatch, glibc=(2, 17))
    assert any("module" in a or "conda" in a for a in advice)


def test_arm_linux_is_refused_outright(monkeypatch):
    """apbs-binary publishes no aarch64 wheel at all."""
    rows, _ = verdict(monkeypatch, machine="aarch64")
    assert status_of(rows, "platform") == FAIL


def test_arm_linux_explains_why(monkeypatch):
    _, advice = verdict(monkeypatch, machine="aarch64")
    assert any("aarch64" in a for a in advice)


def test_windows_is_refused(monkeypatch):
    rows, _ = verdict(monkeypatch, system="Windows", machine="AMD64", glibc=None)
    assert status_of(rows, "platform") == FAIL


@pytest.mark.parametrize("python", [(3, 8, 10), (3, 14, 0)])
def test_an_unsupported_python_is_flagged(monkeypatch, python):
    rows, _ = verdict(monkeypatch, python=python)
    assert status_of(rows, "python") in (FAIL, WARN)


def test_an_undeterminable_glibc_warns_rather_than_guessing(monkeypatch):
    rows, _ = verdict(monkeypatch, glibc=None)
    assert status_of(rows, "glibc") == WARN


# --- the exit code is what a script will branch on -------------------------


def test_a_usable_machine_exits_zero(monkeypatch, capsys):
    verdict(monkeypatch)  # installs the monkeypatches
    assert probe.main([]) == 0


def test_an_unusable_machine_exits_nonzero(monkeypatch, capsys):
    verdict(monkeypatch, machine="aarch64")
    assert probe.main([]) == 1


def test_the_probe_imports_nothing_outside_the_standard_library():
    """It has to run before `pip install`, so it cannot need the package."""
    source = (Path(probe.__file__)).read_text()
    assert "qpocket_pipeline" not in source


# --- the smoke tests the installed doctor runs -----------------------------


def _is_mocked(name, attribute):
    """conftest replaces the heavyweight wheels with mocks for the whole suite.

    It installs a real ModuleType carrying MagicMock attributes, so the module
    type looks genuine and the attribute is what gives it away.
    """
    import sys as _sys
    from unittest.mock import MagicMock

    module = _sys.modules.get(name)
    return isinstance(getattr(module, attribute, None), MagicMock)


@pytest.mark.skipif(_is_mocked("apbs_binary", "popen_apbs"),
                    reason="apbs_binary is mocked by conftest")
def test_apbs_actually_runs():
    """An import proves nothing: apbs-binary patches its library path on macOS only."""
    from qpocket_pipeline.doctor import run_apbs_smoke_test

    seconds, detail = run_apbs_smoke_test()
    assert "grid" in detail


@pytest.mark.skipif(_is_mocked("apbs_binary", "popen_apbs"),
                    reason="toolchain is mocked by conftest")
def test_pdb2pqr_actually_runs():
    from qpocket_pipeline.doctor import run_pdb2pqr_smoke_test

    seconds, detail = run_pdb2pqr_smoke_test()
    assert "bytes" in detail
