"""Deciding whether a machine can run Q-Pocket, before anything is installed.

One dependency ships only as platform wheels with no source fallback, so the
install either can or cannot succeed and pip is the wrong thing to find that out
with:

    apbs-binary        x86_64, glibc >= 2.28

PyMOL used to add a stricter one (glibc >= 2.34, CPython 3.11 only on Rocky 9),
which ruled out Rocky/RHEL 8 -- the most common cluster OS -- entirely. It was
replaced by Biopython. These tests pin the verdict for each common machine,
because the machine that matters cannot be reached from here to try it on.
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
    assert "apbs-binary wheel" not in rows and status_of(rows, "platform") == OK


# --- the machines PyMOL used to rule out ------------------------------------


@pytest.mark.parametrize("python", [(3, 9, 18), (3, 11, 7), (3, 12, 4)])
def test_rocky_8_is_supported_on_any_current_python(monkeypatch, python):
    """glibc 2.28 is exactly the apbs-binary floor, and nothing else is gated."""
    rows, _ = verdict(monkeypatch, glibc=(2, 28), python=python)
    assert all(s != FAIL for s, _ in rows.values())


@pytest.mark.parametrize("python", [(3, 10, 0), (3, 12, 4)])
def test_rocky_9_no_longer_needs_python_311(monkeypatch, python):
    rows, _ = verdict(monkeypatch, glibc=(2, 34), python=python)
    assert all(s != FAIL for s, _ in rows.values())


def test_pymol_is_no_longer_checked(monkeypatch):
    rows, _ = verdict(monkeypatch, glibc=(2, 28))
    assert not any("pymol" in label for label in rows)


# --- machines that do not ---------------------------------------------------


def test_centos_7_cannot_install_apbs(monkeypatch):
    rows, _ = verdict(monkeypatch, glibc=(2, 17))
    assert status_of(rows, "apbs-binary wheel") == FAIL


def test_centos_7_is_told_where_else_to_get_apbs(monkeypatch):
    _, advice = verdict(monkeypatch, glibc=(2, 17))
    assert any("module" in a or "container" in a for a in advice)


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


@pytest.mark.parametrize("python", [(3, 6, 8), (3, 8, 10)])
def test_a_too_old_python_fails_with_a_way_forward(monkeypatch, python):
    """3.6 is the system python3 on RHEL 7 and 8 -- the machine this crashed on."""
    rows, advice = verdict(monkeypatch, python=python)
    assert status_of(rows, "python") == FAIL and any("module" in a for a in advice)


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


def test_the_probe_parses_as_python_3_4():
    """It crashed on a node whose python3 was 3.6, before it could say so."""
    import ast
    source = Path(probe.__file__).read_text()
    tree = ast.parse(source, feature_version=(3, 4))  # f-strings would fail here
    futures = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module == "__future__" for a in n.names}
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)}
    keywords = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords}
    annotated = any(isinstance(n, ast.FunctionDef) and (n.returns or any(a.annotation for a in n.args.args))
                    for n in ast.walk(tree))
    assert "annotations" not in futures      # 3.7+
    assert "run" not in calls                # subprocess.run is 3.5+
    assert not {"capture_output", "text"} & keywords  # 3.7+
    assert not annotated


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
