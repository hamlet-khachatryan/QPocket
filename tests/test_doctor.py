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


# --- the machines PyMOL used to rule out ------------------------------------


# --- machines that do not ---------------------------------------------------


# --- the exit code is what a script will branch on -------------------------


def test_a_usable_machine_exits_zero(monkeypatch, capsys):
    verdict(monkeypatch)  # installs the monkeypatches
    assert probe.main([]) == 0


def test_an_unusable_machine_exits_nonzero(monkeypatch, capsys):
    verdict(monkeypatch, machine="aarch64")
    assert probe.main([]) == 1


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


# --- the installed doctor: the optional screen extra -------------------------


