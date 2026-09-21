#!/usr/bin/env python3
"""Check whether this machine can run Q-Pocket, before anything is installed.

Two of Q-Pocket's dependencies ship as platform wheels with no source fallback,
so whether the install can succeed at all is decided by the architecture and the
glibc version, not by pip:

    apbs-binary        x86_64, glibc >= 2.28
    pymol-open-source  x86_64, glibc >= 2.35 (3.2.0a0)
                       x86_64, glibc >= 2.34, CPython 3.11 only (3.1.0a0)

Run this first, on the machine that will do the work:

    python3 scripts/qpocket_doctor.py

It imports nothing outside the standard library and does not need Q-Pocket
installed, which is the point: a preflight you can only run after a successful
install tells you nothing you did not already know.

Once Q-Pocket *is* installed, ``qpocket doctor`` runs the fuller check that
actually executes APBS and featurises a pocket.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from typing import List, Optional, Tuple

#: Minimum glibc for each wheel-only dependency, and what it costs to miss it.
APBS_MIN_GLIBC = (2, 28)
PYMOL_MIN_GLIBC = (2, 35)
PYMOL_FALLBACK_GLIBC = (2, 34)  # 3.1.0a0, CPython 3.11 only

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def _mark(status: str) -> str:
    return {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]


def report(rows: List[Tuple[str, str, str]]) -> None:
    width = max(len(label) for label, _, _ in rows)
    for label, status, detail in rows:
        print(f"{_mark(status)} {label:<{width}}  {detail}")


def glibc_version() -> Optional[Tuple[int, int]]:
    """The running glibc version, or ``None`` when not glibc (macOS, musl).

    ``platform.libc_ver`` reads the executable rather than asking the loader and
    is unreliable on manylinux, so ``ldd --version`` is tried first.
    """
    try:
        out = subprocess.run(
            ["ldd", "--version"], capture_output=True, text=True, timeout=10
        ).stdout
        first = out.splitlines()[0] if out else ""
        token = first.strip().split()[-1]
        parts = token.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        pass

    try:
        name, version = platform.libc_ver()
        if name == "glibc" and version:
            parts = version.split(".")
            return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    return None


def fmt(version: Optional[Tuple[int, int]]) -> str:
    return "unknown" if version is None else f"{version[0]}.{version[1]}"


def check_platform() -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Report the machine, and collect any advice the caller needs to act on."""
    rows: List[Tuple[str, str, str]] = []
    advice: List[str] = []

    system = platform.system()
    machine = platform.machine()
    glibc = glibc_version() if system == "Linux" else None

    if system == "Linux":
        detail = f"{system} {machine}, glibc {fmt(glibc)}"
    else:
        detail = f"{system} {machine}"

    if system not in ("Linux", "Darwin"):
        rows.append(("platform", FAIL, f"{detail} -- only Linux and macOS are supported"))
        advice.append("Q-Pocket has only been built for Linux and macOS.")
    elif system == "Linux" and machine not in ("x86_64", "AMD64"):
        rows.append(("platform", FAIL, f"{detail} -- no aarch64 wheels exist"))
        advice.append(
            "apbs-binary publishes no aarch64 wheel, so APBS cannot come from pip on "
            "this machine. Use a site module, a conda-forge build, or a container."
        )
    else:
        rows.append(("platform", OK, detail))

    py = sys.version_info
    py_text = f"{py[0]}.{py[1]}.{py[2]} ({platform.python_implementation()})"
    if py < (3, 9):
        rows.append(("python", FAIL, f"{py_text} -- 3.9 or newer is required"))
        advice.append("Install Python 3.9 or newer.")
    elif py >= (3, 14):
        rows.append(("python", WARN, f"{py_text} -- newer than any pymol wheel"))
        advice.append("pymol-open-source publishes wheels for CPython 3.9-3.13.")
    else:
        rows.append(("python", OK, py_text))

    if system != "Linux":
        return rows, advice

    if glibc is None:
        rows.append(("glibc", WARN, "could not be determined; wheel checks are a guess"))
        return rows, advice

    if glibc >= APBS_MIN_GLIBC:
        rows.append(("apbs-binary wheel", OK, f"needs glibc >= {fmt(APBS_MIN_GLIBC)}"))
    else:
        rows.append(("apbs-binary wheel", FAIL,
                     f"needs glibc >= {fmt(APBS_MIN_GLIBC)}, found {fmt(glibc)}"))
        advice.append(
            "apbs-binary will not install here. Load APBS from a site module or "
            "conda-forge and put it on PATH."
        )

    if glibc >= PYMOL_MIN_GLIBC:
        rows.append(("pymol wheel", OK,
                     f"pymol-open-source 3.2.0a0 (glibc >= {fmt(PYMOL_MIN_GLIBC)})"))
    elif glibc >= PYMOL_FALLBACK_GLIBC and py[:2] == (3, 11):
        rows.append(("pymol wheel", OK,
                     "pymol-open-source 3.1.0a0 -- pin it, 3.2 needs glibc 2.35"))
        advice.append('Pin PyMOL: pip install "pymol-open-source==3.1.0a0"')
    elif glibc >= PYMOL_FALLBACK_GLIBC:
        rows.append(("pymol wheel", FAIL,
                     f"glibc {fmt(glibc)} only has a 3.1.0a0 wheel, and only for CPython 3.11"))
        advice.append(
            "Use CPython 3.11 and pin pymol-open-source==3.1.0a0, or get PyMOL from "
            "a site module or conda-forge."
        )
    else:
        rows.append(("pymol wheel", FAIL,
                     f"needs glibc >= {fmt(PYMOL_FALLBACK_GLIBC)}, found {fmt(glibc)}"))
        advice.append(
            "No PyMOL wheel works here, and every qpocket command imports it. "
            "A container on a newer base image is the practical route."
        )

    return rows, advice


def check_tools() -> List[Tuple[str, str, str]]:
    """Report site-provided tools that can stand in for a missing wheel."""
    import shutil

    rows = []
    for tool, note in (("apbs", "site APBS"), ("pymol", "site PyMOL"),
                       ("pdb2pqr", "pdb2pqr"), ("sbatch", "SLURM")):
        found = shutil.which(tool)
        rows.append((f"{note} on PATH", OK if found else WARN, found or "not found"))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quiet", action="store_true", help="print only problems and the verdict"
    )
    args = parser.parse_args(argv)

    print("Q-Pocket preflight (pre-install)\n")
    rows, advice = check_platform()
    tools = check_tools()

    if args.quiet:
        rows = [r for r in rows if r[1] != OK]
        tools = [r for r in tools if r[1] != OK]
    report(rows + tools)

    failures = [r for r in rows if r[1] == FAIL]
    print()
    if failures:
        print("This machine cannot install Q-Pocket as it stands.\n")
    else:
        print("This machine can install Q-Pocket:\n")
        print("    python3 -m venv .venv && . .venv/bin/activate")
        print("    pip install -e .")
        print("    qpocket doctor          # the fuller check, once installed\n")

    for line in advice:
        print(f"  - {line}")
    if advice:
        print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
