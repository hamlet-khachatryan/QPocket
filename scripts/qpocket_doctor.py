#!/usr/bin/env python3
"""Check whether this machine can run Q-Pocket, before anything is installed.

One of Q-Pocket's dependencies ships only as platform wheels with no source
fallback, so whether the install can succeed is decided by the architecture and
the glibc version, not by pip:

    apbs-binary        x86_64, glibc >= 2.28   (RHEL/Rocky 8 and newer)

PyMOL used to be a second, stricter one (glibc >= 2.34); it has been replaced by
Biopython and is no longer needed.

Run this first, on the machine that will do the work:

    python3 scripts/qpocket_doctor.py

It imports nothing outside the standard library and does not need Q-Pocket
installed, which is the point: a preflight you can only run after a successful
install tells you nothing you did not already know.

For the same reason it is written to *run* on any Python still found on a
cluster -- 2.7 and 3.4 onwards -- even though Q-Pocket itself needs 3.9. The
system ``python3`` on RHEL 7 and 8 is 3.6, and the most useful thing this script
can say there is "this Python is too old", which it cannot say if it fails to
parse. So: no f-strings, no annotations, no ``subprocess.run``.

Once Q-Pocket *is* installed, ``qpocket doctor`` runs the fuller check that
actually executes APBS and featurises a pocket.
"""

from __future__ import print_function

import argparse
import os
import platform
import subprocess
import sys

#: Minimum glibc for the one wheel-only dependency.
APBS_MIN_GLIBC = (2, 28)

#: Oldest Python Q-Pocket itself runs on.
MIN_PYTHON = (3, 9)

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def _mark(status):
    return {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]


def report(rows):
    width = max(len(label) for label, _, _ in rows)
    for label, status, detail in rows:
        print("{0} {1:<{2}}  {3}".format(_mark(status), label, width, detail))


def _run(command):
    """stdout of a command, or '' -- without subprocess.run (added in 3.5)."""
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        out, err = proc.communicate()
        return out or err or ""
    except Exception:
        return ""


def which(name):
    """shutil.which, which Python 2.7 does not have."""
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def glibc_version():
    """The running glibc version, or None when not glibc (macOS, musl).

    platform.libc_ver reads the executable rather than asking the loader and is
    unreliable on manylinux, so ``ldd --version`` is tried first.
    """
    out = _run(["ldd", "--version"])
    if out:
        try:
            token = out.splitlines()[0].strip().split()[-1]
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


def fmt(version):
    return "unknown" if version is None else "{0}.{1}".format(version[0], version[1])


def check_platform():
    """Report the machine, and collect any advice the caller needs to act on."""
    rows = []
    advice = []

    system = platform.system()
    machine = platform.machine()
    glibc = glibc_version() if system == "Linux" else None

    if system == "Linux":
        detail = "{0} {1}, glibc {2}".format(system, machine, fmt(glibc))
    else:
        detail = "{0} {1}".format(system, machine)

    if system == "Windows" and machine in ("AMD64", "x86_64"):
        rows.append(("platform", WARN, detail + " -- APBS must be installed separately"))
        advice.append(
            "No APBS wheel exists for Windows. Install APBS from "
            "github.com/Electrostatics/apbs/releases and put apbs.exe on PATH, or set "
            "QPOCKET_APBS to its full path. Everything else installs with uv. SLURM "
            "submission is Linux-only; use `qpocket screen run-local` for docking."
        )
    elif system not in ("Linux", "Darwin"):
        rows.append(("platform", FAIL, detail + " -- only Linux, macOS and Windows x86_64 are supported"))
        advice.append("Q-Pocket is built for Linux, macOS and Windows on x86_64.")
    elif system == "Linux" and machine not in ("x86_64", "AMD64"):
        rows.append(("platform", FAIL, detail + " -- no aarch64 wheels exist"))
        advice.append(
            "apbs-binary publishes no aarch64 wheel, so APBS cannot come from pip on "
            "this machine. Use a site module or a container."
        )
    else:
        rows.append(("platform", OK, detail))

    py = sys.version_info
    py_text = "{0}.{1}.{2} ({3}) at {4}".format(
        py[0], py[1], py[2], platform.python_implementation(), sys.executable
    )
    if tuple(py[:2]) < MIN_PYTHON:
        rows.append(("python", FAIL,
                     py_text + " -- Q-Pocket needs {0} or newer".format(fmt(MIN_PYTHON))))
        advice.append(
            "This python is too old to install Q-Pocket. Look for a newer one: "
            "`module avail python` then `module load python/3.11`, and re-run this "
            "check with that python. (The system python3 on RHEL 7 and 8 is 3.6.)"
        )
    else:
        rows.append(("python", OK, py_text))

    if system != "Linux":
        return rows, advice

    if glibc is None:
        rows.append(("glibc", WARN, "could not be determined; wheel checks are a guess"))
        return rows, advice

    if glibc >= APBS_MIN_GLIBC:
        rows.append(("apbs-binary wheel", OK, "needs glibc >= " + fmt(APBS_MIN_GLIBC)))
    else:
        rows.append(("apbs-binary wheel", FAIL,
                     "needs glibc >= {0}, found {1}".format(fmt(APBS_MIN_GLIBC), fmt(glibc))))
        advice.append(
            "apbs-binary will not install here. Load APBS from a site module and "
            "put it on PATH, or use a container."
        )

    return rows, advice


def check_tools():
    """Report site-provided tools that can stand in for a missing wheel."""
    rows = []
    for tool, note in (("apbs", "site APBS"), ("pdb2pqr", "pdb2pqr"),
                       ("sbatch", "SLURM")):
        found = which(tool)
        rows.append((note + " on PATH", OK if found else WARN, found or "not found"))
    return rows


def main(argv=None):
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
    if rows + tools:
        report(rows + tools)

    failures = [r for r in rows if r[1] == FAIL]
    print()
    if failures:
        print("This machine cannot install Q-Pocket as it stands.\n")
    else:
        print("This machine can install Q-Pocket:\n")
        print("    python3 -m pip install --user uv      # once; or the uv installer script")
        print("    uv sync --locked --extra viz --python \"$(command -v python3)\"")
        print("    . .venv/bin/activate")
        print("    qpocket doctor          # the fuller check, once installed\n")

    for line in advice:
        print("  - " + line)
    if advice:
        print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
