"""Proving the toolchain works, rather than that it imports.

``apbs_binary`` patches its library path only on macOS
(``apbs_binary/executable.py``); on Linux the wheel's binary has to find its
shared objects by itself. It imports cleanly either way, so an import check says
nothing useful. The only honest test is to run APBS on a real input deck and
look at what comes back.

The system used here is deliberately tiny -- three atoms on a coarse grid -- so
the check costs a second or two and can be run inside a SLURM allocation without
thinking about it.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Tuple

#: A three-atom PQR: enough for a valid Poisson-Boltzmann calculation.
SMOKE_PQR = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  0.1000 1.8240
ATOM      2  CA  ALA A   1       1.500   0.000   0.000 -0.1000 1.9080
ATOM      3  C   ALA A   1       2.000   1.400   0.000  0.2000 1.9080
TER
END
"""

SMOKE_IN = """\
read
    mol pqr {pqr}
end
elec name smoke
    mg-auto
    dime 33 33 33
    cglen 20.0 20.0 20.0
    fglen 15.0 15.0 15.0
    cgcent mol 1
    fgcent mol 1
    mol 1
    lpbe
    bcfl sdh
    pdie 2.0
    sdie 78.54
    srfm smol
    chgm spl2
    sdens 10.0
    srad 1.4
    swin 0.3
    temp 298.15
    calcenergy total
    calcforce no
    write pot dx "{out}"
end
quit
"""


def run_apbs_smoke_test(timeout: int = 300) -> Tuple[float, str]:
    """Run APBS on a three-atom system and confirm it writes a potential grid.

    Args:
        timeout: Seconds to wait before giving up.

    Returns:
        ``(seconds, detail)`` where *detail* names the grid that was produced.

    Raises:
        ImportError: If ``apbs_binary`` is not installed.
        RuntimeError: If APBS exits non-zero, times out, or writes no grid. The
            message carries APBS's own stderr, which is where a missing shared
            object shows up on Linux.
    """
    from apbs_binary import popen_apbs

    started = time.time()
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        pqr = tmp / "smoke.pqr"
        deck = tmp / "smoke.in"
        out = tmp / "smoke"
        pqr.write_text(SMOKE_PQR)
        deck.write_text(SMOKE_IN.format(pqr=pqr, out=out))

        # cwd=tmp for the same reason the pipeline does it: APBS drops an io.mc
        # beside the process, not beside its output.
        process = popen_apbs(
            [str(deck)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(tmp)
        )
        try:
            _, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError(f"APBS did not finish within {timeout}s")

        message = (err or b"").decode(errors="replace").strip()
        grid = next((p for p in (out.with_suffix(".dx"), tmp / "smoke.pot.dx")
                     if p.exists()), None)

        if grid is None:
            detail = message or f"exit code {process.returncode}, no output"
            raise RuntimeError(
                "APBS produced no potential grid. On Linux this is usually a "
                f"missing shared library the wheel cannot locate. APBS said: {detail}"
            )

        return time.time() - started, f"wrote {grid.stat().st_size // 1024} KiB grid"


def run_pdb2pqr_smoke_test(timeout: int = 300) -> Tuple[float, str]:
    """Confirm the ``pdb2pqr`` console script runs and produces a PQR.

    Returns:
        ``(seconds, detail)``.

    Raises:
        RuntimeError: If pdb2pqr is missing, fails, or writes nothing.
    """
    import shutil

    if shutil.which("pdb2pqr") is None:
        raise RuntimeError("pdb2pqr is not on PATH")

    started = time.time()
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        pdb = tmp / "smoke.pdb"
        pqr = tmp / "smoke.pqr"
        pdb.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n"
            "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O\n"
            "ATOM      5  CB  ALA A   1       1.988  -0.773  -1.199  1.00  0.00           C\n"
            "TER\nEND\n"
        )
        done = subprocess.run(
            ["pdb2pqr", "--ff=PARSE", "--whitespace", "--noopt", str(pdb), str(pqr)],
            capture_output=True, text=True, timeout=timeout, cwd=str(tmp),
        )
        if not pqr.exists() or pqr.stat().st_size == 0:
            raise RuntimeError(
                f"pdb2pqr wrote no PQR (exit {done.returncode}): {done.stderr.strip()[:300]}"
            )
        return time.time() - started, f"wrote {pqr.stat().st_size} bytes"
