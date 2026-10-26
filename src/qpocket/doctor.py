"""Proving the toolchain works, rather than that it imports."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

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
        RuntimeError: If APBS is missing, exits non-zero, times out, or writes no grid.
    """
    from qpocket.io.tools import apbs_process

    started = time.time()
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        pqr = tmp / "smoke.pqr"
        deck = tmp / "smoke.in"
        out = tmp / "smoke"
        pqr.write_text(SMOKE_PQR, encoding="utf-8")
        deck.write_text(SMOKE_IN.format(pqr=pqr, out=out), encoding="utf-8")

        # cwd=tmp for the same reason the pipeline does it: APBS drops an io.mc
        # beside the process, not beside its output.
        process = apbs_process(
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
    from qpocket.io.tools import executable

    pdb2pqr = executable("pdb2pqr")

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
            "TER\nEND\n",
            encoding="utf-8",
        )
        done = subprocess.run(
            [pdb2pqr, "--ff=PARSE", "--whitespace", "--noopt", str(pdb), str(pqr)],
            capture_output=True, text=True, timeout=timeout, cwd=str(tmp),
        )
        if not pqr.exists() or pqr.stat().st_size == 0:
            raise RuntimeError(
                f"pdb2pqr wrote no PQR (exit {done.returncode}): {done.stderr.strip()[:300]}"
            )
        return time.time() - started, f"wrote {pqr.stat().st_size} bytes"


def required_tools(cfg) -> Dict[str, Tuple[str, ...]]:
    """Executables the configured plug-ins run, and which plug-ins need each.

    Args:
        cfg: A QPocket configuration.

    Returns:
        Tool name to the plug-in names that declare it.
    """
    from qpocket.core.ports import CHARGES, CONTACTS, FEATURES, POCKETS

    pocket = "cavities"
    plugins = (list(FEATURES.resolve(list(cfg.features.enabled))) + [CHARGES.get(cfg.charges.use)]
               + list(CONTACTS.resolve(list(cfg.contacts.use))) + [POCKETS.get(pocket)])
    needs: Dict[str, Tuple[str, ...]] = {}
    for plugin in plugins:
        for tool in plugin.tools:
            needs[tool] = needs.get(tool, ()) + (plugin.name,)
    return needs
