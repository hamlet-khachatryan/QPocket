"""Docking engines share one contract: a command line and a pose reader."""

import pytest

from qpocket.core import screening as scr
from qpocket.core.ports import DOCKING

VINA_OUT = """MODEL 1
REMARK VINA RESULT:    -8.1      0.000      0.000
ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  O   UNL     1       1.200   0.000   0.000  1.00  0.00     0.000 OA
ENDMDL
MODEL 2
REMARK VINA RESULT:    -7.4      1.900      2.600
ATOM      1  C   UNL     1       5.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  O   UNL     1       6.200   0.000   0.000  1.00  0.00     0.000 OA
ENDMDL
"""


def _job(**extra):
    row = {c: "1" for c in scr.JOB_COLUMNS}
    row.update(protein_pdbqt="r.pdbqt", ligand_pdbqt="l.pdbqt", output_pdbqt="poses/j.pdbqt")
    row.update(extra)
    return row


@pytest.mark.parametrize("name", ["quickvina", "vina"])
def test_vina_family_commands_are_todays(name):
    job = _job(engine=name)
    assert DOCKING.get(name).fn().argv(job, "/bin/x", 4) == scr.vina_argv(job, "/bin/x", 4)


def test_gnina_command_carries_the_box_and_output():
    argv = DOCKING.get("gnina").fn().argv(_job(output_pdbqt="poses/j.sdf"), "/bin/gnina", 2)
    assert argv[:5] == ["/bin/gnina", "-r", "r.pdbqt", "-l", "l.pdbqt"]
    assert argv[argv.index("-o") + 1] == "poses/j.sdf" and "--center_x" in argv and "--cpu" in argv


