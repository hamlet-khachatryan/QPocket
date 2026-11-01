"""Pocket comparison: standalone reproduces the pipeline, specs parse, CLI runs."""
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner

from qpocket.cli import cli
from qpocket.types import StructureData

GOLD = Path(__file__).resolve().parent / "golden/expected/7KG6"


def _blob(center, n=80, seed=0):
    return np.random.default_rng(seed).normal(center, 1.2, size=(n, 3))


def _pdb(tmp_path):
    lines = []
    serial = 1
    for i in range(1, 9):  # a short strand of alanines along x
        for name, dx, el in (("N", 0.0, "N"), ("CA", 1.2, "C"), ("C", 2.4, "C")):
            lines.append(f"ATOM  {serial:5d}  {name:<3} ALA A{i:4d}    {i * 3.8 + dx:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           {el}")
            serial += 1
    lines.append(f"HETATM{serial:5d}  C1  LIG A 900    {15.0:8.3f}{4.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    path = tmp_path / "toy.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def _fake_tools(monkeypatch):
    from qpocket.io import tables, tools
    monkeypatch.setattr(tools, "run_cavfind", lambda path, name: [_blob((50, 50, 50)), _blob((15, 4, 0), seed=2)])
    monkeypatch.setattr(tools, "protonate", lambda pdb, out: Path(out).write_text("") or Path(out))
    monkeypatch.setattr(tables, "read_pqr", lambda path: None)
    monkeypatch.setattr(tools, "structure_data", lambda path, pqr: StructureData(
        np.zeros((3, 3)), np.ones(3), np.ones(3), np.zeros((0, 3)), np.zeros((1, 3)), np.zeros((0, 3))))
    monkeypatch.setattr(tools, "apbs_potential", lambda path, pts: np.zeros(len(pts)))


def test_cli_featurize_then_compare(tmp_path, monkeypatch):
    _fake_tools(monkeypatch)
    pdb = _pdb(tmp_path)
    runner = CliRunner()
    for name, pocket in (("a", ["--ligand", "LIG"]), ("b", ["--residues", "ALA4,ALA5"])):
        res = runner.invoke(cli, ["featurize", str(pdb), *pocket, "-o", str(tmp_path / f"{name}.npz")])
        assert res.exit_code == 0, res.output
    res = runner.invoke(cli, ["compare", str(tmp_path / "a.npz"), str(tmp_path / "b.npz"), "-o", str(tmp_path / "d.csv")])
    assert res.exit_code == 0, res.output
    assert pd.read_csv(tmp_path / "d.csv", index_col=0).shape == (2, 2)


