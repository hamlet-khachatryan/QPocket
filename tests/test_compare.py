"""Pocket comparison: standalone reproduces the pipeline, specs parse, CLI runs."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import qpocket as qp
from qpocket.cli import cli
from qpocket.io.cache import load_geodesic, load_scaler
from qpocket.types import StructureData

GOLD = Path(__file__).resolve().parent / "golden/expected/7KG6"


@pytest.mark.skipif(not (GOLD / "metadata/scaler.npz").exists(), reason="needs golden run")
def test_pair_distance_reproduces_the_run():
    scaler = load_scaler(GOLD / "metadata/scaler.npz")
    fps = [qp.fingerprint(*load_geodesic(GOLD / "pockets" / pid / "geospectral_cache.npz"),
                          scaler, global_k=100, sigma=10.0) for pid in ("4B1J", "6O9Y")]
    got = qp.pocket_distance(*fps)
    run = pd.read_csv(GOLD / "metadata/spectral_distance_matrix.csv", index_col=0)
    assert np.isclose(got["combined"], run.loc["4B1J", "6O9Y"], rtol=1e-6)


@pytest.mark.parametrize("text,expected", [
    ("7KG6:WDG", qp.PocketSpec("7KG6", None, "WDG")),
    ("a.pdb@B:TYR795,GLU756", qp.PocketSpec("a.pdb", "B", None, ("TYR795", "GLU756"))),
    ("x.cif:tyr795", qp.PocketSpec("x.cif", None, None, ("TYR795",))),
    (r"C:\data\a.pdb:A1R", qp.PocketSpec(r"C:\data\a.pdb", None, "A1R")),
])
def test_spec_parsing(text, expected):
    assert qp.PocketSpec.parse(text) == expected


@pytest.mark.parametrize("bad", ["7KG6", "7KG6:"])
def test_spec_needs_a_pocket(bad):
    with pytest.raises(ValueError):
        qp.PocketSpec.parse(bad)


def _blob(center, n=80, seed=0):
    return np.random.default_rng(seed).normal(center, 1.2, size=(n, 3))


def test_compare_features_is_symmetric_and_zero_on_the_diagonal():
    rng = np.random.default_rng(1)
    pockets = [(_blob(0, seed=i), rng.random((80, 6)) + i) for i in range(3)]
    result = qp.compare_features(pockets, ["a", "b", "c"])
    frame = result.to_frame()
    assert np.allclose(frame.values, frame.values.T) and np.allclose(np.diag(frame.values), 0)
    assert (frame.values[~np.eye(3, dtype=bool)] > 0).all()


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
    from qpocket.io import tools
    monkeypatch.setattr(tools, "run_cavfind", lambda path, name: [_blob((50, 50, 50)), _blob((15, 4, 0), seed=2)])
    monkeypatch.setattr(tools, "structure_data", lambda path: StructureData(
        np.zeros((3, 3)), np.ones(3), np.ones(3), np.zeros((0, 3)), np.zeros((1, 3)), np.zeros((0, 3))))
    monkeypatch.setattr(tools, "apbs_potential", lambda path, pts: np.zeros(len(pts)))


def test_featurize_picks_the_cavity_at_the_ligand(tmp_path, monkeypatch):
    _fake_tools(monkeypatch)
    points, features = qp.featurize(qp.PocketSpec(str(_pdb(tmp_path)), ligand="LIG"), tmp_path / "w")
    assert np.allclose(points.mean(axis=0), (15, 4, 0), atol=0.5) and features.shape == (80, 6)


def test_featurize_by_residues_crops_to_their_neighbourhood(tmp_path, monkeypatch):
    _fake_tools(monkeypatch)
    points, _ = qp.featurize(qp.PocketSpec(str(_pdb(tmp_path)), residues=("ALA4",), padding=5.0), tmp_path / "w")
    lining = np.array([[4 * 3.8 + dx, 0, 0] for dx in (0, 1.2, 2.4)])
    assert len(points) and (np.linalg.norm(points[:, None] - lining[None], axis=2).min(axis=1) <= 5.0).all()


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


def test_cli_rejects_a_spec_without_a_pocket():
    res = CliRunner().invoke(cli, ["compare-structures", "7KG6", "4B1J:A1R"])
    assert res.exit_code != 0 and "POCKET" in res.output
