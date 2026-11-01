"""`qpocket prepare fragalysis`: ligand anchoring, definitions, the protein copy, the run around it."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from rdkit import Chem
from rdkit.Chem import AllChem

from qpocket.cli import cli
from qpocket.io import complex_prep as cp
from qpocket.io import components, fragalysis, interactions


def _fragment(smiles="NC(=O)c1ccccc1", offset=(0.0, 0.0, 0.0)):
    """A 3D heavy-atom molecule placed at ``offset``."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + offset[0], p.y + offset[1], p.z + offset[2]))
    return mol


def _names(mol):
    counts, names = {}, []
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
        names.append(f"{atom.GetSymbol()}{counts[atom.GetSymbol()]}")
    return names


def _atom_line(record, name, altloc, resname, chain, resseq, xyz, occ, element):
    return (f"{record:<6}{1:5d} {name:<4}{altloc:1}{resname:>3} {chain:1}{resseq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occ:6.2f}{20.0:6.2f}          {element:>2}")


def _ligand_lines(mol, chain, resseq, resname="LIG", altloc="", occ=1.0, shift=(0, 0, 0)):
    pos = mol.GetConformer().GetPositions() + np.asarray(shift, float)
    return [_atom_line("HETATM", n, altloc, resname, chain, resseq, p, occ, a.GetSymbol())
            for n, p, a in zip(_names(mol), pos, mol.GetAtoms())]


def _protein_lines(chain, origin, n=4):
    """A short backbone stub along x from ``origin``."""
    lines = []
    for i in range(n):
        for k, (name, el) in enumerate((("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"))):
            xyz = (origin[0] + 3.8 * i + 0.5 * k, origin[1], origin[2])
            lines.append(_atom_line("ATOM", name, "", "GLY", chain, i + 1, xyz, 1.0, el))
    return lines


# --- complex_prep -----------------------------------------------------------------------------

def test_anchor_matches_the_sdf_copy_of_two():
    mol = _fragment(offset=(20, 0, 0))
    text = "\n".join(_ligand_lines(mol, "A", 201, shift=(-20, 0, 0)) + _ligand_lines(mol, "B", 202))
    assert cp.match_anchor(cp.read_atoms(text), mol).key == ("B", 202, "", "LIG")


def test_anchor_needs_matching_coordinates_and_skips_protein():
    mol = _fragment()
    assert cp.match_anchor(cp.read_atoms("\n".join(_ligand_lines(mol, "A", 1, shift=(0.2, 0, 0)))), mol) is None
    as_protein = [line.replace("HETATM", "ATOM  ") for line in _ligand_lines(mol, "A", 1)]
    assert cp.match_anchor(cp.read_atoms("\n".join(as_protein)), mol) is None


def test_anchor_picks_the_matching_altloc():
    mol = _fragment()
    text = "\n".join(_ligand_lines(mol, "A", 201, altloc="A", occ=0.27)
                     + _ligand_lines(mol, "A", 201, altloc="B", occ=0.35, shift=(1.5, 0, 0)))
    anchor = cp.match_anchor(cp.read_atoms(text), mol)
    assert anchor.altloc == "A" and set(anchor.name_to_index) == set(_names(mol))


def test_codes_are_deterministic_and_sorted():
    assert cp.assign_codes(["c1ccccc1", "CCO", "CCO"]) == {"CCO": "Q00", "c1ccccc1": "Q01"}
    assert sorted(cp.assign_codes("C" * (i + 1) for i in range(37)).values())[36] == "Q10"


def test_definition_round_trips_through_qpocket(tmp_path, monkeypatch):
    mol = _fragment()
    anchor = cp.match_anchor(cp.read_atoms("\n".join(_ligand_lines(mol, "A", 201))), mol)
    (tmp_path / "Q00.cif").write_text(cp.component_cif("Q00", mol, anchor), encoding="utf-8")
    ligand_pdb = tmp_path / "lig.pdb"
    ligand_pdb.write_text("\n".join(_ligand_lines(mol, "A", 201, resname="Q00")) + "\nEND\n", encoding="utf-8")
    monkeypatch.setenv("QPOCKET_CCD_CACHE", str(tmp_path))
    components.component_bonds.cache_clear()
    try:
        assert interactions.ligand_smiles(ligand_pdb) == Chem.MolToSmiles(mol)
    finally:
        components.component_bonds.cache_clear()


def test_resolution_from_remarks():
    assert cp.resolution("REMARK   3   RESOLUTION RANGE HIGH (ANGSTROMS) :   1.95\n") == 1.95
    assert cp.resolution("REMARK   2 RESOLUTION.    1.80 ANGSTROMS.\n") == 1.80
    assert cp.resolution("HEADER    ----\n") is None


def _two_copy_complex():
    """Chains A and B 20 Å apart; the fragment sits on B (filed under chain A), as do EDO and a water."""
    mol = _fragment(offset=(5, 16.0, 0))
    lines = _protein_lines("A", (0, 0, 0)) + _protein_lines("B", (0, 20, 0))
    lines += _ligand_lines(mol, "A", 201)
    lines += [_atom_line("HETATM", n, "", "EDO", "A", 3, (10 + i * 0.7, 23, 2), 1.0, e)
              for i, (n, e) in enumerate([("C1", "C"), ("O1", "O"), ("C2", "C"), ("O2", "O")])]
    lines += _ligand_lines(_fragment(offset=(1, -3, 0)), "A", 202)       # the copy on chain A
    lines.append(_atom_line("HETATM", "O", "", "HOH", "W", 1, (3, 22, 0), 1.0, "O"))
    atoms = cp.read_atoms("\n".join(lines))
    return atoms, cp.match_anchor(atoms, mol)


def test_keeps_the_bound_chain_under_the_requested_id():
    atoms, anchor = _two_copy_complex()
    p = cp.prepare_complex(atoms, anchor, "Q00", ["HEADER    ----"], out_chain="A")
    out = cp.read_atoms("\n".join(p.lines))
    assert p.kept_chain == "B" and {a.chain for a in out} == {"A"}
    assert {a.resname for a in out if a.record == "HETATM"} == {"Q00", "EDO", "HOH"}
    assert sum(a.resname == "Q00" for a in out) == len(anchor.name_to_index)   # the chain-A copy is gone
    assert p.lines[0] == "HEADER    ----" and p.lines[-1] == "END"


def test_keeps_the_input_chain_id_by_default():
    atoms, anchor = _two_copy_complex()
    out = cp.read_atoms("\n".join(cp.prepare_complex(atoms, anchor, "Q00", []).lines))
    assert {a.chain for a in out} == {"B"}


# --- a Fragalysis download --------------------------------------------------------------------

def _download(root: Path) -> Path:
    """Two observations of one compound (on chains B and A) and one with no ligand file."""
    download = root / "Target"
    mol = _fragment(offset=(5, 16.0, 0))
    lines = _protein_lines("A", (0, 0, 0)) + _protein_lines("B", (0, 20, 0))
    lines += _ligand_lines(mol, "B", 301) + _ligand_lines(mol, "A", 302, shift=(0, -20, 0))
    complex_text = "\n".join(["REMARK   3   RESOLUTION RANGE HIGH (ANGSTROMS) :   1.50", *lines, "END"]) + "\n"
    for code, shift in (("x0001a", (0, 0, 0)), ("x0001b", (0, -20, 0))):
        folder = download / "aligned_files" / code
        folder.mkdir(parents=True)
        (folder / f"{code}.pdb").write_text(complex_text, encoding="utf-8")
        placed = Chem.Mol(mol)
        conf = placed.GetConformer()
        for i in range(placed.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (p.x + shift[0], p.y + shift[1], p.z + shift[2]))
        Chem.MolToMolFile(placed, str(folder / f"{code}_ligand.sdf"))
    (download / "aligned_files" / "x0002a").mkdir()
    (download / "aligned_files" / "x0002a" / "x0002a.pdb").write_text(complex_text, encoding="utf-8")
    (download / "Target_combined.sdf").write_text("", encoding="utf-8")
    pd.DataFrame({"Code": ["x0001a", "x0001b"], "Compound code": ["Z1", "Z1"],
                  "CanonSites alias": ["1 - site", "1 - site"]}).to_csv(download / "metadata.csv", index=False)
    return download


def test_prepare_writes_a_runnable_fragment_screen(tmp_path):
    out = tmp_path / "prep"
    summary = fragalysis.prepare(_download(tmp_path), out, fragalysis.Options(workers=2))
    assert (summary.prepared, summary.compounds, summary.reference) == (2, 1, "x0001a")
    assert summary.skipped == {"missing_file": 1}
    inputs = pd.read_csv(out / "inputs.csv")
    assert list(inputs["code"]) == ["x0001a", "x0001b"]
    assert list(inputs["source_chain"]) == ["B", "A"] and set(inputs["compound_code"]) == {"Z1"}
    assert set(inputs["canon_site"]) == {"1 - site"} and set(inputs["residue_code"]) == {"Q00"}
    assert (out / "ligand_definitions" / "Q00.cif").exists()
    assert all(Path(p).exists() for p in inputs["pdb_path"])
    config = (out / "qpocket.yaml").read_text()
    assert "pdb_id: Target" in config and "chain_id: A" in config and "workers: 2" in config
    assert f"reference_path: {out.resolve() / 'structures' / 'x0001a.pdb'}" in config
    assert f"QPOCKET_CCD_CACHE={out.resolve() / 'ligand_definitions'}" in (out / "run.sh").read_text()


def test_path_prefix_rewrites_every_path(tmp_path):
    out = tmp_path / "prep"
    fragalysis.prepare(_download(tmp_path), out, fragalysis.Options(path_prefix=Path("/vm/data/prep")))
    assert all(p.startswith("/vm/data/prep/structures/") for p in pd.read_csv(out / "inputs.csv")["pdb_path"])
    assert "workdir: /vm/data/prep/runs" in (out / "qpocket.yaml").read_text()


def test_not_a_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="aligned_files"):
        fragalysis.prepare(tmp_path, tmp_path / "prep")


def test_cli(tmp_path):
    download = _download(tmp_path)
    result = CliRunner().invoke(cli, ["prepare", "fragalysis", str(download), str(tmp_path / "prep"),
                                      "--run-name", "T1", "--cofactors", "MG|ZN"])
    assert result.exit_code == 0, result.output
    assert "prepared 2 observations, 1 compounds; reference x0001a (1.5 Å)" in result.output
    assert "skipped 1: missing_file" in result.output
    config = (tmp_path / "prep" / "qpocket.yaml").read_text()
    assert "pdb_id: T1" in config and 'allowed_cofactors: "MG|ZN"' in config
    result = CliRunner().invoke(cli, ["prepare", "fragalysis", str(tmp_path / "prep"), str(tmp_path / "x")])
    assert result.exit_code != 0 and "aligned_files" in result.output
