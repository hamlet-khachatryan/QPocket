"""actives.smi is the screening input, so it must list every ligand, correctly.

Two defects it had: only the reference structure contributed a row, and the
SMILES came from MolFromPDBBlock, which bonds by proximity and makes every bond
single — so a xanthine came out as an alcohol.
"""
from pathlib import Path

import pytest
from rdkit import Chem

REPO = Path(__file__).resolve().parents[1]
LIGANDS = REPO / "runs/parg-query/7KG6/structures/extracted_ligands"


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else ""


@pytest.mark.skipif(not (LIGANDS / "7KG6_WDG.pdb").exists(),
                    reason="needs an extracted ligand to read")
def test_the_smiles_carries_bond_orders():
    """The ligand is a xanthine: two carbonyls and an aromatic ring, not alcohols."""
    from qpocket.io.interactions import ligand_smiles

    found = _canonical(ligand_smiles(LIGANDS / "7KG6_WDG.pdb"))
    assert found, "no SMILES produced"
    assert "=O" in found, f"no double bonds: {found}"
    assert found == _canonical("Cn1c(=O)c2[nH]c(SCCN3CCC(C(=O)O)CC3)nc2n(C)c1=O")


@pytest.mark.skipif(not (LIGANDS / "7KG6_WDG.pdb").exists(),
                    reason="needs an extracted ligand to read")
def test_the_smiles_is_the_neutral_parent():
    """Downstream screening protonates for itself; the list stores the parent."""
    from qpocket.io.interactions import ligand_smiles

    assert "[O-]" not in ligand_smiles(LIGANDS / "7KG6_WDG.pdb")
    assert "+]" not in ligand_smiles(LIGANDS / "7KG6_WDG.pdb")


def test_an_unreadable_ligand_gives_an_empty_string(tmp_path):
    from qpocket.io.interactions import ligand_smiles

    empty = tmp_path / "nothing.pdb"
    empty.write_text("END\n", encoding="utf-8")
    assert ligand_smiles(empty) == ""


@pytest.mark.golden
def test_every_bound_structure_reaches_actives_smi(tmp_path_factory):
    """Only the reference used to contribute a row."""
    import shutil
    import subprocess
    import sys

    import pandas as pd

    source = REPO / "experiments/PARG/7KG6/structures/original"
    if not source.exists() or shutil.which("uv") is None:
        pytest.skip("needs the bundled PARG structures and uv")
    workdir = tmp_path_factory.mktemp("actives")
    subprocess.run(
        [str(Path(sys.executable).parent / "qpocket"), "run",
         "--mode", "fragment_screen", "--input-type", "paths",
         "--reference-path", str(source / "7KG6.cif"),
         "--structure-paths", str(source / "7KG6.cif"),
         "--structure-paths", str(source / "4B1H.cif"),
         "--chain-id", "A", "--workdir", str(workdir)],
        cwd=REPO, capture_output=True, text=True, timeout=2400,
    )
    run = workdir / "7KG6"
    master = pd.read_csv(run / "metadata/master.csv")
    bound = set(master.loc[master["protein_state"] == "BOUND_STATE", "pdb_id"].astype(str))
    actives = pd.read_csv(run / "screening/actives.smi", sep="\t")
    assert set(actives["PDB_ID"].astype(str)) == bound, "a bound structure is missing"
    assert all("=O" in s for s in actives.loc[actives["Ligand_Name"] == "WDG", "SMILES"]), "bond orders lost"
