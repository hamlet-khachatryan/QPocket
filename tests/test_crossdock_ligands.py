"""Actives are deduplicated by structure and keep every crystal copy."""
import pandas as pd

from qpocket.arms.screen.ligands import distinct_actives, prepare_ligands


def _inputs(tmp_path):
    smi = tmp_path / "actives.smi"
    smi.write_text("SMILES\tPDB_ID\tLigand_Name\n"
                   "OCC\t1AAA\tETH\n"
                   "C(O)C\t2BBB\tETH\n"          # same molecule, written differently
                   "c1ccccc1\t3CCC\tBNZ\n"
                   "CCN\t4DDD\tETH\n")           # different molecule, same code
    master = tmp_path / "master.csv"
    pd.DataFrame({"pdb_id": ["1AAA", "2BBB", "3CCC", "4DDD"],
                  "ligand_file": [str(tmp_path / f"{p}_L.pdb") for p in ("1AAA", "2BBB", "3CCC", "4DDD")]}
                 ).to_csv(master, index=False)
    return smi, master


def test_one_row_per_molecule_with_every_crystal_copy(tmp_path):
    actives = distinct_actives(*_inputs(tmp_path))
    assert len(actives) == 3 and "SMILES" not in set(actives["smiles"])
    eth = actives[actives["canonical"] == "CCO"].iloc[0]
    assert eth["crystal_files"].split(";") == [str(tmp_path / "1AAA_L.pdb"), str(tmp_path / "2BBB_L.pdb")]


def test_ids_stay_unique_when_codes_collide(tmp_path):
    ids = list(distinct_actives(*_inputs(tmp_path))["active_id"])
    assert ids == ["ETH", "BNZ", "ETH_2"]


def test_preparation_skips_failures(tmp_path, monkeypatch):
    from qpocket.arms.screen import ligands
    from qpocket.screen_config import ScreenConfig

    def fake_prepare(smiles, out, cfg, ligand_id):
        if smiles == "c1ccccc1":
            return False  # this one fails to prepare
        out.write_text("x")
        return True

    monkeypatch.setattr(ligands, "prepare_ligand_pdbqt", fake_prepare)
    done = prepare_ligands(distinct_actives(*_inputs(tmp_path)), tmp_path / "ligands", ScreenConfig())
    assert sorted(done) == ["ETH", "ETH_2"] and done["ETH"] == tmp_path / "ligands" / "ETH.pdbqt"
