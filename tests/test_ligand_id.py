"""The ligand's component ID is carried, not re-derived from a filename."""
import json

from qpocket.pipelines.stages import StructureResult


def test_structure_result_round_trips_the_ligand_id():
    result = StructureResult(pdb_id="7KG6", ligand_file="/runs/7KG6_WDG.pdb",
                             ligand_id="WDG", protein_state="BOUND_STATE",
                             potential=["TYR_795_A"],
                             observed=["TYR_795_A|Hydrophobic"],
                             smiles=("c1ccccc1", "WDG"))
    payload = json.loads(json.dumps(result.to_json()))
    assert StructureResult.from_json(payload).ligand_id == "WDG"


def test_a_sidecar_without_the_field_still_loads():
    """Sidecars written before this change must not break a resumed run."""
    payload = {"pdb_id": "4A0D", "ligand_file": None, "protein_state": "UNBOUND_STATE",
               "potential": [], "observed": ["no_ligand_found"], "smiles": None}
    assert StructureResult.from_json(payload).ligand_id is None


def test_apo_structures_record_the_sentinel():
    result = StructureResult(pdb_id="4A0D", ligand_file=None, ligand_id=None,
                             protein_state="UNBOUND_STATE", potential=[],
                             observed=["no_ligand_found"], smiles=None)
    assert result.observed == ["no_ligand_found"]
    assert result.to_json()["ligand_id"] is None


def test_collect_writes_the_ligand_id_column(tmp_path):
    """master.csv gains ligand_id, between ligand_file and protein_state."""
    import json as _json

    import pandas as pd

    from qpocket.pipelines.stages import collect_structure_results

    dirs = {"metadata": tmp_path / "metadata", "screening": tmp_path / "screening",
            "parts": tmp_path / "metadata" / "parts"}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pdb_id": ["7KG6"], "rmsd_to_ref": [0.0],
                  "aligned_file": ["/a/7KG6.pdb"],
                  "cleaned_file": ["/c/7KG6.pdb"]}).to_csv(
        dirs["metadata"] / "master.csv", index=False)
    (dirs["parts"] / "7KG6.json").write_text(_json.dumps(
        StructureResult(pdb_id="7KG6", ligand_file="/l/7KG6_WDG.pdb", ligand_id="WDG",
                        protein_state="REFERENCE", potential=[], observed=[],
                        smiles=None).to_json()), encoding="utf-8")

    from qpocket.config import FeaturesConfig, QPocketConfig
    cfg = QPocketConfig()
    cfg.features = FeaturesConfig()
    collect_structure_results(cfg, dirs)

    columns = list(pd.read_csv(dirs["metadata"] / "master.csv").columns)
    assert columns == ["pdb_id", "rmsd_to_ref", "aligned_file", "cleaned_file",
                       "ligand_file", "ligand_id", "protein_state"]
    assert pd.read_csv(dirs["metadata"] / "master.csv")["ligand_id"][0] == "WDG"
