"""The ligand's component ID is carried, not re-derived from a filename."""
import json

import pandas as pd

from qpocket.core.types import Structure
from qpocket.io import layout


def test_a_record_round_trips_the_ligand_id():
    record = Structure("7KG6", "7KG6.cif", ligand_file="/runs/7KG6_WDG.pdb", ligand_id="WDG",
                       protein_state="BOUND_STATE", potential=("TYR_795_A",),
                       observed=("TYR_795_A|Hydrophobic",), smiles=("c1ccccc1", "WDG"))
    payload = json.loads(json.dumps(record.to_json()))
    assert Structure.from_json(payload).ligand_id == "WDG"


def test_a_sidecar_without_the_field_still_loads():
    """Sidecars written before this change must not break a resumed run."""
    payload = {"pdb_id": "4A0D", "source": "4A0D.cif", "ligand_file": None,
               "protein_state": "UNBOUND_STATE", "potential": [], "observed": ["no_ligand_found"],
               "smiles": None}
    assert Structure.from_json(payload).ligand_id is None


def test_apo_structures_record_the_sentinel(tmp_path):
    from tests.conftest import make_config
    from qpocket.pipelines import structure

    ws = layout.create(str(tmp_path), "run")
    record = Structure("4A0D", "4A0D.cif", protein_state="UNBOUND_STATE", cleaned_file="/nope.pdb")
    out = structure.read_contacts(structure.structure_params(make_config()), ws, record)
    assert out.observed == ("no_ligand_found",)
    assert out.to_json()["ligand_id"] is None


def test_collect_writes_the_ligand_id_column(tmp_path):
    """master.csv carries ligand_id, between ligand_file and protein_state."""
    from qpocket.pipelines.collect import collect_run

    ws = layout.create(str(tmp_path), "run")
    record = Structure("7KG6", "7KG6.cif", 0.0, "/a/7KG6.pdb", cleaned_file="/c/7KG6.pdb",
                       protein_state="REFERENCE", ligand_file="/l/7KG6_WDG.pdb", ligand_id="WDG")
    collect_run(ws, [record], [], ["hbond"], "gzip")
    master = pd.read_csv(ws.metadata / "master.csv")
    assert list(master.columns) == ["pdb_id", "rmsd_to_ref", "aligned_file", "cleaned_file",
                                    "ligand_file", "ligand_id", "protein_state"]
    assert master["ligand_id"][0] == "WDG"
