"""The chain's records: master rows, sidecars and the reference survive a round trip."""
import json
import pickle

from qpocket.core.types import AlignmentContext, Ligand, Reference, Structure

AR6 = Ligand("4B1H", "A", "AR6", 1001, " ", 36, file="/r/4B1H_A_AR6_1001.pdb", smiles="c1ccccc1")


def _done(**kw):
    base = dict(pdb_id="4B1H", source="/in/4B1H.cif", rmsd_to_ref=0.42,
                aligned_file="/r/4B1H_aligned.pdb", cleaned_file="/r/4B1H_clean.pdb",
                protein_state="BOUND_STATE", protonated_file="/r/p/4B1H.pdb", ligands=(AR6,),
                contacts=((AR6.key, ("TYR_795_A|HBDonor",)),), cavities_file="/r/cav/4B1H.csv")
    base.update(kw)
    return Structure(**base)


def test_master_row_has_its_columns_in_order():
    row = _done().master_row()
    assert list(row) == ["pdb_id", "rmsd_to_ref", "aligned_file", "cleaned_file", "protonated_file",
                         "cavities_file", "n_ligands", "protein_state"]
    assert row["n_ligands"] == 1


def test_interface_runs_keep_the_chain_mapping_after_aligned_file():
    row = _done(aux_chain_mapping="{'B': 'B'}").master_row()
    assert list(row)[3] == "aux_chain_mapping"


def test_a_record_survives_its_sidecar():
    record = _done()
    back = Structure.from_json(json.loads(json.dumps(record.to_json())))
    assert back == record and isinstance(back.ligands[0], Ligand)


def test_an_old_style_sparse_sidecar_still_loads():
    back = Structure.from_json({"pdb_id": "X", "source": "x.pdb"})
    assert back.ligands == () and back.contacts == () and back.cavities_file is None


def _reference():
    ctx = AlignmentContext("/r/7KG6_aligned.pdb", "A", "MKV", None, {}, {"B": (1.0, 2.0, 3.0)},
                           {}, False, False)
    record = Structure("7KG6", "/in/7KG6.cif", 0.0, "/r/7KG6_aligned.pdb",
                       cleaned_file="/r/7KG6_clean.pdb")
    return Reference("7KG6", ctx, record)


def test_the_reference_survives_reference_json():
    ref = _reference()
    back = Reference.from_json(ref.to_json())
    assert back == ref


def test_an_old_reference_json_with_a_pocket_still_loads():
    payload = {**_reference().to_json(), "centroid": [1.0, 2.0, 3.0], "box": None}
    assert Reference.from_json(payload) == _reference()


def test_records_pickle_for_worker_processes():
    pickle.loads(pickle.dumps(_reference()))
