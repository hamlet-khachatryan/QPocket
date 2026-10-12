"""The chain's records: master rows, sidecars and the reference survive a round trip."""
import pickle

import numpy as np

from qpocket.core.types import AlignmentContext, Reference, Structure


def _done(**kw):
    base = dict(pdb_id="4B1H", source="/in/4B1H.cif", rmsd_to_ref=0.42,
                aligned_file="/r/4B1H_aligned.pdb", cleaned_file="/r/4B1H_clean.pdb",
                protein_state="BOUND_STATE", ligand_file="/r/4B1H_AR6.pdb", ligand_id="AR6",
                protonated_file="/r/p/4B1H.pdb", potential=("TYR_795_A",),
                observed=("TYR_795_A|HBDonor",), smiles=("c1ccccc1", "AR6"))
    base.update(kw)
    return Structure(**base)


def test_master_row_has_todays_columns_in_order():
    assert list(_done().master_row()) == ["pdb_id", "rmsd_to_ref", "aligned_file", "cleaned_file",
                                          "ligand_file", "ligand_id", "protein_state"]


def test_interface_runs_keep_the_chain_mapping_after_aligned_file():
    row = _done(aux_chain_mapping="{'B': 'B'}").master_row()
    assert list(row)[3] == "aux_chain_mapping"


def test_a_record_survives_its_sidecar():
    record = _done(points=np.zeros((2, 3)))
    back = Structure.from_json(record.to_json())
    assert back == record._replace(points=None)


def test_an_old_style_sparse_sidecar_still_loads():
    back = Structure.from_json({"pdb_id": "X", "source": "x.pdb"})
    assert back.potential == () and back.smiles is None and back.protein_state is None


def _reference():
    ctx = AlignmentContext("/r/7KG6_aligned.pdb", "A", "MKV", None, {}, {"B": (1.0, 2.0, 3.0)},
                           {}, False, False)
    record = Structure("7KG6", "/in/7KG6.cif", 0.0, "/r/7KG6_aligned.pdb",
                       cleaned_file="/r/7KG6_clean.pdb")
    return Reference("7KG6", ctx, record, np.array([1.0, 2.0, 3.0]),
                     (np.zeros(3), np.ones(3)))


def test_the_reference_survives_reference_json():
    ref = _reference()
    back = Reference.from_json(ref.to_json())
    assert back.alignment == ref.alignment and back.record == ref.record
    assert np.array_equal(back.centroid, ref.centroid)
    assert all(np.array_equal(a, b) for a, b in zip(back.box, ref.box))


def test_a_reference_without_a_pocket_round_trips():
    ref = _reference()._replace(centroid=None, box=None)
    back = Reference.from_json(ref.to_json())
    assert back.centroid is None and back.box is None


def test_records_pickle_for_worker_processes():
    pickle.loads(pickle.dumps(_reference()))
