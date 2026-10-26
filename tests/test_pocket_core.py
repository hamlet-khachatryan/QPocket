"""Pure pocket choice and interaction tables."""
import numpy as np
import pandas as pd

from qpocket.core import interactions, pocket


def test_select_cavity_by_centroid_picks_the_first_overlapping_one():
    a = np.array([[0, 0, 0], [1, 0, 0]], float)
    b = np.array([[20, 20, 20], [21, 20, 20]], float)
    assert pocket.select_cavity_by_centroid([b, a], np.array([0.5, 0, 0])) is a
    assert pocket.select_cavity_by_centroid([b], np.array([0.5, 0, 0])) is None


def test_interacting_residue_coords_honours_chain_tags():
    atoms = pd.DataFrame({"residue_name": ["TYR", "TYR"], "residue_number": [795, 795],
                          "chain_id": ["A", "B"], "x_coord": [1.0, 9.0],
                          "y_coord": [0.0, 0.0], "z_coord": [0.0, 0.0]})
    got = pocket.interacting_residue_coords(atoms, ["TYR_795_B|H-bond Donor"])
    assert got.tolist() == [[9.0, 0.0, 0.0]]
    assert pocket.interacting_residue_coords(atoms, ["no separator"]) is None


def test_potential_frequencies_counts_structures_not_entries():
    df = interactions.potential_frequencies({"P1": ["A_1", "A_1", "B_2"], "P2": ["A_1"]})
    assert df.set_index("Residue")["Count"].to_dict() == {"A_1": 2, "B_2": 1}
    assert interactions.potential_frequencies({}) is None


def test_observed_frequencies_weights_type_shares_by_presence():
    df = interactions.observed_frequencies({"P1": ["A_1|Hydrophobic", "A_1|H-bond Donor"], "P2": []})
    row = df.set_index("amino").loc["A_1"]
    assert row["Hydrophobic"] == 25.0 and row["H-bond Donor"] == 25.0


def test_utilization_is_the_engaged_share_of_the_pocket():
    df = interactions.utilization({"P1": ["A_1|Hydrophobic"]}, {"P1": ["A_1", "B_2"], "P2": ["C_3"]})
    assert df.to_dict("records") == [{"pdb_id": "P1", "utilization": 50.0}]
