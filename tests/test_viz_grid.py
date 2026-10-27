"""The pocket grid's choices, outside the notebook."""
import numpy as np
import pandas as pd

from qpocket.viz import grid


def _info():
    return pd.DataFrame({
        "pdb_id": ["A", "B", "C", "D"],
        "cluster_id": [0, 0, 1, 1],
        "is_representative": [False, True, False, False],
        "dist_to_exemplar": [0.3, 0.0, 0.5, 0.1],
    })


def test_each_cluster_gets_one_panel():
    rows, fallbacks = grid.representative_rows(_info())
    assert list(rows["pdb_id"]) == ["B", "D"] and fallbacks == [1]


def test_without_distances_the_first_member_stands_in():
    rows, fallbacks = grid.representative_rows(_info().drop(columns="dist_to_exemplar"))
    assert list(rows["pdb_id"]) == ["B", "C"]


def _pocket(offset):
    return pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0] if offset == 0 else [0.0, 0.0],
                         "apbs": [1.0 + offset, 2.0 + offset]})


def test_feature_values_pass_through():
    pockets = [("A", pd.Series({"pdb_id": "A"}), _pocket(0))]
    out = grid.panel_values(pockets, "feature value", "apbs", None, None, "default", 99.0)
    coords, values = out.values["A"]
    assert np.array_equal(values, [1.0, 2.0]) and out.label == "apbs"


def test_a_missing_feature_shows_zeros():
    pockets = [("A", pd.Series({"pdb_id": "A"}), _pocket(0).drop(columns="apbs"))]
    out = grid.panel_values(pockets, "feature value", "apbs", None, None, "default", 99.0)
    assert np.array_equal(out.values["A"][1], [0.0, 0.0])


def test_notes_count_spheres_and_panels():
    panels = [{"n_points": 2, "n_ctx": 0, "n_forced": 0}, {"n_points": 3, "n_ctx": 0, "n_forced": 0}]
    notes = grid.grid_notes(panels, across_clusters=False, fallbacks=[], missing=[],
                            protein_style="none", context_radius=8.0, contact_r=4.0,
                            run_interactions=None, mode="feature value", value_label="apbs", picks=())
    assert notes[-1] == "5 spheres across 2 panels"


def test_difference_on_a_partly_featurised_site_shows_feature_values():
    """A pocket still without its feature column cannot be differenced; values, not a KeyError."""
    pockets = [("A", pd.Series({"pdb_id": "A"}), _pocket(0)),
               ("B", pd.Series({"pdb_id": "B"}), _pocket(1).drop(columns="apbs"))]
    out = grid.panel_values(pockets, "difference from representative", "apbs", pockets[0][1], 1.0, "default", 0.0)
    assert out.label == "apbs" and np.array_equal(out.values["B"][1], [0.0, 0.0])


# --- the cloud picker --------------------------------------------------------------------

def _picker_info():
    import pandas as pd

    return pd.DataFrame({
        "pdb_id": [f"P{i}" for i in range(8)],
        "cluster_id": [0, 0, 0, 1, 1, 2, 2, 2],
        "shape_cluster_id": [0, 0, 0, 1, 1, 2, 2, 2],
        "chem_cluster_id": 0,
        "is_representative": [False, True, False, True, False, False, False, True],
        "protein_state": ["BOUND_STATE", "UNBOUND_STATE"] * 4,
    })


def test_cloud_options_label_every_pocket():
    from qpocket.viz.grid import cloud_options

    options = cloud_options(_picker_info())
    assert len(options) == 8 and options["P1 · S1 · apo"] == "P1"


def test_default_selection_representatives_first():
    from qpocket.viz.grid import default_cloud_selection

    picked = default_cloud_selection(_picker_info(), 2, limit=40)
    assert picked == ["P1", "P3", "P7", "P5", "P6"]
    assert default_cloud_selection(_picker_info(), 2, limit=2) == ["P1", "P3"]


def test_cloud_selection_capped_with_note():
    from qpocket.viz.grid import capped

    kept, note = capped([f"P{i}" for i in range(55)], 40)
    assert len(kept) == 40 and "15" in note
    assert capped(["P1"], 40) == (["P1"], "")
