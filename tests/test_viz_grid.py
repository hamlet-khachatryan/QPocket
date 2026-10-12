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
