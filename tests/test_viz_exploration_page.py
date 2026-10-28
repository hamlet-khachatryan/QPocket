"""The Exploration page's helpers: per-cluster residue profile, the contact heatmap, the representative picker."""
import pandas as pd

from qpocket.viz.interactions import RunInteractions
from qpocket.viz.pages import exploration as page


def _run():
    observed = {"A": ("TYR_1_A|HBDonor", "GLY_2_A|Hydrophobic"), "B": ("TYR_1_A|Hydrophobic",), "C": ()}
    potential = {"A": ("TYR_1_A", "GLY_2_A"), "B": ("TYR_1_A",), "C": ("TYR_1_A", "SER_3_A")}
    return RunInteractions(observed, potential, ("A", "B"), ("TYR_1_A", "GLY_2_A", "SER_3_A"),
                           frozenset({"TYR_1_A", "GLY_2_A"}), ("HBDonor", "Hydrophobic"), ())


def test_residue_profile_percentages():
    profile = page.residue_profile(_run(), {"S1": ["A", "C"], "S2": ["B"]})
    assert list(profile.columns) == ["cluster", "residue", "contacted_pct", "lining_pct", "n_members", "n_with_contacts"]
    row = profile.set_index(["cluster", "residue"])
    assert row.loc[("S1", "TYR_1_A"), "contacted_pct"] == 100.0   # A is S1's only holo member
    assert row.loc[("S1", "TYR_1_A"), "lining_pct"] == 100.0
    assert row.loc[("S1", "SER_3_A"), "contacted_pct"] == 0.0 and row.loc[("S1", "SER_3_A"), "lining_pct"] == 50.0
    assert ("S2", "GLY_2_A") not in row.index                     # neither contacted nor lining in S2
    assert row.loc[("S1", "GLY_2_A"), "n_members"] == 2 and row.loc[("S1", "GLY_2_A"), "n_with_contacts"] == 1
    assert profile["cluster"].drop_duplicates().tolist() == ["S1", "S2"]


def test_residue_profile_without_archives_is_empty():
    profile = page.residue_profile(None, {"S1": ["A"]})
    assert profile.empty and "contacted_pct" in profile.columns


def test_heatmap_keeps_the_top_residues():
    fig = page.residue_heatmap(page.residue_profile(_run(), {"S1": ["A", "C"], "S2": ["B"]}), top=1)
    heat = fig.data[0]
    assert list(heat.x) == ["TYR_1_A"] and list(heat.y) == ["S1", "S2"]
    assert heat.zmin == 0 and heat.zmax == 100


def test_empty_profile_heatmap_says_so():
    fig = page.residue_heatmap(page.residue_profile(None, {}))
    assert "No observed contacts" in fig.layout.title.text


def test_representative_options_put_the_current_first():
    info = pd.DataFrame({"pdb_id": ["A", "B", "C"], "cluster_id": [0, 0, 1], "is_representative": [False, True, True],
                         "dist_to_exemplar": [0.0012, 0.0, 0.0], "protein_state": ["BOUND_STATE", "UNBOUND_STATE", "x"]})
    options = page.representative_options(info, 0)
    assert list(options.values()) == ["B", "A"]
    assert list(options)[1] == "A · holo · d=0.0012"


def test_an_unclustered_site_has_no_clusters_to_pick():
    info = pd.DataFrame({"pdb_id": ["A", "B"], "cluster_id": [-1, -1], "is_representative": [False, False],
                         "protein_state": ["BOUND_STATE", "x"]})        # no dist_to_exemplar, as before clustering
    assert page.representative_options(info, -1) == {}
    assert page.cluster_members(info, {-1: "S0"}) == {}
