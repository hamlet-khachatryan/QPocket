"""The Exploration page's helpers: per-cluster residue profile, the contact heatmap, the representative picker."""

from qpocket.viz.interactions import RunInteractions
from qpocket.viz.pages import exploration as page


def _run():
    observed = {"A": ("TYR_1_A|HBDonor", "GLY_2_A|Hydrophobic"), "B": ("TYR_1_A|Hydrophobic",), "C": ()}
    potential = {"A": ("TYR_1_A", "GLY_2_A"), "B": ("TYR_1_A",), "C": ("TYR_1_A", "SER_3_A")}
    return RunInteractions(observed, potential, ("A", "B"), ("TYR_1_A", "GLY_2_A", "SER_3_A"),
                           frozenset({"TYR_1_A", "GLY_2_A"}), ("HBDonor", "Hydrophobic"), ())


def test_empty_profile_heatmap_says_so():
    fig = page.residue_heatmap(page.residue_profile(None, {}))
    assert "No observed contacts" in fig.layout.title.text


