"""ProLIF results become the interaction strings the pipeline stores."""
from qpocket.core import detect


def test_labels_invert_prolif_direction():
    """ProLIF names interactions from the ligand; the pipeline names them from the protein."""
    assert detect.LABELS["HBDonor"] == "H-bond Acceptor"
    assert detect.LABELS["HBAcceptor"] == "H-bond Donor"
    assert detect.LABELS["Cationic"] == "Saltbridge (-)"
    assert detect.LABELS["Anionic"] == "Saltbridge (+)"
    assert detect.LABELS["CationPi"] == "Pi-Cation (Pi)"
    assert detect.LABELS["PiCation"] == "Pi-Cation (Cat)"


def test_pistacking_union_is_not_requested():
    """FaceToFace and EdgeToFace already cover it; all three would triple-count."""
    assert "PiStacking" not in detect.INTERACTION_NAMES
    assert "FaceToFace" in detect.INTERACTION_NAMES
    assert "EdgeToFace" in detect.INTERACTION_NAMES


def test_every_requested_name_has_a_label():
    assert set(detect.INTERACTION_NAMES) <= set(detect.LABELS)


def test_format_matches_the_stored_contract():
    assert detect.format_entry("TYR", 795, "A", "FaceToFace") == "TYR_795_A|Pi-stacking (P)"
    assert detect.format_entry("gly", 871, "B", "HBAcceptor") == "GLY_871_B|H-bond Donor"
