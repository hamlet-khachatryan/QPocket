"""Interaction detection is configurable, and bad names are caught at load time."""
import pytest

from qpocket.config import validate_config
from qpocket.core.detect import INTERACTION_NAMES
from tests.conftest import make_config
from qpocket.config import evolve


def test_defaults_match_the_detector():
    assert tuple(make_config().interactions.enabled) == INTERACTION_NAMES


def test_unknown_interaction_is_a_config_error():
    cfg = make_config()
    cfg = evolve(cfg, "interactions.enabled", ["Hydrophobic", "Telepathy"])
    with pytest.raises(ValueError, match="Telepathy"):
        validate_config(cfg)


def test_the_pistacking_union_is_rejected():
    """Requesting it beside FaceToFace and EdgeToFace would count stacks twice."""
    cfg = make_config()
    cfg = evolve(cfg, "interactions.enabled", ["PiStacking"])
    with pytest.raises(ValueError, match="FaceToFace"):
        validate_config(cfg)


def test_an_empty_selection_is_a_config_error():
    cfg = make_config()
    cfg = evolve(cfg, "interactions.enabled", [])
    with pytest.raises(ValueError, match="interactions.enabled"):
        validate_config(cfg)


def test_parameters_are_carried_through():
    cfg = make_config()
    cfg = evolve(cfg, "interactions.parameters", {"HBDonor": {"distance": 4.1}})
    validate_config(cfg)
    assert cfg.interactions.parameters["HBDonor"]["distance"] == 4.1


def test_parameters_for_a_disabled_interaction_are_rejected():
    cfg = make_config()
    cfg = evolve(cfg, "interactions.enabled", ["Hydrophobic"])
    cfg = evolve(cfg, "interactions.parameters", {"HBDonor": {"distance": 4.1}})
    with pytest.raises(ValueError, match="HBDonor"):
        validate_config(cfg)


def test_a_config_file_can_narrow_the_interactions(tmp_path):
    """The loader used to drop this section and run with the defaults."""
    from qpocket.config import load_qpocket_config

    path = tmp_path / "cfg.yaml"
    path.write_text("reference:\n  pdb_id: 7KG6\n  chain_id: A\n"
                    "interactions:\n  enabled: [Hydrophobic, HBDonor]\n"
                    "  parameters:\n    HBDonor: {distance: 3.9}\n", encoding="utf-8")
    cfg = load_qpocket_config(config_path=str(path))
    assert list(cfg.interactions.enabled) == ["Hydrophobic", "HBDonor"]
    assert cfg.interactions.parameters["HBDonor"]["distance"] == 3.9
