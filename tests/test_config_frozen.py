"""A loaded or constructed config cannot change; evolve() makes changed copies."""
import dataclasses

import pytest
from omegaconf.errors import ReadonlyConfigError

from qpocket.config import QPocketConfig, evolve, load_qpocket_config


def test_dataclass_config_is_frozen():
    cfg = QPocketConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.mode = "fragment_screen"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.reference.pdb_id = "1ABC"


def test_loaded_config_is_read_only():
    cfg = load_qpocket_config(overrides=["reference.pdb_id=7KG6"])
    with pytest.raises((ReadonlyConfigError, dataclasses.FrozenInstanceError)):
        cfg.reference.pdb_id = "1ABC"


def test_evolve_replaces_a_nested_value_and_leaves_the_original():
    cfg = QPocketConfig()
    changed = evolve(cfg, "reference.pdb_id", "1ABC")
    assert changed.reference.pdb_id == "1ABC"
    assert cfg.reference.pdb_id != "1ABC"


def test_evolve_works_on_a_loaded_config():
    cfg = load_qpocket_config(overrides=["reference.pdb_id=7KG6"])
    changed = evolve(cfg, "reference.pdb_id", "1ABC")
    assert changed.reference.pdb_id == "1ABC" and cfg.reference.pdb_id == "7KG6"


def test_evolve_rejects_an_unknown_path():
    with pytest.raises(AttributeError):
        evolve(QPocketConfig(), "reference.nope", 1)
