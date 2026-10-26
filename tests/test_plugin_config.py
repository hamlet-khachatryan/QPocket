"""Plug-ins are chosen and tuned from config; bad names and options fail at load time."""
import numpy as np
import pytest

from qpocket.config import evolve, plugin_params, validate_config
from qpocket.core.ports import FEATURES
from tests.conftest import make_config


def test_defaults_choose_todays_plugins():
    cfg = make_config()
    assert cfg.charges.use == "pdb2pqr" and list(cfg.contacts.use) == ["prolif"]


def test_options_become_the_plugins_params():
    params = plugin_params(FEATURES, "template", {"template": {"value": 2.5}})
    assert params.value == 2.5


def test_an_unknown_option_is_a_config_error():
    with pytest.raises(ValueError, match="template.*nope"):
        plugin_params(FEATURES, "template", {"template": {"nope": 1}})


@pytest.mark.parametrize("path,value,match", [
    ("charges.use", "xtb", "charges.use"),
    ("contacts.use", ["prolif", "telepathy"], "contacts.use"),
    ("features.options", {"template": {"nope": 1}}, "features.options"),
])
def test_validation_names_the_bad_key(path, value, match):
    with pytest.raises(ValueError, match=match):
        validate_config(evolve(make_config(), path, value))


def test_feature_options_reach_the_channel():
    from qpocket.core.features import pocket_features
    from qpocket.types import StructureData

    out = pocket_features(np.zeros((3, 3)), StructureData.empty(), np.zeros(3), ["template"],
                          options={"template": {"value": 4.0}})
    assert np.all(out["template"] == 4.0)


def test_the_chain_reads_its_plugins_from_config():
    from qpocket.pipelines.structure import structure_params

    p = structure_params(make_config())
    assert p.contacts == ("prolif",) and set(p.contact_params) == {"prolif"}


def test_the_yaml_declares_both_sections():
    from qpocket.config import load_qpocket_config

    cfg = load_qpocket_config(overrides=["reference.pdb_id=7KG6", "charges.use=pdb2pqr",
                                         "contacts.use=[prolif]"])
    assert cfg.charges.use == "pdb2pqr"


def test_options_for_an_enabled_plugin_are_checked():
    cfg = evolve(make_config(), "features.enabled", ["hbond", "template"])
    with pytest.raises(ValueError, match="features.options.*nope"):
        validate_config(evolve(cfg, "features.options", {"template": {"nope": 1}}))
    validate_config(evolve(cfg, "features.options", {"template": {"value": 2.0}}))
