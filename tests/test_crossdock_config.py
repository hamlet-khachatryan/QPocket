"""Cross-docking settings: defaults, YAML, validation."""
import pytest

from qpocket.screen_config import ScreenConfig, load_screen_config, validate_screen_config


def test_defaults_match_the_spec():
    cfg = ScreenConfig()
    assert (cfg.analysis.rmsd_threshold, cfg.analysis.top_k) == (2.0, 3)
    assert (cfg.receptor.repair, cfg.receptor.loop_radius, cfg.receptor.use) == (True, 10.0, "pdb2pqr")
    assert (cfg.crossdock.box_mode, cfg.crossdock.box_padding) == ("representative_pockets", 5.0)
    assert cfg.docking.engine == "quickvina" and cfg.docking.engine_path is None


def test_the_packaged_yaml_loads_the_same_defaults():
    cfg = load_screen_config()
    assert cfg.crossdock.box_mode == "representative_pockets" and cfg.analysis.top_k == 3


@pytest.mark.parametrize("section,field,value,match", [
    ("crossdock", "box_mode", "everywhere", "crossdock.box_mode"),
    ("crossdock", "box_padding", -1.0, "crossdock.box_padding"),
    ("receptor", "loop_radius", -2.0, "receptor.loop_radius"),
    ("analysis", "top_k", 0, "analysis.top_k"),
])
def test_bad_values_are_named(section, field, value, match):
    cfg = ScreenConfig()
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(ValueError, match=match):
        validate_screen_config(cfg)
