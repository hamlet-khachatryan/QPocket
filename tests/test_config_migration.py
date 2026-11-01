"""Renamed configuration keys, and the configs written before they moved.

Renaming a config key is cheap. Silently ignoring the old one is not: the run
proceeds with a default the user thought they had overridden, and nothing says
so. Every rename here is therefore required to keep working, to say once that it
has moved, and to land on the same effective value as the new name would.

The load-every-experiment-config test is the real guarantee: those files were
written against the old names and must not need touching.
"""

from __future__ import annotations

import contextlib
import glob
import io

import pytest
from omegaconf import OmegaConf

from qpocket.config import load_qpocket_config
from qpocket.config import migrate_legacy


def migrate(payload):
    cfg = OmegaConf.create(payload)
    return cfg, migrate_legacy(cfg)


def load_with_notices(path):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        cfg = load_qpocket_config(config_path=str(path))
    return cfg, [ln for ln in err.getvalue().splitlines() if ln.strip()]


# --- each rename lands in the right place ---------------------------------


@pytest.mark.parametrize(
    "legacy, value, new_path",
    [
        ("spectral_distance_matrix.alpha", 0.5, "distance.shape_weight"),
    ],
)
def test_a_renamed_key_keeps_its_value(legacy, value, new_path):
    cfg = OmegaConf.create({})
    OmegaConf.update(cfg, legacy, value, force_add=True)
    migrate_legacy(cfg)
    assert OmegaConf.select(cfg, new_path) == value


def test_an_unrecognised_key_warns_but_does_not_raise():
    _, notices = migrate({"reference": {"pdb_id": "1ABC"}, "nonsense": 1})
    assert any("nonsense" in n and "not a recognised setting" in n for n in notices)


# --- filling omitted keys --------------------------------------------------


# --- the configs that already exist ---------------------------------------


EXPERIMENTS = sorted(glob.glob("experiments/*.yaml"))


def test_old_criterion_migrates_to_percentile(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text("mode: pdb_query\nreference:\n  pdb_id: 1ABC\ndistance:\n  clustering:\n    shape:\n"
                    "      criterion: relative\n      threshold: 0.5\n")
    cfg = load_qpocket_config(config_path=str(path))
    assert cfg.distance.clustering.shape.percentile == 15.0
    notes = migrate_legacy(OmegaConf.create({"distance": {"clustering": {"shape": {
        "criterion": "relative", "threshold": 0.5}}}}))
    assert any("percentile" in note for note in notes)


