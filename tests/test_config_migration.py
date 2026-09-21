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
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from qpocket.pipelines.stages import load_qpocket_config
from qpocket.config import LEGACY_KEYS, fill_defaults, migrate_legacy


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
        ("spectral_distance_matrix.beta", 20.0, "distance.chem_scale"),
        ("spectral_distance_matrix.sigma", 7.0, "distance.sigma"),
        ("spectral_distance_matrix.global_k", 42, "distance.global_k"),
        ("spectral_distance_matrix.affinity_damping", 0.75,
         "distance.clustering.single.damping"),
        ("spectral_distance_matrix.affinity_max_iter", 99,
         "distance.clustering.single.max_iter"),
        ("spectral_distance_matrix.clustering.chem.min_cluster_size", 9,
         "distance.clustering.min_cluster_size"),
        ("fragment_screen.interaction_padding", 8.0,
         "reference.interaction_padding"),
    ],
)
def test_a_renamed_key_keeps_its_value(legacy, value, new_path):
    cfg = OmegaConf.create({})
    OmegaConf.update(cfg, legacy, value, force_add=True)
    migrate_legacy(cfg)
    assert OmegaConf.select(cfg, new_path) == value


@pytest.mark.parametrize("legacy", sorted(LEGACY_KEYS))
def test_every_legacy_key_is_removed_after_migration(legacy):
    """Leaving it behind would make it look like a live setting."""
    cfg = OmegaConf.create({})
    OmegaConf.update(cfg, legacy, 0.5, force_add=True)
    migrate_legacy(cfg)
    assert OmegaConf.select(cfg, legacy, default=None) is None


@pytest.mark.parametrize("legacy", sorted(LEGACY_KEYS))
def test_every_legacy_key_says_something(legacy):
    cfg = OmegaConf.create({})
    OmegaConf.update(cfg, legacy, 0.5, force_add=True)
    assert migrate_legacy(cfg)


def test_a_whole_renamed_block_moves_with_its_contents(tmp_path):
    """The parent rename must not drop settings a child rename already moved."""
    cfg, _ = migrate({"spectral_distance_matrix": {
        "clustering": {"mode": "single", "shape": {"k": 3, "min_cluster_size": 7}}}})
    assert (OmegaConf.select(cfg, "distance.clustering.mode") == "single"
            and OmegaConf.select(cfg, "distance.clustering.shape.k") == 3
            and OmegaConf.select(cfg, "distance.clustering.min_cluster_size") == 7)


def test_an_obsolete_key_is_reported_not_relocated():
    cfg, notices = migrate({"spectral_distance_matrix": {"max_clusters": 10}})
    assert any("max_clusters" in n and "obsolete" in n for n in notices)


def test_the_new_name_wins_when_both_are_set():
    cfg, notices = migrate({
        "spectral_distance_matrix": {"alpha": 0.1},
        "distance": {"shape_weight": 0.9},
    })
    assert OmegaConf.select(cfg, "distance.shape_weight") == 0.9


def test_renames_are_reported_once_per_section_not_once_per_key():
    """A 0.1 config trips a dozen of these; a dozen near-identical lines read as noise."""
    _, notices = migrate({"spectral_distance_matrix": {
        "alpha": 0.5, "beta": 2.0, "sigma": 3.0, "global_k": 4}})
    assert len([n for n in notices if "renamed" in n]) == 1


def test_an_unrecognised_key_warns_but_does_not_raise():
    _, notices = migrate({"reference": {"pdb_id": "1ABC"}, "nonsense": 1})
    assert any("nonsense" in n and "not a recognised setting" in n for n in notices)


def test_a_current_config_produces_no_notices():
    _, notices = migrate({"reference": {"pdb_id": "1ABC"},
                          "distance": {"shape_weight": 0.5}})
    assert notices == []


# --- filling omitted keys --------------------------------------------------


def test_a_setting_the_yaml_omits_is_filled_from_the_schema():
    """The shipped YAML omits `linkage` where the method ignores it."""
    cfg = OmegaConf.create({"distance": {"clustering": {"chem": {"method": "kmedoids"}}}})
    fill_defaults(cfg)
    assert OmegaConf.select(cfg, "distance.clustering.chem.linkage") == "average"


def test_a_whole_omitted_section_is_filled():
    cfg = OmegaConf.create({"mode": "pdb_query"})
    fill_defaults(cfg)
    assert OmegaConf.select(cfg, "slurm.partition") == "compute"


def test_filling_never_overwrites_what_was_set():
    cfg = OmegaConf.create({"distance": {"sigma": 3.0}})
    fill_defaults(cfg)
    assert OmegaConf.select(cfg, "distance.sigma") == 3.0


# --- the configs that already exist ---------------------------------------


EXPERIMENTS = sorted(glob.glob("experiments/*.yaml"))


@pytest.mark.skipif(not EXPERIMENTS, reason="no experiment configs checked in")
@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: Path(p).name)
def test_an_existing_config_still_loads(path):
    """Written against the old names; renaming must not have broken any of them."""
    cfg, _ = load_with_notices(path)
    assert cfg.distance.shape_weight is not None


@pytest.mark.skipif(not EXPERIMENTS, reason="no experiment configs checked in")
@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: Path(p).name)
def test_an_existing_config_reports_at_most_one_rename_notice(path):
    _, notices = load_with_notices(path)
    assert len([n for n in notices if "renamed" in n]) <= 1
