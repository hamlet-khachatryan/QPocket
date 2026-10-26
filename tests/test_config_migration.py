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

from qpocket.config import load_qpocket_config
from qpocket.config import LEGACY_KEYS, QPocketConfig, fill_defaults, migrate_legacy


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


def test_ligand_id_is_obsolete():
    cfg = OmegaConf.create({"reference": {"ligand_id": "LKQ", "ligand_chain_id": "A",
                                          "pocket_mode": "bounding_box"}})
    warnings = migrate_legacy(cfg)
    assert any("reference.ligand_id is obsolete" in w for w in warnings)
    assert any("reference.pocket_mode is obsolete" in w for w in warnings)


def test_the_old_slurm_stages_load_with_a_warning(tmp_path):
    """A config written for the apbs/featurize/distance stages must still load, not fail validation."""
    path = tmp_path / "old.yaml"
    path.write_text("reference:\n  pdb_id: 1ABC\nslurm:\n  split_apbs_stage: true\n  stages:\n"
                    "    featurize: {cpus: 1, mem: 8G, time: '04:00:00'}\n"
                    "    distance: {cpus: 16, mem: 32G, time: '06:00:00'}\n", encoding="utf-8")
    cfg, notices = load_with_notices(str(path))
    assert set(cfg.slurm.stages) == {"structure", "site_featurize", "site_cluster"}
    for key in ("slurm.split_apbs_stage", "slurm.stages.featurize", "slurm.stages.distance"):
        assert any(f"{key} is obsolete" in n for n in notices), key


def test_an_obsolete_slurm_stage_names_its_successors():
    cfg = OmegaConf.create({"slurm": {"stages": {"featurize": {"cpus": 1}}}})
    warnings = migrate_legacy(cfg)
    assert ("slurm.stages.featurize is obsolete; set slurm.stages.structure (per-structure chain), "
            "slurm.stages.site_featurize (features/APBS) or slurm.stages.site_cluster instead") in warnings


def test_sites_defaults():
    s = QPocketConfig().sites
    assert (s.min_heavy_atoms, s.cavity_overlap, s.box_padding) == (6, 0.5, 0.0)


def test_save_intermediate_is_obsolete(tmp_path):
    import inspect

    from qpocket.config import IOConfig
    from qpocket.pipelines.run import run_qpocket_pipeline

    path = tmp_path / "old.yaml"
    path.write_text("reference:\n  pdb_id: 1ABC\nio:\n  save_intermediate: true\n", encoding="utf-8")
    _, notices = load_with_notices(str(path))
    assert any("io.save_intermediate is obsolete" in n for n in notices)
    assert not hasattr(IOConfig(), "save_intermediate")
    assert "save_intermediate" not in inspect.signature(run_qpocket_pipeline).parameters


def test_k_range_is_obsolete():
    cfg = OmegaConf.create({"distance": {"clustering": {"shape": {"k_range": [2, 9]}}}})
    assert any("k_range is obsolete" in w for w in migrate_legacy(cfg))


def test_a_config_with_k_range_still_loads(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text("reference:\n  pdb_id: 1ABC\ndistance:\n  clustering:\n    chem:\n"
                    "      method: kmedoids\n      k: 3\n      criterion: count\n      k_range: [2, 6]\n",
                    encoding="utf-8")
    cfg, notices = load_with_notices(str(path))
    assert cfg.distance.clustering.chem.k == 3
    assert any("distance.clustering.chem.k_range is obsolete" in n for n in notices)


def test_a_run_saved_with_the_silhouette_defaults_still_loads(tmp_path):
    import subprocess

    from qpocket.config import load_run_config

    old = subprocess.run(["git", "show", "d9bcb50:conf/config.yaml"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    if old.returncode:
        pytest.skip("commit d9bcb50 is not in this checkout")
    (tmp_path / "metadata").mkdir()
    saved = old.stdout.replace("pdb_id: null", 'pdb_id: "1ABC"', 1)  # a run always has its reference
    (tmp_path / "metadata" / "config.yaml").write_text(saved, encoding="utf-8")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        cfg = load_run_config(tmp_path)
    c = cfg.distance.clustering
    for name in ("shape", "chem"):
        stage = getattr(c, name)
        assert (stage.method, stage.criterion, stage.threshold) == ("hierarchical", "relative", 0.5)
        assert f"distance.clustering.{name} chose k by silhouette" in err.getvalue()
    assert c.single.method == "affinity"


def test_an_explicit_k_means_count(tmp_path):
    path = tmp_path / "k.yaml"
    path.write_text("reference:\n  pdb_id: 1ABC\ndistance:\n  clustering:\n"
                    "    shape:\n      k: 3\n    chem:\n      method: kmedoids\n      k: 3\n",
                    encoding="utf-8")
    cfg, _ = load_with_notices(str(path))
    assert (cfg.distance.clustering.shape.criterion, cfg.distance.clustering.shape.k) == ("count", 3)
    assert (cfg.distance.clustering.chem.criterion, cfg.distance.clustering.chem.k) == ("count", 3)
    cfg = load_qpocket_config(overrides=["reference.pdb_id=1ABC", "distance.clustering.shape.k=4"])
    assert (cfg.distance.clustering.shape.criterion, cfg.distance.clustering.shape.k) == ("count", 4)


def test_an_old_link_cutoff_loads_and_is_dropped(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text("mode: pdb_query\nreference:\n  pdb_id: 1ABC\nsites:\n  link_cutoff: 4.0\n")
    cfg = load_qpocket_config(config_path=str(path))
    assert not hasattr(cfg.sites, "link_cutoff")
