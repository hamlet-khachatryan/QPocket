"""A heavy plug-in raises the featurize job's request; nothing else changes."""
import pytest

from qpocket.config import evolve
from qpocket.core.ports import FEATURES
from qpocket.core.registry import Resources
from qpocket.pipelines import slurm
from tests.conftest import make_config


def test_defaults_are_unchanged_for_todays_plugins():
    cfg = make_config()
    res = slurm.stage_resources(cfg, "featurize")
    assert (res.cpus, res.mem, res.time, res.gpu) == (cfg.slurm.defaults.cpus, cfg.slurm.defaults.mem,
                                                      cfg.slurm.defaults.time, 0)


def test_a_heavy_feature_raises_the_featurize_request(monkeypatch):
    plugin = FEATURES.get("template")
    monkeypatch.setitem(FEATURES._plugins, "template",
                        plugin._replace(resources=Resources(cpus=8, mem_gb=32, walltime_min=600, gpu=1)))
    cfg = evolve(make_config(), "features.enabled", ["hbond", "template"])
    res = slurm.stage_resources(cfg, "featurize")
    assert (res.cpus, res.mem, res.time, res.gpu) == (8, "32G", "10:00:00", 1)
    assert slurm.stage_resources(cfg, "collect").cpus == cfg.slurm.defaults.cpus


def test_an_explicit_stage_override_wins(monkeypatch):
    from qpocket.config import SlurmStageResources

    cfg = evolve(make_config(), "slurm.stages", {"featurize": SlurmStageResources(cpus=2, mem="3G",
                                                                                   time="00:10:00")})
    assert slurm.stage_resources(cfg, "featurize").mem == "3G"


def test_geodesic_is_no_longer_generated_but_still_runs_as_a_task():
    from qpocket.pipelines import tasks

    assert "geodesic" not in [s.name for s in slurm.active_stages(make_config())]
    assert "geodesic" in tasks.TASK_NAMES


@pytest.mark.parametrize("text,gb", [("4G", 4), ("4000M", 4000 / 1024), ("2T", 2048), ("512", 512 / 1024)])
def test_memory_strings_parse(text, gb):
    assert slurm._mem_gb(text) == pytest.approx(gb)


@pytest.mark.parametrize("text,minutes", [("01:00:00", 60), ("1-02:00:00", 1560), ("30:00", 30)])
def test_walltimes_parse(text, minutes):
    assert slurm._minutes(text) == minutes
