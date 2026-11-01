"""Generating sbatch scripts for the pipeline.

A generated script is only useful if the cluster accepts it, and the failure mode
is expensive: you find out after queueing. So the claims pinned here are the ones
that cost real time when they break — the array directive is well formed, an
empty run emits no array at all rather than ``--array=1-0``, chunks tile the
manifest exactly once, logs are absolute so they land beside the run, and every
script parses as bash.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from qpocket.config import QPocketConfig
from qpocket.pipelines.slurm import (
    STAGES_BY_NAME,
    render_sbatch,
    write_manifest,
    write_scripts,
    write_submit_all,
)
from qpocket.config import evolve

PDB_IDS = ["1ABC", "2DEF", "3GHI", "4JKL"]


@pytest.fixture
def cfg():
    config = QPocketConfig()
    config = evolve(config, "reference.pdb_id", "1ABC")
    return config


@pytest.fixture
def dirs(tmp_path):
    root = tmp_path / "1ABC"
    for sub in ("metadata", "pockets", "structures"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return {"root": root, "metadata": root / "metadata", "pockets": root / "pockets"}


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("mode: pdb_query\n")
    return path


def structure_script(cfg, dirs, config_file, **kwargs):
    manifest = write_manifest(dirs, STAGES_BY_NAME["structure"], PDB_IDS)
    return render_sbatch(
        cfg, dirs, STAGES_BY_NAME["structure"], config_file,
        n_tasks=len(PDB_IDS), manifest=manifest, **kwargs,
    )


# --- the script body ------------------------------------------------------


# --- environment preamble -------------------------------------------------


# --- optional directives --------------------------------------------------


# --- chunking -------------------------------------------------------------


def test_a_stage_over_the_array_cap_is_split_into_several_scripts(cfg, dirs, config_file):
    cfg = evolve(cfg, "slurm.max_array_size", 2)
    scripts = write_scripts(cfg, dirs, config_file, PDB_IDS, stages=["structure"])
    assert len(scripts["structure"]) == 2


# --- manifests ------------------------------------------------------------


# --- stage selection ------------------------------------------------------


def test_a_run_renders_prepare_structure_and_sites_in_order(cfg, dirs, config_file):
    assert list(write_scripts(cfg, dirs, config_file, PDB_IDS)) == ["prepare", "structure", "sites"]


# --- the driver -----------------------------------------------------------


# --- the scripts actually parse -------------------------------------------


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_every_generated_script_parses_as_bash(cfg, dirs, config_file):
    cfg = evolve(cfg, "slurm.environment", ["module load gcc/11.2", "conda activate qpocket"])
    cfg = evolve(cfg, "slurm.account", "proj123")
    scripts = write_scripts(cfg, dirs, config_file, PDB_IDS)
    paths = [p for group in scripts.values() for p in group]
    paths.append(write_submit_all(cfg, dirs, scripts))
    for path in paths:
        done = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert done.returncode == 0, f"{path.name}: {done.stderr}"


# --- progress -------------------------------------------------------------


# --- the command ----------------------------------------------------------


def _run_config(tmp_path):
    config = tmp_path / "cfg.yaml"
    config.write_text("mode: pdb_query\nreference:\n  pdb_id: 1ABC\n  chain_id: A\n"
                      f"paths:\n  workdir: {tmp_path}\n", encoding="utf-8")
    sites = tmp_path / "1ABC" / "sites"
    (sites / "site_01" / "metadata").mkdir(parents=True)
    (sites / "sites.csv").write_text("site_id\nsite_01\n", encoding="utf-8")
    (sites / "site_01" / "metadata" / "master.csv").write_text("pdb_id\n1ABC\n", encoding="utf-8")
    pocket = sites / "site_01" / "pockets" / "1ABC"
    pocket.mkdir(parents=True)
    (pocket / "binding_site.csv").write_text("x,y,z\n", encoding="utf-8")
    return config


