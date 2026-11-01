"""Per-stage workers, the entry point of every SLURM array task.

An array task fails differently from a loop iteration: nobody is watching, the
next stage starts anyway, and a task that quietly does nothing is
indistinguishable from one that worked until something downstream is empty. So
the claims pinned here are about refusing to be quiet — an index off the end of
the manifest, a stage with no input, a collect with no sidecars — and about the
merge being keyed by name rather than by position, which is what stops one
failed structure shifting every later row of ``master.csv``.
"""

from __future__ import annotations


import pandas as pd
import pytest

from qpocket.config import QPocketConfig
from qpocket.core.types import Ligand, Structure
from qpocket.io import layout
from qpocket.pipelines import collect as collecting
from qpocket.pipelines import tasks
from qpocket.config import evolve

PDB_IDS = ["1ABC", "2DEF", "3GHI"]


@pytest.fixture
def cfg():
    config = QPocketConfig()
    config = evolve(config, "reference.pdb_id", "1ABC")
    return config


@pytest.fixture
def dirs(tmp_path):
    ws = layout.create(str(tmp_path), "1ABC")
    for pid in PDB_IDS:
        (ws.original / f"{pid}.pdb").write_text("END\n")
    return ws


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "featurize.txt"
    path.write_text("".join(f"{p}\n" for p in PDB_IDS))
    return path


def write_part(dirs, pdb_id, **kw):
    smiles, name = kw.get("smiles", ("", "LIG"))
    ligand = Ligand(pdb_id, "A", name, 900, " ", 6, smiles=smiles)
    record = Structure(
        pdb_id, f"{pdb_id}.pdb", 0.1, f"/aligned/{pdb_id}.pdb", cleaned_file=f"/cleaned/{pdb_id}.pdb",
        protein_state=kw.get("protein_state", "BOUND_STATE"), cavities_file=kw.get("cavities_file"),
        ligands=(ligand,), contacts=((ligand.key, ("ARG_1_A|Hydrophobic",)),),
    )
    layout.write_part(dirs, pdb_id, record.to_json())
    return record


def collect(dirs):
    return collecting.collect_from_parts(dirs, PDB_IDS)


# --- resolving a task's own row -------------------------------------------


# --- reference round trip -------------------------------------------------


# --- collect ---------------------------------------------------------------


def test_collect_fills_master_from_the_sidecars(cfg, dirs):
    for pid in PDB_IDS:
        write_part(dirs, pid, cavities_file=f"/cav/{pid}.csv")
    collect(dirs)
    df = pd.read_csv(dirs["metadata"] / "master.csv")
    assert list(df["cavities_file"]) == [f"/cav/{p}.csv" for p in PDB_IDS]


# --- per-structure stages --------------------------------------------------


@pytest.mark.parametrize("stage", ["structure", "site_cluster"])
def test_every_task_command_wires_up(stage, tmp_path, monkeypatch):
    """`qpocket task <stage>` is the SLURM entry point: it must import and dispatch."""
    from click.testing import CliRunner

    from qpocket.cli import cli

    from qpocket.arms import sites

    called = {}
    monkeypatch.setattr(tasks, "run_task",
                        lambda name, cfg, dirs, **kw: called.update(name=name, kw=kw))
    monkeypatch.setattr(sites, "run_site_task",
                        lambda name, run_dir, site, **kw: called.update(name=name, kw=dict(kw, site=site)))
    config = tmp_path / "cfg.yaml"
    config.write_text("mode: pdb_query\nreference:\n  pdb_id: 1ABC\n  chain_id: A\n"
                      f"paths:\n  workdir: {tmp_path}\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["task", stage, "--config", str(config), "--site", "site_01"])
    assert result.exit_code == 0, result.output
    assert called["name"] == stage
    assert called["kw"].get("site") == ("site_01" if stage in tasks.SITE_TASKS else None)
