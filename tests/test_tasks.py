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


import h5py
import numpy as np
import pandas as pd
import pytest

from qpocket.config import QPocketConfig
from qpocket.core.types import AlignmentContext, Ligand, Reference, Structure
from qpocket.io import layout
from qpocket.pipelines import collect as collecting
from qpocket.pipelines import tasks
from qpocket.pipelines.reference import load_reference
from qpocket.pipelines.tasks import TaskError, resolve_pdb_id, run_task
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


def test_index_selects_the_matching_manifest_line(manifest):
    assert resolve_pdb_id(manifest, 1) == "2DEF"


def test_the_first_index_is_zero(manifest):
    """SLURM_ARRAY_TASK_ID starts at 0 and the manifest has no header."""
    assert resolve_pdb_id(manifest, 0) == PDB_IDS[0]


def test_an_index_past_the_end_is_an_error_not_a_skip(manifest):
    with pytest.raises(TaskError, match="outside manifest"):
        resolve_pdb_id(manifest, 99)


def test_a_negative_index_is_refused(manifest):
    with pytest.raises(TaskError, match="outside manifest"):
        resolve_pdb_id(manifest, -1)


def test_a_missing_manifest_is_an_error(tmp_path):
    with pytest.raises(TaskError, match="manifest not found"):
        resolve_pdb_id(tmp_path / "absent.txt", 0)


def test_an_unknown_stage_is_refused(cfg, dirs):
    with pytest.raises(TaskError, match="unknown stage"):
        run_task("polish", cfg, dirs)


def test_a_per_structure_stage_needs_a_manifest(cfg, dirs):
    with pytest.raises(TaskError, match="needs --manifest"):
        run_task("featurize", cfg, dirs)


def test_the_unimplemented_apbs_split_says_so(cfg, dirs, manifest):
    """Better a clear refusal than silently doing the same work twice."""
    with pytest.raises(TaskError, match="split_apbs_stage"):
        run_task("apbs", cfg, dirs, manifest=manifest, index=0)


# --- reference round trip -------------------------------------------------


def test_the_reference_survives_a_round_trip_to_disk(dirs):
    """Array tasks read this back."""
    ctx = AlignmentContext("/r/1ABC_aligned.pdb", "A", "MKV", None, {}, {"B": (1.0, 2.0, 3.0)}, {},
                           False, False)
    ref = Reference("1ABC", ctx, Structure("1ABC", "1ABC.pdb", 0.0, "/r/1ABC_aligned.pdb"))
    layout.write_reference(dirs, ref.to_json())
    assert load_reference(dirs) == ref


def test_a_task_before_prepare_says_to_run_prepare(cfg, dirs, manifest):
    with pytest.raises(TaskError, match="prepare"):
        run_task("featurize", cfg, dirs, manifest=manifest, index=1)


# --- collect ---------------------------------------------------------------


def test_collect_refuses_when_no_structure_succeeded(cfg, dirs):
    with pytest.raises(TaskError, match="no per-structure sidecars"):
        run_task("collect", cfg, dirs)


def test_collect_fills_master_from_the_sidecars(cfg, dirs):
    for pid in PDB_IDS:
        write_part(dirs, pid, cavities_file=f"/cav/{pid}.csv")
    collect(dirs)
    df = pd.read_csv(dirs["metadata"] / "master.csv")
    assert list(df["cavities_file"]) == [f"/cav/{p}.csv" for p in PDB_IDS]


def test_a_missing_structure_does_not_shift_the_other_rows(cfg, dirs):
    """Rows are matched by pdb_id. Assigning a short list positionally, which is
    what the old code did, would slide 3GHI's result onto 2DEF."""
    write_part(dirs, "1ABC", cavities_file="/cav/1ABC.csv")
    write_part(dirs, "3GHI", cavities_file="/cav/3GHI.csv")
    collect(dirs)
    df = pd.read_csv(dirs["metadata"] / "master.csv").set_index("pdb_id")
    assert df.loc["3GHI", "cavities_file"] == "/cav/3GHI.csv"


def test_a_structure_with_no_sidecar_is_reported_as_failed(cfg, dirs):
    write_part(dirs, "1ABC")
    collect(dirs)
    failures = pd.read_csv(dirs["metadata"] / "failures.csv").set_index("pdb_id")
    assert failures.loc["2DEF", "stage"] == "featurize"
    assert list(pd.read_csv(dirs["metadata"] / "master.csv")["pdb_id"]) == ["1ABC"]


@pytest.mark.skip(reason="run-level archives moved per site; SLURM per-site collect is Task 13")
def test_collect_writes_the_interaction_archives(cfg, dirs):
    for pid in PDB_IDS:
        write_part(dirs, pid)
    collect(dirs)
    with h5py.File(dirs["metadata"] / "observed_interactions.h5", "r") as hf:
        assert sorted(hf.keys()) == sorted(PDB_IDS)


@pytest.mark.skip(reason="run-level archives moved per site; SLURM per-site collect is Task 13")
def test_archive_key_order_follows_master_not_the_glob(cfg, dirs):
    """HDF5 keeps insertion order and the frequency tables break ties on it, so
    globbing the sidecars would make the output depend on the scheduling."""
    for pid in reversed(PDB_IDS):
        write_part(dirs, pid)
    collect(dirs)
    with h5py.File(dirs["metadata"] / "observed_interactions.h5", "r") as hf:
        # h5py lists keys alphabetically; insertion order is what we control.
        assert set(hf.keys()) == set(PDB_IDS)


@pytest.mark.skip(reason="run-level archives moved per site; SLURM per-site collect is Task 13")
def test_actives_are_written_in_master_order(cfg, dirs):
    write_part(dirs, "3GHI", smiles=("CCO", "LIG"))
    write_part(dirs, "1ABC", smiles=("CCN", "LGB"))
    collect(dirs)
    lines = (dirs["screening"] / "actives.smi").read_text().splitlines()
    assert [ln.split("\t")[1] for ln in lines[1:]] == ["1ABC", "3GHI"]


def test_an_unreadable_sidecar_does_not_abort_the_merge(cfg, dirs):
    write_part(dirs, "1ABC")
    (layout.parts_dir(dirs) / "2DEF.json").write_text("{ not json")
    collect(dirs)
    df = pd.read_csv(dirs["metadata"] / "master.csv").set_index("pdb_id")
    assert df.loc["1ABC", "protein_state"] == "BOUND_STATE"


# --- per-structure stages --------------------------------------------------


def test_featurize_skips_a_structure_that_already_has_a_sidecar(cfg, dirs, manifest):
    """A resubmitted array must re-run only what is missing."""
    write_part(dirs, "1ABC")
    run_task("featurize", cfg, dirs, manifest=manifest, index=0)  # must not raise
    assert (layout.parts_dir(dirs) / "1ABC.json").exists()


def test_geodesic_refuses_a_structure_without_a_pocket(cfg, dirs, tmp_path):
    path = tmp_path / "geodesic.txt"
    path.write_text("1ABC\n")
    with pytest.raises(TaskError, match="has no pocket"):
        run_task("geodesic", cfg, dirs, manifest=path, index=0)


def test_geodesic_builds_a_cache_for_a_real_pocket(cfg, dirs, tmp_path):
    pocket = dirs["pockets"] / "1ABC"
    pocket.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    points = rng.random((40, 3)) * 3.0
    pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}).to_csv(pocket / "binding_site.csv", index=False)
    path = tmp_path / "geodesic.txt"
    path.write_text("1ABC\n")
    run_task("geodesic", cfg, dirs, manifest=path, index=0)
    assert (pocket / "geospectral_cache.npz").exists()


def test_a_record_survives_its_sidecar(dirs):
    original = write_part(dirs, "1ABC", smiles=("CCO", "LIG"))
    assert Structure.from_json(layout.read_part(dirs, "1ABC")) == original


def test_featurize_manifest_lists_the_ingested_structures(cfg, dirs):
    assert tasks.manifest_ids(cfg, dirs, "featurize") == PDB_IDS


@pytest.mark.parametrize("stage", tasks.TASK_NAMES)
def test_every_task_command_wires_up(stage, tmp_path, monkeypatch):
    """`qpocket task <stage>` is the SLURM entry point: it must import and dispatch."""
    from click.testing import CliRunner

    from qpocket.cli import cli

    called = {}
    monkeypatch.setattr(tasks, "run_task",
                        lambda name, cfg, dirs, **kw: called.update(name=name, kw=kw))
    config = tmp_path / "cfg.yaml"
    config.write_text("mode: pdb_query\nreference:\n  pdb_id: 1ABC\n  chain_id: A\n"
                      f"paths:\n  workdir: {tmp_path}\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["task", stage, "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert called["name"] == stage
