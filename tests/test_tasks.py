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

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from qpocket.pipelines import api, tasks
from qpocket.pipelines.api import ExtractionContext, StructureResult
from qpocket.config import QPocketConfig
from qpocket.pipelines.tasks import TaskError, resolve_pdb_id, run_task

PDB_IDS = ["1ABC", "2DEF", "3GHI"]


@pytest.fixture
def cfg():
    config = QPocketConfig()
    config.reference.pdb_id = "1ABC"
    return config


@pytest.fixture
def dirs(tmp_path):
    root = tmp_path / "1ABC"
    made = {
        "root": root,
        "metadata": root / "metadata",
        "pockets": root / "pockets",
        "screening": root / "screening",
        "extracted_ligands": root / "structures" / "extracted_ligands",
        "original": root / "structures" / "original",
    }
    for path in made.values():
        path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "pdb_id": PDB_IDS,
        "rmsd_to_ref": [0.0, 0.4, 0.5],
        "aligned_file": [f"/aligned/{p}.pdb" for p in PDB_IDS],
        "cleaned_file": [f"/cleaned/{p}.pdb" for p in PDB_IDS],
    }).to_csv(made["metadata"] / "master.csv", index=False)
    return made


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "featurize.txt"
    path.write_text("".join(f"{p}\n" for p in PDB_IDS))
    return path


def write_part(dirs, pdb_id, **kw):
    result = StructureResult(
        pdb_id=pdb_id,
        ligand_file=kw.get("ligand_file"),
        protein_state=kw.get("protein_state", "BOUND_STATE"),
        potential=kw.get("potential", [f"ARG_1_A"]),
        observed=kw.get("observed", [f"ARG_1_A|Hydrophobic"]),
        smiles=kw.get("smiles"),
    )
    (api.parts_dir(dirs) / f"{pdb_id}.json").write_text(json.dumps(result.to_json()))
    return result


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


# --- context round trip ----------------------------------------------------


def test_the_reference_geometry_survives_a_round_trip_to_disk(dirs):
    """Array tasks read this back; a lossy round trip would silently move pockets."""
    context = ExtractionContext(
        ref_centroid=np.array([1.0, 2.0, 3.0]),
        bbox_center=None,
        bbox_dims=None,
        ref_pocket_bbox=(np.array([0.0, 0.0, 0.0]), np.array([4.0, 5.0, 6.0])),
    )
    api.save_extraction_context(dirs, context)
    back = api.load_extraction_context(dirs)
    assert np.allclose(back.ref_pocket_bbox[1], [4.0, 5.0, 6.0])


def test_a_context_with_no_reference_pocket_round_trips_as_none(dirs):
    api.save_extraction_context(dirs, ExtractionContext(None, None, None, None))
    assert api.load_extraction_context(dirs).ref_pocket_bbox is None


def test_a_task_before_prepare_says_to_run_prepare(dirs):
    with pytest.raises(FileNotFoundError, match="prepare"):
        api.load_extraction_context(dirs)


# --- collect ---------------------------------------------------------------


def test_collect_refuses_when_no_structure_succeeded(cfg, dirs):
    with pytest.raises(TaskError, match="no per-structure sidecars"):
        run_task("collect", cfg, dirs)


def test_collect_fills_master_from_the_sidecars(cfg, dirs):
    for pid in PDB_IDS:
        write_part(dirs, pid, ligand_file=f"/lig/{pid}.pdb")
    api.collect_structure_results(cfg, dirs)
    df = pd.read_csv(dirs["metadata"] / "master.csv")
    assert list(df["ligand_file"]) == [f"/lig/{p}.pdb" for p in PDB_IDS]


def test_a_missing_structure_does_not_shift_the_other_rows(cfg, dirs):
    """Rows are matched by pdb_id. Assigning a short list positionally, which is
    what the old code did, would slide 3GHI's result onto 2DEF."""
    write_part(dirs, "1ABC", ligand_file="/lig/1ABC.pdb")
    write_part(dirs, "3GHI", ligand_file="/lig/3GHI.pdb")
    api.collect_structure_results(cfg, dirs)
    df = pd.read_csv(dirs["metadata"] / "master.csv").set_index("pdb_id")
    assert df.loc["3GHI", "ligand_file"] == "/lig/3GHI.pdb"


def test_a_structure_with_no_sidecar_is_marked_failed(cfg, dirs):
    write_part(dirs, "1ABC")
    api.collect_structure_results(cfg, dirs)
    df = pd.read_csv(dirs["metadata"] / "master.csv").set_index("pdb_id")
    assert df.loc["2DEF", "protein_state"] == "FAILED"


def test_collect_writes_the_interaction_archives(cfg, dirs):
    for pid in PDB_IDS:
        write_part(dirs, pid)
    api.collect_structure_results(cfg, dirs)
    with h5py.File(dirs["metadata"] / "observed_interactions.h5", "r") as hf:
        assert sorted(hf.keys()) == sorted(PDB_IDS)


def test_archive_key_order_follows_master_not_the_glob(cfg, dirs):
    """HDF5 keeps insertion order and the frequency tables break ties on it, so
    globbing the sidecars would make the output depend on the scheduling."""
    for pid in reversed(PDB_IDS):
        write_part(dirs, pid)
    api.collect_structure_results(cfg, dirs)
    with h5py.File(dirs["metadata"] / "observed_interactions.h5", "r") as hf:
        # h5py lists keys alphabetically; insertion order is what we control.
        assert set(hf.keys()) == set(PDB_IDS)


def test_actives_are_written_in_master_order(cfg, dirs):
    write_part(dirs, "3GHI", smiles=("CCO", "LIG"))
    write_part(dirs, "1ABC", smiles=("CCN", "LGB"))
    api.collect_structure_results(cfg, dirs)
    lines = (dirs["screening"] / "actives.smi").read_text().splitlines()
    assert [ln.split("\t")[1] for ln in lines[1:]] == ["1ABC", "3GHI"]


def test_an_unreadable_sidecar_does_not_abort_the_merge(cfg, dirs):
    write_part(dirs, "1ABC")
    (api.parts_dir(dirs) / "2DEF.json").write_text("{ not json")
    api.collect_structure_results(cfg, dirs)
    df = pd.read_csv(dirs["metadata"] / "master.csv").set_index("pdb_id")
    assert df.loc["1ABC", "protein_state"] == "BOUND_STATE"


# --- per-structure stages --------------------------------------------------


def test_featurize_skips_a_structure_that_already_has_a_sidecar(cfg, dirs, manifest):
    """A resubmitted array must re-run only what is missing."""
    write_part(dirs, "1ABC")
    run_task("featurize", cfg, dirs, manifest=manifest, index=0)  # must not raise
    assert (api.parts_dir(dirs) / "1ABC.json").exists()


def test_geodesic_refuses_a_pocket_that_was_never_featurised(cfg, dirs, tmp_path):
    path = tmp_path / "geodesic.txt"
    path.write_text("1ABC\n")
    with pytest.raises(TaskError, match="not featurised"):
        run_task("geodesic", cfg, dirs, manifest=path, index=0)


def test_geodesic_builds_a_cache_for_a_real_pocket(cfg, dirs, tmp_path):
    pocket = dirs["pockets"] / "1ABC"
    pocket.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    points = rng.random((40, 3)) * 3.0
    pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": points[:, 2],
                  "apbs": 0.0, "cp": 0.0}).to_csv(pocket / "featurized.csv", index=False)
    path = tmp_path / "geodesic.txt"
    path.write_text("1ABC\n")
    run_task("geodesic", cfg, dirs, manifest=path, index=0)
    assert (pocket / "geospectral_cache.npz").exists()


def test_a_structure_result_survives_a_round_trip(dirs):
    original = write_part(dirs, "1ABC", smiles=("CCO", "LIG"))
    back = StructureResult.from_json(
        json.loads((api.parts_dir(dirs) / "1ABC.json").read_text())
    )
    assert back == original
