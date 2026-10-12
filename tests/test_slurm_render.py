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
from pathlib import Path

import pytest

from qpocket.config import QPocketConfig
from qpocket.pipelines.slurm import (
    STAGES_BY_NAME,
    active_stages,
    chunk_bounds,
    read_manifest,
    render_sbatch,
    stage_progress,
    stage_resources,
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


def featurize_script(cfg, dirs, config_file, **kwargs):
    manifest = write_manifest(dirs, STAGES_BY_NAME["featurize"], PDB_IDS)
    return render_sbatch(
        cfg, dirs, STAGES_BY_NAME["featurize"], config_file,
        n_tasks=len(PDB_IDS), manifest=manifest, **kwargs,
    )


# --- the script body ------------------------------------------------------


def test_a_failing_task_aborts_the_script(cfg, dirs, config_file):
    """Without this a failed APBS exits 0 and the stage looks successful."""
    assert "set -euo pipefail" in featurize_script(cfg, dirs, config_file)


def test_the_array_directive_spans_the_manifest(cfg, dirs, config_file):
    assert "#SBATCH --array=0-3%20" in featurize_script(cfg, dirs, config_file)


def test_the_worker_is_handed_its_own_row_to_resolve(cfg, dirs, config_file):
    """No shell CSV parsing: the index goes to Python, which owns the lookup."""
    assert "--index $(( SLURM_ARRAY_TASK_ID + 0 ))" in featurize_script(cfg, dirs, config_file)


def test_log_paths_are_absolute(cfg, dirs, config_file):
    """Relative log paths land in whatever directory you submitted from."""
    for line in featurize_script(cfg, dirs, config_file).splitlines():
        if line.startswith("#SBATCH --output=") or line.startswith("#SBATCH --error="):
            assert Path(line.split("=", 1)[1]).is_absolute()


def test_stray_writes_are_kept_out_of_the_submit_directory(cfg, dirs, config_file):
    """APBS drops an io.mc beside the process; --chdir sends it to scratch."""
    assert "#SBATCH --chdir=" in featurize_script(cfg, dirs, config_file)


def test_a_single_job_gets_no_array_directive(cfg, dirs, config_file):
    text = render_sbatch(cfg, dirs, STAGES_BY_NAME["distance"], config_file)
    assert "--array" not in text


def test_a_single_job_logs_by_job_id_not_array_id(cfg, dirs, config_file):
    text = render_sbatch(cfg, dirs, STAGES_BY_NAME["distance"], config_file)
    assert "%j.out" in text


def test_an_array_stage_without_a_manifest_is_refused(cfg, dirs, config_file):
    with pytest.raises(ValueError, match="manifest"):
        render_sbatch(cfg, dirs, STAGES_BY_NAME["featurize"], config_file, n_tasks=4)


# --- environment preamble -------------------------------------------------


def test_environment_lines_appear_verbatim_and_in_order(cfg, dirs, config_file):
    cfg = evolve(cfg, "slurm.environment", ["module load gcc/11.2", "conda activate qpocket"])
    text = featurize_script(cfg, dirs, config_file)
    assert text.index("module load gcc/11.2") < text.index("conda activate qpocket")


def test_environment_lines_come_before_the_work(cfg, dirs, config_file):
    cfg = evolve(cfg, "slurm.environment", ["conda activate qpocket"])
    text = featurize_script(cfg, dirs, config_file)
    assert text.index("conda activate qpocket") < text.index("qpocket task")


def test_an_empty_environment_leaves_a_note_rather_than_silence(cfg, dirs, config_file):
    assert "PATH" in featurize_script(cfg, dirs, config_file)


# --- optional directives --------------------------------------------------


def test_account_is_omitted_when_unset(cfg, dirs, config_file):
    assert "--account" not in featurize_script(cfg, dirs, config_file)


def test_account_is_emitted_when_set(cfg, dirs, config_file):
    cfg = evolve(cfg, "slurm.account", "proj123")
    assert "#SBATCH --account=proj123" in featurize_script(cfg, dirs, config_file)


def test_qos_is_emitted_when_set(cfg, dirs, config_file):
    cfg = evolve(cfg, "slurm.qos", "high")
    assert "#SBATCH --qos=high" in featurize_script(cfg, dirs, config_file)


def test_a_stage_override_beats_the_defaults(cfg, dirs, config_file):
    cfg = evolve(cfg, "slurm.stages", {"featurize": type(cfg.slurm.defaults)(cpus=8, mem="16G", time="04:00:00")})
    text = featurize_script(cfg, dirs, config_file)
    assert "#SBATCH --mem=16G" in text and "#SBATCH --cpus-per-task=8" in text


def test_a_stage_without_an_override_uses_the_defaults(cfg, dirs, config_file):
    assert stage_resources(cfg, "geodesic") is cfg.slurm.defaults


# --- chunking -------------------------------------------------------------


def test_an_empty_run_emits_no_array_job(cfg, dirs, config_file):
    """`--array=1-0` is rejected by SLURM; emitting nothing is the right answer."""
    scripts = write_scripts(cfg, dirs, config_file, [])
    assert scripts["featurize"] == []


def test_chunks_tile_the_manifest_exactly_once():
    bounds = chunk_bounds(2500, 1000)
    covered = [i for offset, size in bounds for i in range(offset, offset + size)]
    assert covered == list(range(2500))


def test_a_stage_over_the_array_cap_is_split_into_several_scripts(cfg, dirs, config_file):
    cfg = evolve(cfg, "slurm.max_array_size", 2)
    scripts = write_scripts(cfg, dirs, config_file, PDB_IDS, stages=["featurize"])
    assert len(scripts["featurize"]) == 2


def test_each_chunk_offsets_into_the_right_part_of_the_manifest(cfg, dirs, config_file):
    cfg = evolve(cfg, "slurm.max_array_size", 2)
    scripts = write_scripts(cfg, dirs, config_file, PDB_IDS, stages=["featurize"])
    offsets = [
        "+ 0 ))" in p.read_text() or "+ 2 ))" in p.read_text()
        for p in scripts["featurize"]
    ]
    assert all(offsets)


def test_a_run_inside_the_cap_stays_a_single_script(cfg, dirs, config_file):
    scripts = write_scripts(cfg, dirs, config_file, PDB_IDS, stages=["featurize"])
    assert len(scripts["featurize"]) == 1


# --- manifests ------------------------------------------------------------


def test_the_manifest_is_one_id_per_line_with_no_header(cfg, dirs):
    path = write_manifest(dirs, STAGES_BY_NAME["featurize"], PDB_IDS)
    assert path.read_text().splitlines() == PDB_IDS


def test_index_i_resolves_to_line_i_plus_one(cfg, dirs):
    path = write_manifest(dirs, STAGES_BY_NAME["featurize"], PDB_IDS)
    assert read_manifest(path)[2] == "3GHI"


def test_a_manifest_round_trips_ids_unchanged(cfg, dirs):
    path = write_manifest(dirs, STAGES_BY_NAME["geodesic"], PDB_IDS)
    assert read_manifest(path) == PDB_IDS


def test_a_config_path_with_a_space_is_quoted(cfg, dirs, tmp_path):
    """An unquoted path would split into two arguments and the task would fail."""
    spaced = tmp_path / "my runs" / "run.yaml"
    spaced.parent.mkdir()
    spaced.write_text("mode: pdb_query\n")
    text = render_sbatch(cfg, dirs, STAGES_BY_NAME["distance"], spaced)
    assert "'" in text.split("--config")[1].split()[0] or '"my runs"' in text


# --- stage selection ------------------------------------------------------


def test_apbs_runs_inside_featurisation_by_default(cfg):
    assert "apbs" not in [s.name for s in active_stages(cfg)]


def test_apbs_becomes_its_own_stage_when_split(cfg):
    cfg = evolve(cfg, "slurm.split_apbs_stage", True)
    assert "apbs" in [s.name for s in active_stages(cfg)]


def test_the_split_apbs_stage_runs_before_featurisation(cfg):
    cfg = evolve(cfg, "slurm.split_apbs_stage", True)
    names = [s.name for s in active_stages(cfg)]
    assert names.index("apbs") < names.index("featurize")


# --- the driver -----------------------------------------------------------


def test_submit_all_chains_the_configured_dependency(cfg, dirs, config_file):
    scripts = write_scripts(cfg, dirs, config_file, PDB_IDS)
    text = write_submit_all(cfg, dirs, scripts).read_text()
    assert "--dependency=afterany$PREV" in text


def test_submit_all_can_be_switched_to_strict_dependencies(cfg, dirs, config_file):
    cfg = evolve(cfg, "slurm.dependency", "afterok")
    scripts = write_scripts(cfg, dirs, config_file, PDB_IDS)
    text = write_submit_all(cfg, dirs, scripts).read_text()
    assert "--dependency=afterok$PREV" in text


def test_submit_all_submits_every_generated_script(cfg, dirs, config_file):
    scripts = write_scripts(cfg, dirs, config_file, PDB_IDS)
    text = write_submit_all(cfg, dirs, scripts).read_text()
    for paths in scripts.values():
        for path in paths:
            assert path.name in text


def test_the_first_stage_is_submitted_without_a_dependency(cfg, dirs, config_file):
    """`${PREV:+...}` expands to nothing while PREV is empty."""
    scripts = write_scripts(cfg, dirs, config_file, PDB_IDS)
    text = write_submit_all(cfg, dirs, scripts).read_text()
    assert 'PREV=""' in text


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


def test_progress_counts_a_stage_with_nothing_done_as_zero(cfg, dirs, config_file):
    write_scripts(cfg, dirs, config_file, PDB_IDS)
    rows = {r["stage"]: r for r in stage_progress(cfg, dirs)}
    assert rows["featurize"]["complete"] == 0


def test_progress_counts_the_artifacts_that_exist(cfg, dirs, config_file):
    write_scripts(cfg, dirs, config_file, PDB_IDS)
    for pid in PDB_IDS[:2]:
        target = Path(dirs["root"]) / "pockets" / pid
        target.mkdir(parents=True, exist_ok=True)
        (target / "featurized.csv").write_text("x,y,z\n")
    rows = {r["stage"]: r for r in stage_progress(cfg, dirs)}
    assert rows["featurize"]["complete"] == 2


def test_progress_names_what_is_still_missing(cfg, dirs, config_file):
    write_scripts(cfg, dirs, config_file, PDB_IDS)
    rows = {r["stage"]: r for r in stage_progress(cfg, dirs)}
    assert rows["featurize"]["missing"] == PDB_IDS
