# Explorer Phase 3: the Screening page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From a finalised site, the Screening tab defines and saves the docking box in either of two modes, with 3D viewers, and sets the docking engine and parameters. It runs the cross-docking here, on SLURM (scripts only) or on a VM over SSH, then analyses success and shows the docked poses. `qpocket screen prepare` uses the final representatives and the saved box.

**Architecture:**
- Reading `final/` moves to a neutral `io/final.py`, so the screen arm can use it: arms may not import one another.
- A new `arms/screen/box.py` builds both box previews and saves `screening/box.json`.
- `arms/screen/crossdock.py` prefers final and saved box, and reports why it falls back in `ScreeningPlan.warnings`.
- A new `arms/screen/submit.py` handles pair counts, the background local run with progress, and remote SSH command construction.
- `viz/pages/screening.py` builds the 3Dmol scenes (box viewers, pose viewer), the success matrix and the ligand picker.
- The Screening tab in `explorer.py` is rebuilt on these.

**Tech Stack:** Python 3.11, numpy, pandas, RDKit, 3Dmol (vendored via `qviz.threedmol_script`), marimo 0.24.2, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-explorer-pages-design.md` (decisions 9–18, 21, 22; Phase 3)

## Global Constraints

- Box modes are exactly `interacting_residues` and `representative_pockets` (the existing `CrossdockConfig.box_mode` values).
  - **Interacting residues** are every residue contacted at least once in the site (`final/residue_profile.csv`, `contacted_pct > 0`), in the reference frame.
  - The **cloud box** uses the final representatives' clouds, with the six extremum points that set its faces.
- Viewers:
  - **Ligand viewer:** ≤ 40 superposed ligands, picked from a searchable picker. The default is the representatives' ligands, then one per cluster, then the most frequent compounds.
  - **Cloud viewer:** clouds coloured by cluster, the extremum points, the tight box and the padded box, and the protein cartoon.
- **Save box** writes `sites/<id>/screening/box.json`, and preparation uses it.
- **Engines:** QuickVina 2, AutoDock Vina, GNINA when installed (greyed when absent), and custom engines from the config.
- **Pairs** are all distinct compounds × all representatives; the counts are shown before submission.
- **Submission:**
  - **Run here:** runs in the background, with progress counted from the finished pose files.
  - **SLURM:** writes the scripts and shows the commands; nothing is submitted (decision 18).
  - **VM over SSH:** `remote:` block in `screen_config.yaml`. It never starts or stops a VM, and warns when other jobs are running there.
- The active/decoy benchmark is shown disabled: "Active/decoy benchmark — not implemented yet".
- **Analysis:** a success matrix (receptor × ligand, best-pose RMSD) and top-k success per receptor. The threshold defaults to 2.0 Å and is a field on the page.
- **Pose viewer:** pick receptor and ligand; it shows the receptor cartoon, the crystal pose, and the top-N docked poses coloured by RMSD, with the interacting residues highlighted.
- **CLI prepare (decision 21):** uses `final/` and a saved box when present. A box saved for an older final is ignored with a warning; without a final it falls back with a warning.
- **Real-data checks are samples only:** dock 1–2 jobs, on a scratch copy.
- Tests: `PATH=$PWD/.venv/bin:$HOME/.hermes/bin:$PATH .venv/bin/python -m pytest -q -p no:cacheprovider`.

## Decisions left to the builder, taken here (*my calls*)

- **Reference frame** for residue coordinates and the cartoon: the site's frame structure's aligned file. That is the run's reference structure if it is a site member, else the first representative. Every aligned file is superposed on the reference.
- **Extremum points:** for each axis, the cloud point with the minimum and the maximum coordinate. Six points, labelled `-x, +x, -y, +y, -z, +z`.
- **`box.json` fields:**
  - the existing `mode, padding, center_*, size_*, min_*, max_*`;
  - plus `site_id`, `saved_at`, `final_hash` (the final's `clustering_hash`), `residues` (interacting mode) or `representatives` (cloud mode).
  - `crossdock.prepare` writes its own task-folder `box.json` from it, as today.
- **Local background run:** a detached `qpocket screen run-local docking_jobs.csv --workers N --cpus C`. Its PID and start time go in `<task folder>/local_run.json`.
  - Progress = pose files present / jobs.
  - "Running" = that PID is alive.
- **Remote block** (`RemoteConfig` in `screen_config.py`): `host` (an ssh alias), `directory`, `setup` (shell lines run before qpocket, e.g. activating the env), and `azure_vm` / `azure_resource_group` (optional, only for showing power state).
  - Commands: `rsync -az --exclude poses/ <folder>/ host:dir/<site>/`, then `ssh host "tmux new -d -s qpocket_<site> '<setup>; cd …; qpocket screen run-local …'"`.
  - Poll: `ssh host "ls dir/<site>/poses | wc -l; tmux ls"`. Pull: `rsync -az host:dir/<site>/poses/ <folder>/poses/`.
  - "Other jobs running" = any tmux session on the host not named `qpocket_<site>`. It is shown as a warning, never acted on.
- **RMSD colours** in the pose viewer: green at or below the threshold, amber up to twice it, red beyond.

## Review Focus

- A box saved, then the site re-finalised with different representatives: prepare ignores the old box and says why — Task 3 `test_a_box_saved_for_an_older_final_is_ignored`.
- The interacting-residues mode on a site whose final profile has no contacts (EV-A71 today): the error is clear, and the cloud mode still works — Task 2 `test_interacting_box_without_contacts_says_so`.
- A remote directory or site id with spaces or quotes stays quoted in every command — Task 4 `test_remote_commands_quote_paths`.
- Progress of a local run with no pose files yet, or after the process died, is reported, not crashed — Task 4 `test_progress_of_a_dead_run`.
- The pose viewer for a pair with no poses yet says so — Task 5 `test_pose_scene_without_poses`.

---

### Task 1: `io/final.py` — readers shared by both arms

**Files:**
- Create: `src/qpocket/io/final.py`
- Modify: `src/qpocket/arms/sites/final.py` (import `FINAL_FILES`, `final_status`, `read_final` from io; keep them re-exported)
- Test: `tests/test_final.py`

**Interfaces:**
- Produces, in `qpocket.io.final`:
  - `FINAL_FILES`
  - `final_status(site_dir: Path) -> str`
  - `read_final(site_dir: Path) -> Dict[str, object]`, which raises `FileNotFoundError` when it is missing.
- The sites-arm functions keep their `(run_dir, site_id)` signatures and call these, mapping FileNotFoundError to SiteError.

- [ ] **Step 1: Failing test:** `test_io_final_reads_what_finalise_wrote`: `qpocket.io.final.read_final(site_dir)["settings"]["site_id"] == FAR`.
- [ ] **Step 2: Run** `tests/test_final.py`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_final.py tests/test_layers.py`. Expected: PASS.
- [ ] **Step 5: Commit** `refactor(final): readers of final/ in io, for the screen arm`

### Task 2: the two box modes

**Files:**
- Modify: `src/qpocket/core/screening.py` (append, pure)
- Create: `src/qpocket/arms/screen/box.py`
- Test: `tests/test_screen_box.py`

**Interfaces:**
- Produces, in core:
  - `extremum_points(points: np.ndarray) -> Tuple[np.ndarray, List[str]]`: a `(6, 3)` array and its labels.
- Produces, in `arms/screen/box.py`:
  - `class BoxPreview(NamedTuple)` with fields:
    - `mode`, `padding`;
    - `tight: Dict[str, float]` and `padded: Dict[str, float]` (both from `calculate_box`, padding 0 and `padding`);
    - `points: np.ndarray`;
    - `extremes: np.ndarray` and `extreme_labels: List[str]`;
    - `residues: List[str]`, `clouds: Dict[str, np.ndarray]` (pdb_id → cloud, cloud mode) and `cluster_of: Dict[str, str]`;
    - `frame_pdb: Path`, `final_hash: str`.
  - `interacting_box(run_dir: Path, site_id: str, padding: float) -> BoxPreview`
  - `cloud_box(run_dir: Path, site_id: str, padding: float) -> BoxPreview`
  - `save_box(run_dir: Path, site_id: str, preview: BoxPreview, now: Optional[datetime] = None) -> Path`
  - `saved_box(run_dir: Path, site_id: str) -> Tuple[Optional[dict], str]`: (box, status), status one of `"none"`, `"stale"`, `"current"`.
  - `class BoxError(ValueError)`, raised for a site that is not finalised ("finalise this site first") or has no contacted residues ("no residue was contacted at this site; use the representatives' clouds").

- [ ] **Step 1: Failing tests** (fixture: `_finished_run`, FAR shape-clustered and finalised):
  - `test_extremum_points_set_the_tight_box`
  - `test_cloud_box_covers_every_representative_cloud`: the padded box equals the tight box ± padding.
  - `test_interacting_box_uses_every_contacted_residue`: write a final profile with two contacted residues; the points are those residues' atoms in the frame structure.
  - `test_interacting_box_without_contacts_says_so`
  - `test_save_then_saved_box_is_current_until_refinalised`: refinalise after a representative swap → `"stale"`.
  - `test_box_needs_a_final`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_screen_box.py -q -p no:cacheprovider`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command plus `tests/test_layers.py tests/test_docstrings.py`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(screen): interacting-residue and cloud boxes, saved per site`

### Task 3: prepare prefers final + saved box

**Files:**
- Modify: `src/qpocket/arms/screen/crossdock.py` (`prepare`, `_snapshot_representatives`, `_prepare_in`)
- Test: `tests/test_crossdock_prepare.py`

**Interfaces:**
- Consumes: Task 1 `final_status`/`read_final`, Task 2 `saved_box`.
- Behaviour:
  - When the site's final is `"current"`, the snapshot's `selected_representatives.csv` is written from `final/clusters.csv` (`is_representative`).
  - When `saved_box` is `"current"`, its bounds are used and `box.json` records `"source": "saved"`. Otherwise the bounds come from `cfg.crossdock` as today, with `"source": "config"`.
  - `ScreeningPlan.warnings` gains:
    - `"no final metadata for <site>; using the applied clustering's representatives"`
    - `"the saved box predates the last Finalise; using the configured box"`
    - `"no saved box; using the configured box"`
  - The CLI already prints `plan.warnings` (Phase 0).

- [ ] **Step 1: Failing tests** on the `site` fixture plus a final:
  - `test_prepare_uses_the_final_representatives`
  - `test_prepare_uses_a_current_saved_box`
  - `test_a_box_saved_for_an_older_final_is_ignored`
  - `test_prepare_without_final_warns`
- [ ] **Step 2: Run** `tests/test_crossdock_prepare.py`. Expected: new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_crossdock_prepare.py tests/test_screen*.py`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(screen): prepare uses the final representatives and the saved box`

### Task 4: submission — pair counts, local background run, remote commands

**Files:**
- Create: `src/qpocket/arms/screen/submit.py`
- Modify: `src/qpocket/screen_config.py` (`RemoteConfig`, `ScreenConfig.remote`)
- Test: `tests/test_screen_submit.py`

**Interfaces:**
- Produces:
  - `pair_count(run_dir, site_id) -> Tuple[int, int, int]`: receptors, distinct compounds, pairs, read from the final.
  - `engines(cfg: ScreenConfig) -> Dict[str, Optional[str]]`: engine name → executable, or None when not installed (from the DOCKING registry plus `docking.custom_engines`).
  - `start_local(folder: Path, workers: int, cpus: int) -> dict`: writes `local_run.json` `{pid, started_at, n_jobs, command}` and returns it. It refuses with `RuntimeError` while a run recorded there is alive.
  - `local_progress(folder: Path) -> Dict[str, object]`: `{"done", "total", "running": bool, "started_at"}`.
  - `remote_commands(folder: Path, site_id: str, remote: RemoteConfig, workers: int, cpus: int) -> Dict[str, List[str]]`, with keys `push, start, poll, pull`, each an argv list for `subprocess.run`, built with `shlex.quote` inside remote shell strings.
  - `remote_status(poll_output: str, site_id: str) -> Dict[str, object]`: `{"poses": int, "own_session": bool, "other_jobs": List[str]}`.
  - `vm_power_state(remote: RemoteConfig) -> Optional[str]`: via `az vm get-instance-view`, or None when `az` or the VM name is missing. Never changes the VM.

- [ ] **Step 1: Failing tests:**
  - `test_pair_count_is_reps_times_distinct_compounds`
  - `test_engines_mark_missing_gnina`: monkeypatch `resolve_executable` to raise for gnina.
  - `test_start_local_records_and_refuses_a_second_run`: monkeypatch `subprocess.Popen`.
  - `test_progress_of_a_dead_run`: pid of an exited process → `running False`, with done counted.
  - `test_remote_commands_quote_paths`: directory `"/data/my runs"`, site `S1-G'127`.
  - `test_remote_status_sees_other_jobs`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_screen_submit.py -q -p no:cacheprovider`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command plus config tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(screen): local background runs, remote SSH commands, pair counts`

### Task 5: Screening page helpers — scenes, ligand picker, analysis tables

**Files:**
- Create: `src/qpocket/viz/pages/screening.py`
- Test: `tests/test_viz_screening_page.py`

**Interfaces:**
- Consumes: Task 2 `BoxPreview`, `qviz.threedmol_script`, `qviz.ligand_pdb`.
- Produces:
  - `scene_html(models: List[dict], boxes: List[dict], spheres: List[dict], height: int = 520, offline: bool = True) -> str`. It is one 3Dmol viewer:
    - a model is `{"pdb": str, "style": dict}`;
    - a box is `{"min": [x,y,z], "max": [x,y,z], "color": str, "dashed": bool}`, drawn as a wireframe;
    - a sphere is `{"center": [...], "radius": float, "color": str, "label": str}`.
  - `default_ligands(ligands: pd.DataFrame, clusters: pd.DataFrame, limit: int = 40) -> List[str]`: ligand_keys, the representatives' first, then one per cluster, then by SMILES frequency.
  - `ligand_box_scene(preview, ligand_files: List[str]) -> str`: cartoon plus highlighted residues (sticks), ligands, tight and padded boxes.
  - `cloud_box_scene(preview) -> str`: cartoon, clouds as small spheres coloured by cluster (`cluster_colours`-consistent hues), extremum points labelled, tight and padded boxes.
  - `success_matrix(pairs: pd.DataFrame) -> pd.DataFrame`: representative × active, best-pose RMSD.
  - `matrix_figure(matrix: pd.DataFrame, threshold: float) -> go.Figure`
  - `pose_scene(folder: Path, job: pd.Series, top_n: int, threshold: float, residues: List[str]) -> Tuple[str, pd.DataFrame]`: the HTML plus a table of `rank, score, rmsd` shown. With no pose file it returns a message scene and an empty table.

- [ ] **Step 1: Failing tests:**
  - `test_scene_html_carries_models_boxes_and_spheres` (parse the embedded JSON payload)
  - `test_default_ligands_order_and_cap`
  - `test_cloud_scene_labels_six_extremes`
  - `test_success_matrix_takes_the_best_pose`
  - `test_pose_scene_without_poses`
  - `test_pose_scene_colours_by_rmsd`: a fake engine `read_poses` returning two poses.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_viz_screening_page.py -q -p no:cacheprovider`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat(explorer): screening scenes, ligand picker and success matrix`

### Task 6: the Screening tab

**Files:**
- Modify: `src/qpocket/viz/explorer.py` (replace the three old cross-docking cells; keep the Phase 2 guard)
- Test: `tests/test_notebook_renders.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Sections, each wrapped with `qpages.on("Screening", …)` and each stopping with the guard message while the final is not current:
  1. **Run type**: `mo.ui.radio(["Cross-docking"])`, plus a greyed line "Active/decoy benchmark — not implemented yet".
  2. **Box**: mode radio, padding slider (0–15 Å, step 0.5, default 5), and the viewer for the mode. The ligand picker is `mo.ui.multiselect` over the site's ligands, defaulting to `default_ligands`, capped at 40. **Save box** writes; the current saved-box status is shown.
  3. **Docking**: engine dropdown (missing engines labelled "not installed"; choosing one is refused), exhaustiveness, modes, CPUs; "N receptors × M compounds = P dockings"; **Prepare** → `crossdock.prepare(site_dir, engine=…, docking overrides)` → its warnings.
  4. **Run**:
     - Run here: Start, then progress (Refresh button; `local_progress`).
     - SLURM: shows `plan.command` and the sbatch path.
     - VM: from the config `remote` block, or a message when there is none; Push & start, Poll, Pull buttons; power state, and an other-jobs warning.
  5. **Analysis**: threshold number (default 2.0); **Analyse** → `crossdock.analyze(site_dir, **{"analysis.rmsd_threshold": t})`; success table, matrix figure, top-k per receptor.
  6. **Poses**: receptor dropdown, ligand dropdown, top-N number → `pose_scene`.
- Long-running buttons keep their outcome in `mo.state` (the Phase 1 pattern).

- [ ] **Step 1: Failing render assertions** in the shape-clustered render test, after finalising S1-Y1 in the fixture (`final.finalise_site`): the text has `"Active/decoy benchmark — not implemented yet"`, `"Save box"`, `"dockings"`, `"Run here"`, `"Analyse"`. And a click test `test_save_box_from_the_page` presses Save box → `box.json` exists.
- [ ] **Step 2: Run** those render tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the render tests, then the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(explorer): Screening tab — box, docking, runs, analysis and poses`

### Task 7: sampled check on EV-A71 2A

- [ ] On the scratch copy finalised in Phase 2 (`scratchpad/ev71_copy`): run `cloud_box` with padding 5, save it, and compare with the Phase 0 task-folder box (min/max within 0.01 Å when the representatives agree). Record the timings.
- [ ] Dock **two** sampled jobs (indices 0 and 1) in a scratch task folder. Copy `docking_jobs.csv` (first two rows), their receptors, ligands and `engines.json` from `~/qpocket_runs/EV71_2A_cavity/sites/S1-G127/screening/crossdock`. Run `qpocket screen run-task docking_jobs.csv --index 0/1`, then `analyze` on the copy, and build `pose_scene` for job 0. Expected: poses written, RMSD values, an HTML scene. The real run is untouched.
