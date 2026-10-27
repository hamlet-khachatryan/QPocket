# Explorer Phase 2: the Exploration page and final site metadata — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the Exploration tab the user can override a cluster's representative, featurise (local workers, or SLURM commands), see a clusters × residues contact heatmap, and **Finalise** the site. Finalise writes `sites/<id>/final/`, which Screening reads. Screening refuses to work from a site that is not finalised or whose final metadata is stale.

**Architecture:**
- The representative override rewrites the applied clustering files (`clustering_metadata.csv`, `selected_representatives.csv`) and records itself in `clustering_params.json`. That way every existing reader, including cross-docking, sees it.
- A new `arms/final.py` builds the four final tables from the site's files.
- `viz/pages/exploration.py` holds the profile, heatmap and picker helpers.
- `explorer.py` wires them into the Exploration and Screening tabs (Phase 1's `qpages.on`).

**Tech Stack:** Python 3.11, numpy, pandas, plotly, marimo 0.24.2, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-explorer-pages-design.md` (decisions 5–8, 23; Phase 2)

## Global Constraints

- `final/` holds `clusters.csv`, `settings.json`, `residue_profile.csv`, `ligands.csv`, and `history/`.
  - `clusters.csv`: `pdb_id, cluster, is_representative, dist_to_representative, holo`.
  - `settings.json`: method, percentiles, linkage, feature set, matrix hashes, ISO date.
  - `residue_profile.csv`: `cluster, residue, contacted_pct, lining_pct`.
  - `ligands.csv`: `cluster, pdb_id, ligand_key, resname, smiles`.
- Finalising again moves the previous final into `final/history/<ISO time>/`.
- Representatives are the medoid per cluster by default; any cluster's can be swapped. The override is saved with the clustering.
- Shape-only sites can be finalised: `settings.json` has `"shape_only": true` and `"features": []`.
- The Screening guard says "Finalise this site on the Exploration page." when `final/` is missing, or older than the applied clustering (anti-goal: no silent stale results).
- Heavy steps run only on a button (anti-goal: opening a tab computes nothing).
- Verification on real data: sample only. Finalise S1-G127 on a copy of its metadata, never on the run itself.
- Tests: `PATH=$PWD/.venv/bin:$HOME/.hermes/bin:$PATH .venv/bin/python -m pytest -q -p no:cacheprovider`.

## Decisions left to the builder, taken here (*my calls*)

- **Cluster names** in final tables are the explorer's (`S1`, `S1·c2`), from `qpocket.viz.clusters.cluster_labels_of`. The integer is kept beside them as `cluster_id`.
- **contacted_pct** = share of the cluster's holo members with at least one observed contact on the residue.
- **lining_pct** = share of all members whose pocket lining (potential archive) includes it.
- Only residues with either value > 0 are written.
- **"Older than the applied clustering"** is decided by content: `settings.json` stores the sha256 of `clustering_metadata.csv`, and the final is stale when that file's hash differs.
- **dist_to_representative** uses the matrix the clustering used: combined (spectral), or shape for shape-only. It is recomputed for a cluster whose representative is swapped.
- **Representative overrides** live in `clustering_params.json` as `"representative_overrides": {cluster_id: pdb_id}`. A new Apply makes a fresh clustering without them, since cluster ids change. The page says so next to the picker.
- **"Restyle"** of the existing fingerprint and ligand panels is limited to the shared template (`style.apply`) where a panel builds a plotly figure. Layouts are unchanged.

## Review Focus

- A representative from another cluster is refused — Task 1 `test_a_representative_from_another_cluster_is_refused`.
- Two finalises at the same time stamp keep both histories — Task 2 `test_two_finalises_keep_two_histories`.
- A site with no interaction archives still finalises, with a header-only profile — Task 2 `test_finalise_without_interactions_writes_an_empty_profile`.
- A ligand of a structure that is not a member of this site is not listed — Task 2 `test_ligands_are_this_sites_members_only`.
- Re-clustering or swapping a representative after Finalise makes the final stale — Task 2 `test_new_clustering_makes_the_final_stale`.

---

### Task 1: representative override

**Files:**
- Modify: `src/qpocket/pipelines/clustering.py` (append)
- Modify: `src/qpocket/arms/sites.py` (wrapper)
- Test: `tests/test_site_steps.py`

**Interfaces:**
- Produces: `set_representative(meta_dir: Path, cluster_id: int, pdb_id: str) -> Path` in `pipelines/clustering.py`.
  - Rewrites `is_representative` and that cluster's `dist_to_exemplar` in `clustering_metadata.csv`.
  - Rewrites `selected_representatives.csv` and adds the override to `clustering_params.json["representative_overrides"]` (string keys).
  - Matrix: `spectral_distance_matrix.csv`, unless `mode == "shape_only"` or it is absent, then `shape_distance_matrix.csv`.
  - Raises `ValueError` if `pdb_id` is not in the cluster.
- Produces: `arms.sites.set_representative(run_dir, site_id, cluster_id, pdb_id) -> Path`, which turns that ValueError, and "site not clustered", into `SiteError`.

- [ ] **Step 1: Write failing tests:**
  - `test_set_representative_swaps_it_everywhere`: after shape clustering FAR, pick a non-representative member of its cluster. That member alone is the cluster's representative in both CSVs; its `dist_to_exemplar` is 0; params carry the override.
  - `test_a_representative_from_another_cluster_is_refused`
  - `test_set_representative_before_clustering_is_a_site_error`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_site_steps.py -q -p no:cacheprovider -k representative`. Expected: FAIL (AttributeError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat(clustering): swap a cluster's representative`

### Task 2: Exploration helpers: residue profile, heatmap, representative options

**Files:**
- Create: `src/qpocket/viz/pages/exploration.py`
- Test: `tests/test_viz_exploration_page.py`

**Interfaces:**
- Consumes: `qpocket.viz.interactions.RunInteractions`, `qpocket.viz.style.apply`.
- Produces:
  - `residue_profile(run: Optional[RunInteractions], members: Dict[str, List[str]]) -> pd.DataFrame`, with columns `cluster, residue, contacted_pct, lining_pct, n_members, n_holo`. Clusters keep the dict's order; a None run gives an empty frame with these columns.
  - `residue_heatmap(profile: pd.DataFrame, top: int = 40) -> go.Figure`:
    - rows = clusters in profile order; columns = the `top` residues by mean contacted_pct;
    - colour = contacted_pct on a sequential scale 0–100, hover showing both percentages;
    - an empty profile gives a figure titled "No observed contacts".
  - `representative_options(info: pd.DataFrame, cluster_id: int) -> Dict[str, str]`: label `pdb_id · holo|apo · d=0.0012` → pdb_id, the current representative first.

- [ ] **Step 1: Write failing tests:**
  - `test_residue_profile_percentages`: hand-built RunInteractions, 2 clusters, 3 members, 2 holo.
  - `test_residue_profile_without_archives_is_empty`
  - `test_heatmap_keeps_the_top_residues`
  - `test_empty_profile_heatmap_says_so`
  - `test_representative_options_put_the_current_first`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_viz_exploration_page.py -q -p no:cacheprovider`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat(explorer): residue profile, heatmap and representative picker helpers`

### Task 3: Finalise

**Files:**
- Create: `src/qpocket/arms/final.py`
- Test: `tests/test_final.py`

**Interfaces:**
- Consumes:
  - Task 2 `residue_profile`;
  - `qpocket.viz.clusters.cluster_labels_of`;
  - `qpocket.viz.interactions.load_run_interactions(site_dir)`;
  - `qpocket.arms.sites._site_ws`, `SiteError`.
- Produces:
  - `FINAL_FILES = ("clusters.csv", "settings.json", "residue_profile.csv", "ligands.csv")`
  - `finalise_site(run_dir: Path, site_id: str, now: Optional[datetime] = None) -> Path`: returns `final/`. Raises `SiteError("cluster this site first …")` when there is no `clustering_metadata.csv`. `settings.json` keys:
    - `site_id`, `finalised_at` (ISO, seconds), `shape_only`, `features` (from `metadata/features.json` `columns`, or `[]`), `clustering` (the `clustering_params.json` content);
    - `n_structures`, `n_clusters`, `representatives` (list of pdb_id);
    - `matrices` (`{file name: sha256}` for the distance matrices present), `clustering_hash` (sha256 of `clustering_metadata.csv`).
  - History: an existing `final/` (minus `history/`) moves to `final/history/<finalised_at with ':' → '-'>`, with `-2`, `-3` appended on collision.
  - `final_status(run_dir: Path, site_id: str) -> str`: `"missing"`, `"stale"` or `"current"`.
  - `read_final(run_dir: Path, site_id: str) -> Dict[str, object]`: the four files (frames, and settings as a dict), for Phase 3.

- [ ] **Step 1: Write failing tests** on `_finished_run` with FAR shape-clustered:
  - `test_finalise_writes_the_four_files`: exact columns; one representative per cluster; `holo` True for P2.
  - `test_shape_only_final_records_no_features`
  - `test_two_finalises_keep_two_histories`: the same `now` twice → `history/<t>` and `history/<t>-2`.
  - `test_finalise_without_interactions_writes_an_empty_profile`
  - `test_ligands_are_this_sites_members_only`
  - `test_new_clustering_makes_the_final_stale`: finalise → `"current"`; `set_representative` → `"stale"`; no final → `"missing"`.
  - `test_finalise_before_clustering_is_a_site_error`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_final.py -q -p no:cacheprovider`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command plus `tests/test_docstrings.py`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(sites): finalise a site into final/ with history`

### Task 4: wire the Exploration and Screening tabs

**Files:**
- Modify: `src/qpocket/viz/explorer.py`
- Test: `tests/test_notebook_renders.py`

**Interfaces:**
- Consumes: Tasks 1–3, `qpages.on`.
- Exploration tab:
  1. **Featurise**: a "Workers" number (0 = half the cores) next to the existing Featurise button, passed as `featurize_site(..., workers=)`. The SLURM accordion keeps its commands and writes nothing.
  2. **Representative**, under the cluster table: cluster dropdown → member dropdown (`representative_options`) → "Set representative" run button → `arms.sites.set_representative`. It feeds the existing `applied_at` refresh, so every view rereads. Note: "Applying a new clustering starts from medoids again."
  3. **Residue contacts by cluster**, after the interaction fingerprint: `residue_heatmap(residue_profile(run_interactions, members))` from the applied clustering.
  4. **Finalise**, at the end: a status line (`final_status`, with `finalised_at` when current), a "Finalise site" run button, and a result message with the `final/` path and history count.
- Screening tab: a guard cell at the top. When `final_status != "current"` it shows `mo.callout("Finalise this site on the Exploration page." + reason)`. The existing screening controls stay below it until Phase 3 replaces them.

- [ ] **Step 1: Write the failing render assertions:**
  - `test_a_site_run_leaves_no_error_boxes` (shape-clustered case): shown text has `"Finalise site"`, `"Set representative"`, `"Residue contacts by cluster"`, and `"Finalise this site on the Exploration page."`.
  - `test_a_two_stage_site_renders_the_chemistry_views`: also `"Residue contacts by cluster"`.
- [ ] **Step 2: Run** `PATH=$PWD/.venv/bin:$HOME/.hermes/bin:$PATH .venv/bin/python -m pytest tests/test_notebook_renders.py -q -p no:cacheprovider -k "no_error_boxes"`. Expected: new assertions FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** all render tests once, then the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(explorer): Exploration tab finalises a site; Screening waits for it`

### Task 5: sampled check on EV-A71 2A

- [ ] Copy into a scratch run (read-only on the real run):
  - `metadata/`, `structures/ligands.csv` and `sites/sites.csv` of `~/qpocket_runs/EV71_2A_cavity`;
  - `sites/S1-G127/{metadata,screening}`.

  Do not copy pockets. Run `finalise_site` on the copy. Expected: four files; 36 representatives (the p15 shape clustering); time recorded in the ledger.
- [ ] Render the explorer on the real run with the harness (read-only, no clicks). Expected: no error boxes; the Screening guard shows that the site is not finalised.
