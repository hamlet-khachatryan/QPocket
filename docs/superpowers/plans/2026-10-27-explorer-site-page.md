# Explorer Phase 1: tabbed shell and the Site page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The explorer becomes three tabs (Site · Exploration · Screening). The Site tab can regroup, merge, drop and split sites, with a 3D overview, and asks before deleting results.

**Architecture:**
- Manual site edits are stored as `sites.overrides` in the run config. Each edit names the member cavities it acted on.
- A pure `apply_overrides` in `core/sites.py` re-applies them after automatic grouping. `discover_sites` calls it and records what applied in `grouping.json`.
- `arms/sites.py` gains merge/drop/split/undo plus a `preview_changes` that says which sites lose which results. These go through the existing `sync_sites` + `extract_all_sites` path.
- Notebook-free helpers for the page live in `viz/pages/site.py`. `explorer.py` assembles section views into `mo.ui.tabs`.

**Tech Stack:** Python 3.11, numpy, pandas, plotly, marimo 0.24.2, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-explorer-pages-design.md` (decisions 1, 2, 4, 19, 20; Phase 1)

## Global Constraints

- One notebook, `mo.ui.tabs` with tabs named exactly `Site`, `Exploration`, `Screening`; free navigation; a tab missing its prerequisite says what to do first ("Pick a site on the Site page").
- Overrides are sets of `(pdb_id, cavity)` and follow those cavities across regroups (decision 19).
- Any change to a site's members lists what will be lost (clustering, featurised pockets, final metadata, docking) and runs only after Confirm; then the old folder is deleted as today (decision 20).
- Opening a tab starts no heavy computation (anti-goal).
- Figures use `qpocket.viz.style.apply` and `HUES`/`GREY`.
- Tests: `PATH=$PWD/.venv/bin:$HOME/.hermes/bin:$PATH .venv/bin/python -m pytest -q -p no:cacheprovider` (`uv`, needed by the render tests, lives in `~/.hermes/bin`).

## Decisions left to the builder, taken here (*my calls*)

- **Matching an override to a site:** the current site whose member-cavity set has the highest Jaccard overlap with the recorded set, if that overlap is ≥ 0.5. An override that matches nothing is kept in the config and reported as `unmatched`; it is never silently deleted.
- **Split parts** are the strict complete-linkage cores (`cavity_cores`) inside the site's own group. Each part goes through the normal keep rule (structure share or ligand), so a tiny part can vanish; the report says how many parts were kept.
- **Order of application:** in config order, each override acting on the groups left by the previous one.
- **Dropped sites:** a ligand whose only sites were dropped is a miss with reason `site_dropped`.

## Review Focus

- An override recorded before a regroup that shifts membership slightly (one cavity more or less) still applies — Task 1 `test_override_follows_a_slightly_changed_site`.
- Undoing the only override returns exactly the automatic grouping — Task 3 `test_merge_then_undo_restores_the_automatic_sites`.
- Merging a site with results and a site without: the preview lists only the one with results — Task 3 `test_preview_lists_only_what_exists`.
- A run made before overrides existed (no `sites.overrides`, grouping.json without `applied`) loads and is skipped as before — Task 2 `test_old_grouping_json_is_still_skipped`.
- The explorer on a run with zero sites renders, and Exploration and Screening show their guard — Task 5 `test_tabs_guard_without_a_site`.

---

### Task 1: `apply_overrides` in core

**Files:**
- Modify: `src/qpocket/core/sites.py` (append)
- Test: `tests/test_sites_core.py`

**Interfaces:**
- Produces: `match_site(recorded: Set[Tuple[str, int]], member_sets: Sequence[Set[Tuple[str, int]]], min_jaccard: float = 0.5) -> Optional[int]`.
- Produces: `apply_overrides(groups: List[List[int]], cores: List[List[int]], members: List[List[int]], keys: Sequence[Tuple[str, int]], overrides: Sequence[Mapping]) -> Tuple[List[List[int]], List[List[int]], List[Dict]]`
  - `groups`: site groups after `merge_sites`.
  - `cores`: the strict groups from `cavity_cores`.
  - `members`: per-group membership from `site_members` (used for matching).
  - `keys[i]`: `(pdb_id, cavity)` of candidate `i`.
  - Override dicts: `{"op": "merge", "sites": [[[pdb, cav], …], …]}`, `{"op": "drop", "cavities": [[pdb, cav], …]}`, `{"op": "split", "cavities": [...]}`.
  - Returns `(new_groups, dropped_groups, report)`. `report[i]` = `{"op", "status": "applied"|"unmatched", "parts": n}`. `parts` is present for split/merge.
  - After an applied override, later overrides match against the new groups' members (the union of the members they came from).

- [ ] **Step 1: Write failing tests:**
  - `test_merge_override_joins_two_groups`
  - `test_drop_override_removes_a_group`
  - `test_split_override_returns_its_cores`
  - `test_override_follows_a_slightly_changed_site`: the recorded set misses one cavity of four → still applied.
  - `test_unmatched_override_is_reported_not_applied`: Jaccard < 0.5 → groups unchanged, status `unmatched`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_sites_core.py -q -p no:cacheprovider`. Expected: the new tests FAIL (ImportError).
- [ ] **Step 3: Implement** both functions.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat(sites): re-apply manual merge, drop and split to the automatic grouping`

### Task 2: overrides in the config and in `discover_sites`

**Files:**
- Modify: `src/qpocket/config.py` (`SitesConfig`)
- Modify: `src/qpocket/pipelines/sites.py` (`discover_sites`)
- Test: `tests/test_sites_pipeline.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `SitesConfig.overrides: List[Dict[str, Any]] = field(default_factory=list)`.
  - The `grouping.json` stamp gains `"overrides"`, so changing them re-groups. The file gains:
    - `"applied"`: the report, each entry with `"site_ids"`.
    - `"cores"`: `{site_id: [[[pdb, cav], …], …]}`, the strict groups inside each kept site.
  - The skip check compares only the stamp's keys. A stamp written before overrides existed, compared without an `"overrides"` key, still skips when `overrides` is empty.
  - `MISS_COLUMNS` reason `site_dropped`.
- `discover_sites` order:
  1. `cores = cavity_cores(...)`
  2. `groups = merge_sites(cores, ...)`
  3. Compute boxes and members.
  4. `apply_overrides`.
  5. Recompute `site_boxes`/`members` for the new groups.
  6. Run ligand assignment against kept + dropped boxes. A ligand only in dropped boxes gets reason `site_dropped`.

- [ ] **Step 1: Write failing tests** (with `site_helpers`):
  - `test_a_drop_override_removes_the_site_and_misses_its_ligands`
  - `test_a_merge_override_makes_one_site`
  - `test_grouping_json_records_applied_overrides_and_cores`
  - `test_changing_overrides_regroups`
  - `test_old_grouping_json_is_still_skipped`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_sites_pipeline.py -q -p no:cacheprovider`. Expected: new tests FAIL.
- [ ] **Step 3: Implement.** Reduce `cfg.sites.overrides` with `config._plain` before use, since OmegaConf hands back ListConfig.
- [ ] **Step 4: Run** the same command plus the config tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(sites): overrides in the run config, applied after automatic grouping`

### Task 3: merge / drop / split / undo / preview in the site arm

**Files:**
- Modify: `src/qpocket/pipelines/sites.py` (extract `group_sites` from `discover_sites`)
- Modify: `src/qpocket/arms/sites.py`
- Test: `tests/test_site_steps.py`

**Interfaces:**
- Produces, in `pipelines/sites.py`: `group_sites(cfg, ws, records) -> Grouping`. `Grouping` is a NamedTuple that includes `site_keys: Dict[str, FrozenSet[str]]` (keys `"pdb:cav"`, as `_site_keys`) and whatever `discover_sites` needs to write. It reads but writes nothing.
- Produces, in `arms/sites.py`:
  - `RESULTS = (("clustering", "metadata/clustering_metadata.csv"), ("featurised pockets", "pockets/*/featurized.csv"), ("final metadata", "final"), ("docking", "screening/crossdock/docking_jobs.csv"))`
  - `site_results(run_dir, site_id) -> List[str]`: the labels from RESULTS that exist; featurised pockets with a count, e.g. `"featurised pockets (412)"`.
  - `preview_changes(run_dir, overrides: Optional[List[dict]] = None, **params) -> Dict[str, List[str]]`: old site id → `site_results`, for every existing site whose member set would change. Writes nothing.
  - `list_overrides(run_dir) -> List[dict]`: config entries merged with `status` and `site_ids` from grouping.json `applied`.
  - Builders that return the override without applying it, so the page can preview first: `merge_override(run_dir, site_ids) -> dict`, `drop_override(run_dir, site_id) -> dict`, `split_override(run_dir, site_id) -> dict`.
  - Appliers returning the new site table:
    - `apply_site_overrides(run_dir, overrides: List[dict])` replaces the whole list.
    - `add_override(run_dir, override: dict)`
    - `undo_override(run_dir, index: int)`
  - `regroup_sites` and the appliers share `_resite(run_dir, cfg) -> pd.DataFrame`, which is today's body after the parameter checks.
- Errors are `SiteError`:
  - an unknown site;
  - a merge of < 2 sites;
  - a split of a site with one strict group: "S3-G127 has one strict cavity group; nothing to split";
  - an undo index out of range.

- [ ] **Step 1: Write failing tests** on `_finished_run`:
  - `test_merge_then_undo_restores_the_automatic_sites`
  - `test_drop_removes_the_site_folder`
  - `test_split_of_a_single_core_site_is_refused`
  - `test_preview_lists_only_what_exists`: shape-cluster one site, preview its merge with another; only it lists `"clustering"`.
  - `test_override_survives_a_regroup`: a drop, then a regroup with membership unchanged → still dropped.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_site_steps.py -q -p no:cacheprovider`. Expected: new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command plus `tests/test_sites_pipeline.py tests/test_site_cli.py`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(sites): merge, drop, split and undo sites, with a preview of what is lost`

### Task 4: Site page helpers

**Files:**
- Create: `src/qpocket/viz/pages/__init__.py`, `src/qpocket/viz/pages/site.py`
- Test: `tests/test_viz_site_page.py`

**Interfaces:**
- Consumes: Task 3 `list_overrides`.
- Produces:
  - `overview_figure(run_dir: Path, sites: pd.DataFrame, selected: Optional[str] = None, max_ligands: int = 150) -> go.Figure`:
    - the reference structure's Cα trace, a thin grey line per chain;
    - every site box as a 12-edge wireframe in `HUES[rank % len(HUES)]`, the selected one at width 4 and the others at 1.5, legend entry = site id;
    - heavy atoms of up to `max_ligands` ligands from `structures/ligands.csv` (evenly sampled, seed 0) as 2 px markers in their first site's colour, `GREY` when unassigned.
    - The reference is found as `pipelines.sites._reference_id` finds it. Without one, the first structure is used and the title says so.
  - `override_table(run_dir) -> pd.DataFrame` with columns `#`, `operation`, `sites`, `status`.
  - `loss_markdown(losses: Dict[str, List[str]]) -> str`: `"Nothing is lost."`, or one bullet per site such as `"- **S2-G127**: clustering, featurised pockets (412)"`.

- [ ] **Step 1: Write failing tests:**
  - `test_overview_has_one_box_per_site_and_the_selected_is_thick`
  - `test_overview_caps_ligands`
  - `test_loss_markdown_says_nothing_is_lost`
  - `test_override_table_shows_unmatched`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_viz_site_page.py -q -p no:cacheprovider`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat(explorer): site overview figure and override table`

### Task 5: tabs shell and the Site tab

**Files:**
- Modify: `src/qpocket/viz/explorer.py`
- Test: `tests/test_notebook_renders.py`

**Interfaces:**
- Consumes: Tasks 3–4.
- Layout:
  - The title and run picker stay above the tabs.
  - Section cells assign their content to views (`site_view`, `exploration_view`, `screening_view`, assembled from per-section parts) instead of showing them.
  - One last cell shows `mo.ui.tabs({"Site": …, "Exploration": …, "Screening": …})`.
  - Site tab, in order:
    1. site picker and site table;
    2. 3D overview with the picked site selected;
    3. regroup controls (five parameters);
    4. manual edits: site multiselect + Merge / Drop / Split buttons;
    5. pending-change callout: `loss_markdown(preview_changes(...))` + **Confirm**;
    6. "Overrides" table with an undo selector.
  - Every change (regroup, merge, drop, split, undo) sets a pending change held in `mo.state`. Only Confirm applies it, then it clears.
  - Exploration tab: today's sections from "Steps for …" through "Interaction fingerprint", content unchanged.
  - Screening tab: today's screening section.
  - With no site, Exploration and Screening show `mo.callout(mo.md("Pick a site on the Site page."))`.

- [ ] **Step 1: Write the failing render assertions:**
  - In `test_a_site_run_leaves_no_error_boxes`, the shown text has `"Exploration"`, `"Screening"`, `"Merge"`, `"Split"` and `"Overrides"`.
  - New `test_tabs_guard_without_a_site`: a run with an empty sites table renders with no errors and shows `"Pick a site on the Site page."`.
- [ ] **Step 2: Run** `PATH=$PWD/.venv/bin:$HOME/.hermes/bin:$PATH .venv/bin/python -m pytest tests/test_notebook_renders.py -q -p no:cacheprovider`. Expected: the new assertions FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the render tests, then the full suite. Expected: all pass.
- [ ] **Step 5: Commit** `feat(explorer): Site, Exploration and Screening tabs; site edits with confirm`

### Task 6: check on EV-A71 2A

- [ ] Serve the explorer headless on `~/qpocket_runs/EV71_2A_cavity` with the render harness. Expected: no error boxes; the overview has 10 site boxes.
- [ ] Call `preview_changes` for a merge of S2-G127 and S3-G127 on the real run (read-only). Expected: a loss list; nothing changes on disk (compare `sites/sites.csv` hash before and after). Record the render time in the ledger.
