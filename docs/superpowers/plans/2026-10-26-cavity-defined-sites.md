# Cavity-defined binding sites — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ligand-linked sites with sites built from CavFind cavities (box IoU cores, site merge, multi-membership), assign ligands by box containment, report CavFind misses, name sites `S<rank>-<residue>`, and cap the explorer's ligand view.

**Architecture:** Pure geometry in `core/sites.py` (boxes, IoU, cores, merge, membership, ligand assignment, names) with no I/O. `pipelines/sites.py` reads cavities/ligands, calls the core, writes `sites.csv`, `cavities.csv` (membership), `cavfind_misses.csv`, per-site clouds and master tables. Regroup, CLI and explorer follow the new parameters.

**Tech Stack:** Python 3.11, numpy, scipy (`cluster.hierarchy`), pandas, marimo, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-cavity-defined-sites-design.md`

## Global Constraints

- Cavity points are never modified on the way to a site cloud (no box cut, no crop).
- New `sites.*` keys and defaults: `min_cavity_points: 50`, `cavity_overlap: 0.5`, `min_structure_fraction: 0.05`, `ligand_fraction: 0.5`, `box_padding: 0.0`; `min_heavy_atoms`, `keep_ligands` unchanged; `sites.link_cutoff` removed (legacy key, dropped with a note).
- IoU = intersection volume / union volume of axis-aligned boxes.
- States written: `BOUND_STATE`, `UNBOUND_STATE`; `CAVITY_CLOSED` is never produced; structures with neither cavity nor ligand at a site are not listed there.
- Miss reasons: `no_cavity_at_site`, `no_site`.
- Names `S<rank>-<X><num>`, rank 1 = most structures.
- Explorer site 3D view: at most N ligands, default 25, control range 1–200.
- Tests: `PATH=$PWD/.venv/bin:$PATH python -m pytest -q -p no:cacheprovider`; notebook render tests need a real `uv` on PATH (the scratchpad's).
- Never `git commit -am` (the user's uncommitted `scripts/make_complex_fixtures.py` stays out).

## Review Focus

1. A structure with no candidate cavities (all < 50 points, or none found) but with ligands → each ligand is a miss, the structure is in no site, nothing crashes (Task 3 `test_structure_without_candidates_reports_misses`).
2. Two sites with the same nearest residue → distinct names by rank (Task 1 `test_names_unique_when_residue_repeats`).
3. A run with zero candidate cavities → empty `sites.csv` with headers, every ligand `no_site` (Task 3 `test_no_candidates_gives_empty_sites`).
4. Regroup that changes only `ligand_fraction` → sites whose member cavities are unchanged keep their per-site results (Task 4 `test_regroup_keeps_unchanged_sites`).
5. A ligand exactly at the `ligand_fraction` boundary (5 of 10 atoms inside at 0.5) → assigned (≥) (Task 1 `test_ligand_fraction_boundary_inclusive`).

---

### Task 1: Site geometry core

**Files:**
- Modify: `src/qpocket/core/sites.py` — keep `element_symbol`, `find_ligands`; delete `group_ligands`, `site_ids`, `site_box`, `points_in_box`; add the functions below.
- Test: `tests/test_sites_core.py` — delete tests of the removed functions; add those below.

**Interfaces (Produces):**
- `grid_spacing(points: Sequence[np.ndarray]) -> float` — median nearest-neighbour distance over all given clouds (0.0 when < 2 points).
- `voxel_box(points: np.ndarray, half: float) -> np.ndarray` — `(6,)` `[lo_x, lo_y, lo_z, hi_x, hi_y, hi_z]`, extent widened by `half` per face.
- `box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray` — `(n,6)×(m,6) → (n,m)`; 0 where disjoint.
- `cavity_cores(boxes: np.ndarray, threshold: float) -> List[List[int]]` — complete linkage on `1 − IoU`, cut at `1 − threshold`; condensed distances built in row chunks.
- `median_box(boxes: np.ndarray) -> np.ndarray` — element-wise median, `(6,)`.
- `merge_sites(groups: List[List[int]], boxes: np.ndarray, threshold: float) -> List[List[int]]` — repeatedly merge the pair with the highest IoU of their `median_box`es while ≥ `threshold`; merged group = sorted union.
- `site_members(site_boxes: np.ndarray, boxes: np.ndarray, threshold: float) -> List[List[int]]` — per site, every box with IoU ≥ `threshold` to the site box.
- `ligand_sites(coords: Sequence[np.ndarray], site_boxes: np.ndarray, fraction: float, padding: float) -> List[List[int]]` — per ligand, sites whose padded box holds ≥ `fraction` of its heavy atoms (bounds inclusive).
- `site_name(rank: int, residue: str) -> str` — `f"S{rank}-{residue}"`, or `f"S{rank}"` when `residue` is empty.

- [ ] **Step 1: Write the failing tests**

```python
def test_voxel_box_gives_a_flat_cavity_volume():
    b = voxel_box(np.array([[0, 0, 0], [1, 0, 0]], float), half=0.35)
    assert np.allclose(b, [-0.35, -0.35, -0.35, 1.35, 0.35, 0.35])

def test_iou_identical_disjoint_and_third():
    a = np.array([[0, 0, 0, 2, 1, 1]], float)
    assert box_iou(a, a)[0, 0] == 1.0
    assert box_iou(a, a + [5, 0, 0, 5, 0, 0])[0, 0] == 0.0
    assert box_iou(a, np.array([[1, 0, 0, 3, 1, 1]], float))[0, 0] == pytest.approx(1 / 3)

def test_cores_never_hold_a_pair_below_threshold():      # 60 random boxes, seed 0
    groups = cavity_cores(boxes, 0.5)
    assert all(box_iou(boxes[g], boxes[g]).min() >= 0.5 for g in groups if len(g) > 1)

def test_a_b_c_gives_two_sites_with_b_in_both():
    # A=[0,3], B=[1,4], C=[2,5] along x, unit y and z: IoU(A,B)=IoU(B,C)=0.5, IoU(A,C)=0.2
    sites = site_members(np.vstack([median_box(boxes[[0, 1]]), median_box(boxes[[1, 2]])]), boxes, 0.5)
    assert sorted(map(sorted, sites)) == [[0, 1], [1, 2]]

def test_drifting_pocket_cores_merge_into_one_site():  # 8 boxes [0,10]+k*0.4 along x
    assert len(cavity_cores(boxes, 0.9)) > 1
    assert len(merge_sites(cavity_cores(boxes, 0.9), boxes, 0.9)) == 1

def test_ligand_fraction_boundary_inclusive():           # 5 of 10 atoms inside
    assert ligand_sites([coords], site_boxes, 0.5, 0.0) == [[0]]

def test_padding_widens_the_box():                       # all atoms 1 Å outside one face
    assert ligand_sites([outside], site_boxes, 0.5, 0.0) == [[]]
    assert ligand_sites([outside], site_boxes, 0.5, 1.5) == [[0]]

def test_names_unique_when_residue_repeats():
    assert [site_name(1, "H21"), site_name(4, "H21"), site_name(2, "")] == ["S1-H21", "S4-H21", "S2"]

def test_grid_spacing_of_a_regular_grid():
    assert grid_spacing([grid]) == pytest.approx(0.7)    # 4x4x4 grid, step 0.7
```

- [ ] **Step 2: Run to verify they fail** — `PATH=$PWD/.venv/bin:$PATH python -m pytest tests/test_sites_core.py -q -p no:cacheprovider`; expected: ImportError on the new names.
- [ ] **Step 3: Implement** the functions above. `cavity_cores`: zero boxes → `[]`, one → `[[0]]`; otherwise `scipy.cluster.hierarchy.linkage(condensed, "complete")` and `fcluster(..., 1 - threshold, "distance")`.
- [ ] **Step 4: Run to verify they pass**, then `git add src/qpocket/core/sites.py tests/test_sites_core.py && git commit -m "feat(sites): cavity box geometry, cores, merge and membership"`.

### Task 2: Site configuration

**Files:**
- Modify: `src/qpocket/config.py` (`SitesConfig`, `validate_config`, `LEGACY_KEYS`), `conf/config.yaml` and `src/qpocket/conf/config.yaml` (identical), `docs/configuration.md` (`sites` section)
- Test: `tests/test_config_validation.py`, `tests/test_config_migration.py`

**Interfaces (Produces):** `SitesConfig` fields `min_heavy_atoms: int = 6`, `keep_ligands: List[str]`, `min_cavity_points: int = 50`, `cavity_overlap: float = 0.5`, `min_structure_fraction: float = 0.05`, `ligand_fraction: float = 0.5`, `box_padding: float = 0.0`; no `link_cutoff`.

- [ ] **Step 1: Failing tests**

```python
def test_site_defaults():
    s = make_config(mode="pdb_query").sites
    assert (s.min_cavity_points, s.cavity_overlap, s.min_structure_fraction,
            s.ligand_fraction, s.box_padding) == (50, 0.5, 0.05, 0.5, 0.0)

@pytest.mark.parametrize("key,value", [("cavity_overlap", 0.0), ("cavity_overlap", 1.5),
                                       ("ligand_fraction", 0.0), ("ligand_fraction", 1.1),
                                       ("min_structure_fraction", -0.1), ("min_structure_fraction", 1.1),
                                       ("min_cavity_points", 0), ("box_padding", -1.0)])
def test_site_parameter_ranges(key, value):
    with pytest.raises(ValueError, match=f"sites.{key}"):
        validate_config(evolve(make_config(mode="pdb_query"), f"sites.{key}", value))
```

and in `tests/test_config_migration.py`: a YAML with `sites: {link_cutoff: 4.0}` loads, and the loaded `sites` has no `link_cutoff`.

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** Ranges: `cavity_overlap`, `ligand_fraction` in `(0, 1]`; `min_structure_fraction` in `[0, 1]`; `min_cavity_points ≥ 1`; `box_padding ≥ 0`; drop the `link_cutoff` check. `LEGACY_KEYS["sites.link_cutoff"] = None` with the comment "replaced by cavity-defined sites". One-line YAML comment per key. Docs: the spec's algorithm in 8–10 lines; `reference.pocket_definition` / `interaction_padding` no longer affect sites.
- [ ] **Step 4: Run** the config test files and `tests/test_config_sources.py`; expected pass. Other suites still reference `link_cutoff` until Tasks 3–4 — expected, not fixed here.
- [ ] **Step 5: Commit** — explicit `git add` of the listed files; `feat(config): cavity-defined site parameters`.

### Task 3: Site discovery and extraction pipeline

**Files:**
- Modify: `src/qpocket/pipelines/sites.py` (`discover_sites`, `sync_sites`, `extract_site`, `extract_all_sites`; delete `write_contact_atoms` and the Interactions crop), `src/qpocket/core/types.py` (state docstrings)
- Modify: `tests/site_helpers.py` — cavities as cubic grids of 64 points (`4×4×4`, step 0.7) so the 50-point default admits them; keep helper names.
- Test: `tests/test_sites_pipeline.py` (rewritten); every other test that breaks only on a hard-coded `site_01` or a removed column (`test_orchestrators.py`, `test_site_steps.py`, `test_tasks.py`, `test_screening.py`, `test_layout.py`, `test_crossdock_prepare.py`, `test_site_cli.py`, `test_spectral_distance.py`, `test_slurm_*.py`, `test_parallel_stages.py`) reads the site id from `sites.csv` instead.

**Interfaces:**
- Consumes: Task 1 functions; Task 2 `SitesConfig`.
- Produces:
  - `discover_sites(cfg, ws, records) -> pd.DataFrame` writing `sites/sites.csv` (columns `site_id, n_structures, n_holo, n_apo, n_cavfind_miss, n_ligands, n_cavities, centre_x, centre_y, centre_z, box_lo_x, box_lo_y, box_lo_z, box_hi_x, box_hi_y, box_hi_z, resnames, lining_residues, contact_residues`), `sites/cavities.csv` (`pdb_id, cavity, n_points, lo_x … hi_z, site_ids` `;`-joined, empty for non-members), `structures/ligands.csv` (`Ligand` fields + `site_ids`, `;`-joined), `sites/cavfind_misses.csv` (`pdb_id, ligand_key, resname, sites, reason`), `sites/grouping.json` (parameters + input hash).
  - `extract_site(cfg, ws, site, record, members: Sequence[int], ligand_keys: Set[str]) -> SiteRecord` — called only for structures with ≥ 1 member cavity; `BOUND_STATE` when `ligand_keys`, else `UNBOUND_STATE`.
  - `sync_sites(cfg, ws, records)` — a site folder is kept when its member set of `(pdb_id, cavity)` is unchanged.

- [ ] **Step 1: Failing tests** in `tests/test_sites_pipeline.py`:

```python
def test_holo_cloud_is_the_structures_own_cavity_points_unmodified(tmp_path):  # rows == member cavity rows
def test_two_cavities_of_one_structure_in_one_site_are_pooled(tmp_path):
def test_apo_structure_gets_a_cloud_without_ligand(tmp_path):                  # UNBOUND_STATE
def test_ligand_without_cavity_at_its_site_is_a_miss(tmp_path):                # no_cavity_at_site; no cloud; not in master
def test_ligand_in_no_site_is_a_miss(tmp_path):                                # no_site
def test_holo_somewhere_is_not_a_miss(tmp_path):                               # ligand in X and Y, cavity only in Y
def test_small_cavities_never_join(tmp_path):                                  # a 10-point cavity
def test_site_kept_by_ligand_below_structure_share(tmp_path):
def test_site_dropped_without_ligand_below_structure_share(tmp_path):
def test_sites_named_by_rank_and_nearest_residue(tmp_path):                    # "S1-Y1" from the helper protein
def test_structure_without_candidates_reports_misses(tmp_path):
def test_no_candidates_gives_empty_sites(tmp_path):
def test_rerun_with_same_inputs_is_skipped(tmp_path):
def test_a_pair_that_raises_is_still_listed_at_the_site(tmp_path):
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** Order in `discover_sites`: candidates (≥ `min_cavity_points`) from each record's `cavities_file` → `grid_spacing` over candidates → `voxel_box(half = spacing / 2)` → `cavity_cores(cavity_overlap)` → `merge_sites` → `median_box` per group → `site_members` → `ligand_sites` over every record's ligands → keep (structures with a member ≥ `min_structure_fraction × len(records)` or ≥ 1 ligand) → rank by `n_structures` desc, ties by centre x, y, z → nearest residue: closest ATOM of the reference record's `cleaned_file` (the record whose `pdb_id` equals the run's reference; fallback: first record with a member) to the site centre, `THREE_TO_ONE_MAPPING[resname] + str(resseq)` → `site_name` → misses (spec step 10) → write tables. `lining_residues`: residues with an atom within 4 Å of the reference's member points (the existing `_potential` rule). `contact_residues`: observed contacts of the site's holo ligands, as now. A site's master lists only its BOUND/UNBOUND structures; each cloud = the concatenated x, y, z rows of the member cavities from `cavities_file`, unmodified.
- [ ] **Step 4: Run** `tests/test_sites_pipeline.py`, then the full suite; fix the listed modules' hard-coded ids. Expected: all pass except explorer/regroup tests (Task 4).
- [ ] **Step 5: Commit** — explicit `git add`; `feat(sites): sites defined by CavFind cavities`.

### Task 4: Regroup, CLI and explorer

**Files:**
- Modify: `src/qpocket/arms/sites.py` (`regroup_sites`), `src/qpocket/cli/site.py` (`regroup` options `--cavity-overlap`, `--ligand-fraction`, `--box-padding`, `--min-cavity-points`, `--min-structure-fraction`, each optional, at least one required; `--cutoff` removed), `src/qpocket/viz/runs.py` (`site_link_cutoff` → `site_parameters`), `src/qpocket/viz/__init__.py`, `src/qpocket/viz/explorer.py`
- Test: `tests/test_site_steps.py`, `tests/test_site_cli.py`, `tests/test_viz_runs.py`, `tests/test_notebook_renders.py`

**Interfaces:**
- Consumes: Task 3 `sync_sites`, `extract_all_sites`, `sites.csv` columns, `structures/ligands.csv` `site_ids`.
- Produces: `regroup_sites(run_dir: Path, **overrides: float) -> pd.DataFrame` (keys are `SitesConfig` field names, saved into the run config); `site_parameters(run_dir: Path) -> Dict[str, float]` (the five tunable keys, run config else defaults).

- [ ] **Step 1: Failing tests**

```python
def test_regroup_keeps_unchanged_sites(tmp_path):    # change ligand_fraction only; shape results survive
def test_regroup_cli_takes_new_options(tmp_path):    # --cavity-overlap 0.4 saved to the run config
def test_regroup_cli_needs_an_option(tmp_path):      # no option -> click usage error
def test_site_parameters_read_from_run(tmp_path):
def test_site_view_ligand_limit():                   # helper picking ligands: ≤ N, one per holo structure, largest first
```

plus the existing notebook render test passing on a run with the new tables.

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** Explorer: picker label `f"{site_id} · {n_structures} structures · {n_holo} holo · {n_cavfind_miss} misses"`; table columns `site_id, n_structures, n_holo, n_apo, n_cavfind_miss, n_ligands, lining_residues`; regroup numbers for `cavity_overlap`, `ligand_fraction`, `box_padding` seeded from `site_parameters`, with the matching `qpocket site regroup …` hint; site 3D view gets `mo.ui.number(start=1, stop=200, value=25, label="Ligands shown")` and shows the ligands chosen by `viz.runs.site_view_ligands(run_dir: Path, site_id: str, limit: int) -> List[str]` (ligand files: one per holo structure, largest `n_heavy`, in master order, first `limit`) instead of the site's whole `ligands.pdb`.
- [ ] **Step 4: Run the touched tests, then the full suite** (real `uv` on PATH). Expected: all pass.
- [ ] **Step 5: Commit** — `feat(sites): regroup, CLI and explorer for cavity-defined sites`.

### Task 5: Docs, golden note, and the EV-A71 2A regroup

- [ ] Update `docs/user-guide/fragment-screening.md`, `docs/user-guide/distance-analysis.md`, `docs/user-guide/outputs.md`, and `README.md` / `RUNBOOK.md` mentions of `link_cutoff` or ligand-linked sites, to the new logic and files (`cavities.csv`, `cavfind_misses.csv`, new `sites.csv` columns).
- [ ] `tests/golden/CHANGES.md` entry "sites defined by CavFind cavities" (PARG data absent; baseline re-captured when available).
- [ ] Full suite green; commit `docs: cavity-defined sites`.
- [ ] Laptop copy `~/qpocket_runs/EV71_2A_prep`: `qpocket site regroup ~/qpocket_runs/EV71_2A_prep --cavity-overlap 0.5`; report the site table (names, `n_structures`, `n_holo`, `n_cavfind_miss`, box sizes) and miss counts by reason. Expected shape (spike): about 12 sites, the main pocket one site of ~828 structures, box ~13×8×9 Å.
