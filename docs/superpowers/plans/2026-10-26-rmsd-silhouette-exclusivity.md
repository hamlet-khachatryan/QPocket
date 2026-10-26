# In-place RMSD, no silhouette, no exclusivity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Docked-pose RMSD is measured in place (no superposition). Silhouette disappears from the code, files, config and notebook, and clustering defaults to a scale-free relative tree cut. The mutual-exclusivity analysis and its explorer panel are deleted.

**Architecture:** Three independent changes on branch `feat/binding-site-discovery` (worktree `/Users/hamletkhachatryan/PycharmProjects/QPocket-sites`):
- **RMSD** changes one function.
- **Clustering** replaces silhouette-driven k selection with a new `criterion: "relative"`, which cuts the hierarchical tree at a fraction of its highest merge.
- **Exclusivity removal** deletes a vertical slice: core → viz → explorer cell → tests → docs.

**Tech Stack:** Python 3.11, RDKit (`rdMolAlign.CalcRMS`), SciPy hierarchical clustering, marimo, pytest.

**Spec:** user decisions in conversation, 2026-10-26 (no separate design doc):
1. **RMSD:** in place and symmetry-aware. Use `rdMolAlign.CalcRMS` with the MCS atom map and no superposition, since poses and crystal ligands share the aligned frame.
2. **Silhouette:** removed everywhere. Nothing computes, stores, prints or documents it.
3. **Cluster count:**
   - The criterion is `"threshold"` (absolute height), or `"count"` with an explicit `k`. `k_range` is removed.
   - Defaults: both stages hierarchical, with a relative tree cut at 0.5 of the max merge height.
   - The chemistry stage default switches from kmedoids to hierarchical.
4. **Mutual exclusivity:** deleted everywhere: panel, library functions, tests and docs.

## Global Constraints

- **Where and how to test:** work in `/Users/hamletkhachatryan/PycharmProjects/QPocket-sites`.
  - Tests: `PATH=/Users/hamletkhachatryan/PycharmProjects/QPocket/.venv/bin:$PATH PYTHONPATH=$PWD/src python -m pytest -m "not golden"`.
  - Explorer render tests also need a dummy `uv` first on PATH (`/tmp/fakeuv`).
  - Run `test_packaging` without the dummy `uv`.
- **Config files:** `conf/config.yaml` and `src/qpocket/conf/config.yaml` stay byte-identical.
- **Removed keys:** `distance.clustering.{single,shape,chem}.k_range` warn through `LEGACY_KEYS` (value `None`) and never crash.
- **New clustering defaults, exact:**
  - `shape: {method: hierarchical, linkage: average, criterion: relative, threshold: 0.5}`
  - `chem: {method: hierarchical, linkage: average, criterion: relative, threshold: 0.5}`
  - `single` keeps `method: affinity`.
- **`criterion` values:**
  - `"count"` requires `k >= 2`.
  - `"threshold"` is an absolute height > 0, hierarchical only.
  - `"relative"` takes 0 < threshold <= 1, hierarchical only. The cut height is threshold × the highest merge height of that stage's tree.
- **Layering:** `pipelines/` must not import `arms/` (tests/test_layers.py).

## Review Focus

1. **Old saved runs.** A run whose `clustering_params.json` was written before this change has `k: null` and `criterion: "count"`. Recluster and the explorer must give a clear error that names the fix ("set k or use criterion threshold/relative"), not crash. Test in Task 2.
2. **Small stages.** With fewer than 3 pockets, the relative cut must still return one or two clusters and never raise on an empty or degenerate tree. Test in Task 2.
3. **Partial MCS.** When a docked pose's MCS covers only part of the crystal ligand, RMSD is over the mapped atoms only, with no superposition, so a pose translated 5 Å scores ≈ 5 Å. Test in Task 1.
4. **Explorer output.** On a site with clustering results, the explorer must render with no "silhouette" text anywhere and no exclusivity section. Tests in the Task 3 and Task 4 render tests.

---

### Task 1: In-place, symmetry-aware RMSD

**Files:**
- Modify: `src/qpocket/arms/screen/analysis.py:56-91` (`compute_rmsd_mcs`)
- Test: `tests/test_screening.py`, or the existing RMSD test module (grep for `compute_rmsd_mcs`)

**Interfaces:**
- `compute_rmsd_mcs(docked_mol, crystal_mol) -> Optional[float]` keeps its signature.
- It replaces `rdMolAlign.GetBestRMS(docked, crystal, map=[atom_map])` with `rdMolAlign.CalcRMS(docked, crystal, map=[atom_map])`. That does no alignment, and CalcRMS handles symmetry by enumerating equivalent maps.
- The docstring says "in place, no superposition".

- [ ] Step 1: write failing tests.

```python
def test_rmsd_is_measured_in_place():
    crystal = _ligand()                       # any small rigid ligand with 3D coords
    moved = _translated(crystal, (5.0, 0, 0))
    assert compute_rmsd_mcs(moved, crystal) == pytest.approx(5.0, abs=1e-3)

def test_rmsd_of_identical_pose_is_zero():
    crystal = _ligand()
    assert compute_rmsd_mcs(Chem.Mol(crystal), crystal) == pytest.approx(0.0, abs=1e-6)

def test_symmetric_ligand_is_not_penalised_for_atom_order():
    crystal = _benzene_like()                 # symmetric; renumber atoms in the copy
    assert compute_rmsd_mcs(_renumbered(crystal), crystal) == pytest.approx(0.0, abs=1e-3)
```

- [ ] Step 2: run them. The first test should FAIL, because GetBestRMS gives ≈0 for a pure translation.
- [ ] Step 3: implement. Also update any doc text that calls RMSD "best fit" (grep docs for `RMSD`).
- [ ] Step 4: run the screening and crossdock tests, then the full suite.
- [ ] Step 5: commit `fix: docked-pose RMSD is measured in place, without superposition`.

### Task 2: Clustering without silhouette (relative cut, no k_range)

**Files:**
- `src/qpocket/core/clustering.py`:
  - Delete `silhouette()`, `best_k_by_silhouette()`, the `Clustering.silhouette` / `silhouette_per_structure` fields, and every assignment of them.
  - Add the relative cut to `cluster_labels`/`_stage_labels` (`criterion="relative"`).
  - `_stage_labels(k=None, criterion="count")` raises `ValueError("… set k or use criterion 'threshold' or 'relative'")`.
  - Drop the `k_range` parameters (`two_stage_labels` etc.).
- `src/qpocket/plugins/clustering/{hierarchical,kmedoids}.py`: no `best_k_by_silhouette`; kmedoids requires k.
- `src/qpocket/pipelines/clustering.py:100-220`: stop logging and writing `silhouette` and `shape_silhouette`; the params JSON carries criterion and threshold.
- `src/qpocket/config.py`:
  - `StageClustering`: remove `k_range`; `criterion: Literal["count","threshold","relative"]`; new defaults.
  - Validation per Global Constraints.
  - `LEGACY_KEYS` entry for `k_range`.
- Both YAMLs.
- `src/qpocket/arms/recluster.py` and `src/qpocket/cli/runs.py`: recluster's `--shape-criterion`/`--chem-criterion` accept `relative`; remove any k_range options.
- `src/qpocket/viz/cluster.py`: drop the `silhouette` and `best_k_by_silhouette` exports.
- `src/qpocket/viz/two_stage.py`: remove `Preview.silhouette`; `settings_for`/`height_for_count` are unaffected.
- Tests: `tests/test_clustering_two_stage.py`, `tests/test_viz_cluster.py`, `tests/test_config_validation.py`, `tests/test_config_migration.py`, `tests/test_recluster.py`, `tests/test_clustering_plugins.py`.

**Interfaces:**
- `cluster_labels(distances, method, criterion="relative", threshold=0.5)` cuts at `threshold * max(Z[:, 2])`, where `Z` is the scipy linkage. For n < 3 (n = 1 or 2) it returns everything in one cluster, without error.
- `ClusteringParams()` defaults follow Global Constraints. `clustering_params.json` records `criterion` and `threshold` for each stage.

- [ ] Step 1: write failing tests.

```python
def test_relative_cut_is_scale_free():
    d = _two_blobs()
    assert (labels(d, "relative", 0.5) == labels(d * 1000, "relative", 0.5)).all()
    assert len(set(labels(d, "relative", 0.5))) == 2

def test_relative_cut_on_two_pockets_does_not_raise():
    assert len(cluster_labels(np.array([[0, 1.0], [1.0, 0]]), method="average",
                              criterion="relative", threshold=0.5).labels) == 2

def test_count_without_k_is_refused():
    with pytest.raises(ValueError, match="set k or use criterion"):
        _stage_labels(_two_blobs(), "hierarchical", "average", k=None, criterion="count")

def test_defaults_are_relative_hierarchical():
    c = QPocketConfig().distance.clustering
    assert (c.shape.method, c.shape.criterion, c.shape.threshold) == ("hierarchical", "relative", 0.5)
    assert (c.chem.method, c.chem.criterion, c.chem.threshold) == ("hierarchical", "relative", 0.5)

def test_k_range_is_obsolete():
    cfg = OmegaConf.create({"distance": {"clustering": {"shape": {"k_range": [2, 9]}}}})
    assert any("k_range is obsolete" in w for w in migrate_legacy(cfg))

def test_no_silhouette_anywhere():
    import qpocket.core.clustering as cl
    assert not hasattr(cl, "silhouette") and not hasattr(cl, "best_k_by_silhouette")

def test_clustering_files_carry_no_silhouette(tmp_path):
    ...  # write_clustering on a small assignment; no column/key containing "silhouette" in any written file
```

- [ ] Step 2: run them and confirm they FAIL.
- [ ] Step 3: implement. Afterwards, `grep -rni silhouette src conf` must return nothing, except vendored third-party files if any.
- [ ] Step 4: run the listed tests, then the full suite.
- [ ] Step 5: commit `feat: clustering cuts the tree at a relative height; silhouette is gone`.

### Task 3: Explorer without silhouette

**Files:**
- Modify `src/qpocket/viz/explorer.py`:
  - ~487-488: the preview facts.
  - ~547-566: the per-structure silhouette column and mean.
  - ~577: the legend text.
  - ~588, 622, 691: the column lists.
  - The clustering-method controls offer `relative` (a fraction slider, 0.05–1.0) alongside `threshold`.
- Test: `tests/test_notebook_renders.py`.

- [ ] Step 1: extend the render test so the shape-clustered state's rendered output contains no `"silhouette"` (case-insensitive).
- [ ] Step 2: run it with the dummy uv. It should FAIL.
- [ ] Step 3: implement. Keep `utilization`, `dist_to_exemplar` and `is_representative`.
- [ ] Step 4: run the render tests with the dummy uv, then the full suite without it.
- [ ] Step 5: commit `feat: the explorer shows no silhouette`.

### Task 4: Delete mutual exclusivity

**Files:**
- **Explorer:** delete the exclusivity cell(s) at `explorer.py` ~1283-1316, any heading or markdown cell that introduces them, and cells that only consume `exclusivity_members`, `exclusivity_profile` or `verdicts`.
- **Library functions:** delete only those that `grep` shows have no caller outside the exclusivity slice:
  - from `src/qpocket/viz/interactions.py`: `PairVerdict`, `observed_profile`, `shape_overlaps`, `_verdict`, `pair_verdicts`, `describe`, `exclusivity_figure`, `pair_figure`;
  - from `src/qpocket/core/interactions.py`: `Exclusivity`, `exclusivity`, `pairwise_exclusivity`, and their module comment.
- **Exports:** remove the re-exports from `viz/__init__.py`.
- **Tests:** delete their tests in `tests/test_interaction_stats.py` and `tests/test_viz_interactions.py`.
- **Docs:** remove the mentions found by `grep -rni exclusiv docs README.md RUNBOOK.md src/USERGUIDE.MD`.

- [ ] Step 1: extend the render test so the rendered output contains no `"exclusiv"` (case-insensitive). Run it; it should FAIL.
- [ ] Step 2: delete the slice. Afterwards, `grep -rni "exclusiv\|pair_verdict\|shape_overlaps" src tests docs` returns only vendored licence files.
- [ ] Step 3: run the render tests with the dummy uv, `tests/test_layers.py`, and the full suite without the dummy uv.
- [ ] Step 4: commit `refactor: remove the mutual-exclusivity analysis`.

### Task 5: Docs

**Files:** `docs/configuration.md`, `docs/user-guide/first-run.md`, `docs/user-guide/distance-analysis.md`, `docs/user-guide/virtual-screening.md` (RMSD wording), `src/USERGUIDE.MD`.

- [ ] Step 1: run `grep -rni "silhouette\|k_range\|exclusiv\|best fit\|superimpos" docs README.md RUNBOOK.md src/USERGUIDE.MD`. Rewrite each hit to cover:
  - the relative, threshold and count criteria, with the new defaults;
  - in-place RMSD;
  - the removed exclusivity panel.
- [ ] Step 2: commit `docs: relative tree cut, in-place RMSD, no exclusivity`.
