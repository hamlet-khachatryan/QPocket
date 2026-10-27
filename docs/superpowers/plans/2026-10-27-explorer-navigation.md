# Explorer navigation and percentile cuts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Express every hierarchical cut as a site percentile (the slider slides the percentile, the preview follows live), and redraw the explorer for large sites — collapsed trees with a cluster table, k × k cluster-distance map, label-free landscape, a 40-pocket searchable cloud picker — in one minimal, coloured figure style.

**Architecture:** One core helper turns a percentile into a cut height on the site's whole matrix; every stage then cuts at that height, so the two-stage algorithm is unchanged and chemistry uses one site-wide scale. Config stores `percentile` per hierarchical stage. A new `viz/style.py` owns the Plotly template and cluster colours; `viz/two_stage.py`, `viz/matrix.py`, `viz/grid.py` and a new `viz/clusters.py` provide the collapsed and aggregated views; `viz/explorer.py` keeps its sections, controls and data flow and only swaps what it draws.

**Tech Stack:** Python 3.11, numpy, scipy (`cluster.hierarchy`), pandas, plotly, marimo 0.24.2, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-explorer-navigation-design.md`

## Global Constraints

- Same setup as before: sections, controls, Preview / "Apply to this run" flow, methods and cloud-viewer modes stay; only drawing and cut expression change.
- Percentile = empirical quantile of the upper-triangle pairwise distances of the site matrix; `p` in (0, 100). Chemistry percentile is site-wide (the full chemistry matrix), never per shape cluster.
- Defaults: shape `15`, chemistry `20`, single-mode `15`.
- `kmedoids` keeps `k`; `affinity` keeps its own parameters; percentile applies to hierarchical stages only.
- Small clusters: size < `distance.clustering.min_cluster_size` (4) → grey `#9aa3ad`, pooled as one "small clusters (n)" leaf/row.
- Above-cut branches grey `#9aa3ad`; cut lines dashed; Helvetica/Arial 10 pt; axis lines only, ticks outside, no gridlines, white background.
- Point clouds: at most 40 shown; default = representatives first, then the selected cluster's members in master order.
- Consensus / difference colouring always over the whole cluster.
- Tests: `PATH=$PWD/.venv/bin:$PATH python -m pytest -q -p no:cacheprovider`; notebook renders need the scratchpad's real `uv` on PATH.
- Never `git commit -am` (the user's uncommitted `scripts/make_complex_fixtures.py`).

## Review Focus

1. A site whose distances are all equal (zero spread) → percentile cut gives one cluster, axes don't divide by zero (Task 1 `test_percentile_cut_on_constant_matrix`).
2. Fewer than 3 structures → cut and figures still render (Task 1 `test_percentile_cut_two_structures`; Task 5 `test_collapsed_tree_two_leaves`).
3. More shape clusters than palette hues → colours cycle but stay identical across calls and views (Task 4 `test_colours_cycle_stably`).
4. A cloud selection larger than 40 → the first 40 are shown and the note says how many were dropped (Task 6 `test_cloud_selection_capped_with_note`).
5. An old run config with `criterion: relative, threshold: 0.5` → loads, the stage gets the default percentile, a note is logged (Task 2 `test_old_criterion_migrates_to_percentile`).

---

### Task 1: Percentile cut in the core

**Files:** Modify `src/qpocket/core/clustering.py`; Test `tests/test_clustering_percentile.py` (new)

**Interfaces (Produces):**
- `pair_distances(distances: np.ndarray) -> np.ndarray` — upper-triangle values (k=1), float.
- `percentile_height(distances: np.ndarray, percentile: float) -> float` — `np.percentile(pair_distances, percentile)`; `0.0` with fewer than 2 structures.
- `to_percentile(distances: np.ndarray, values: np.ndarray) -> np.ndarray` — empirical CDF × 100 of `values` against the pairs (`searchsorted(sorted_pairs, values, side="right") / n_pairs * 100`), clipped to [0, 100].
- `cluster_labels(..., criterion="percentile", threshold=p)` — new criterion (`PERCENTILE = "site percentile"` added to `CRITERIA`, short name `"percentile"`): cut at `percentile_height(distances, threshold)`, exactly as the threshold criterion cuts.

- [ ] **Step 1: Failing tests**

```python
def test_percentile_height_is_the_quantile_of_pairs():
    d = squareform([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])          # 4 structures
    assert percentile_height(d, 50) == pytest.approx(3.5)

def test_to_percentile_is_monotone_and_bounded():
    d = squareform(np.arange(1.0, 11.0))                   # 5 structures
    p = to_percentile(d, np.array([0.0, 1.0, 5.5, 10.0, 99.0]))
    assert list(p) == sorted(p) and p[0] == 0 and p[-1] == 100

def test_percentile_cut_matches_the_threshold_cut():
    d = _random_matrix(30, seed=0)
    a = cluster_labels(d, method="average", criterion="percentile", threshold=15)
    b = cluster_labels(d, method="average", criterion="threshold", threshold=percentile_height(d, 15))
    assert np.array_equal(a.labels, b.labels)

def test_percentile_cut_on_constant_matrix():
    d = np.ones((6, 6)) - np.eye(6)
    assert cluster_labels(d, method="average", criterion="percentile", threshold=15).n_clusters == 1

def test_percentile_cut_two_structures():
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert cluster_labels(d, method="average", criterion="percentile", threshold=50).n_clusters in (1, 2)
```

- [ ] **Step 2: Run** `… pytest tests/test_clustering_percentile.py -q`; expected ImportError.
- [ ] **Step 3: Implement** the four interfaces; existing criteria keep working in the core (Task 2 removes them from the config only).
- [ ] **Step 4: Run** the new file plus `tests/test_clustering*.py`; expected pass.
- [ ] **Step 5: Commit** `feat(clustering): percentile cuts on the site's own distances`.

### Task 2: Percentile in the configuration

**Files:** Modify `src/qpocket/config.py` (`StageClustering`, per-stage defaults, `validate_config`, `LEGACY_KEYS` / `migrate_legacy`), `conf/config.yaml`, `src/qpocket/conf/config.yaml`, `docs/configuration.md`; Test `tests/test_config_validation.py`, `tests/test_config_migration.py`

**Interfaces (Produces):** `StageClustering` fields `method`, `linkage`, `percentile: float`, `k: Optional[int]` (kmedoids only), `damping`, `max_iter`; no `criterion`, no `threshold`. `ClusteringParams().shape.percentile == 15.0`, `.chem.percentile == 20.0`, `.single.percentile == 15.0`.

- [ ] **Step 1: Failing tests**

```python
def test_cut_defaults_are_percentiles():
    c = ClusteringParams()
    assert (c.shape.percentile, c.chem.percentile, c.single.percentile) == (15.0, 20.0, 15.0)
    assert not hasattr(c.shape, "criterion") and not hasattr(c.shape, "threshold")

@pytest.mark.parametrize("value", [0.0, 100.0, -5.0])
def test_percentile_range(value):
    # distance.clustering.shape.percentile = value -> ValueError naming that key

def test_kmedoids_needs_k():
    # shape.method kmedoids, k None -> ValueError naming distance.clustering.shape.k

def test_old_criterion_migrates_to_percentile(tmp_path):
    # YAML distance.clustering.shape: {criterion: relative, threshold: 0.5}
    # loads; shape.percentile == 15.0; migrate_legacy notes mention "percentile"
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** Stage defaults via per-stage `default_factory` (shape 15, chem 20, single 15). `LEGACY_KEYS` gains `distance.clustering.{shape,chem,single}.{criterion,threshold}: None`, all with the note "cuts are site percentiles now; set distance.clustering.<stage>.percentile". The count→relative fallback in `migrate_legacy` (config.py ≈ line 540) becomes: a hierarchical stage with `k` and no percentile gets the default percentile, with a note. Validation: `0 < percentile < 100`; `kmedoids` requires `k ≥ 2`. One-line YAML comments. Docs: the `distance.clustering` block shows `percentile`, explained in two sentences (site quantile; chemistry measured on the whole site).
- [ ] **Step 4: Run** the config tests and `tests/test_config_sources.py`; other suites may fail on `criterion` until Task 3 — expected.
- [ ] **Step 5: Commit** `feat(config): clustering cuts as site percentiles`.

### Task 3: Pipelines, plug-ins and recluster CLI on percentiles

**Files:** Modify `src/qpocket/pipelines/clustering.py`, `src/qpocket/plugins/clustering/{hierarchical,kmedoids,affinity}.py`, `src/qpocket/arms/recluster.py`, `src/qpocket/cli/runs.py`; Test: update `tests/test_clustering_plugins.py`, `tests/test_clustering_two_stage.py`, `tests/test_two_stage_threshold.py`, `tests/test_recluster.py`, `tests/test_apply_clustering.py`, `tests/test_site_steps.py` to percentiles, plus the new test below.

**Interfaces:**
- Consumes Task 1 `percentile_height`, Task 2 `StageClustering.percentile`.
- Produces `stage_labels(distances, method, linkage, k, percentile=None, damping=…, max_iter=…, site_distances=None)` — a hierarchical stage cuts at the height `percentile_height(site_distances if given else distances, percentile)`; `cluster_pockets` passes `site_distances=d_chem` to every chemistry sub-cut, so the chemistry scale is site-wide. `clustering_params.json` records `percentile` and the resolved `cut_height` per stage.

- [ ] **Step 1: Failing test** in `tests/test_clustering_two_stage.py`:

```python
def test_chemistry_percentile_is_site_wide():
    # two shape groups; group B chemically uniform (all chem distances 0.1), group A spread 1..10.
    # The site-wide p20 height is above 0.1, so B is not split and A is.
    a = cluster_pockets(settings_with(shape=50, chem=20), d_shape, d_chem, combined)
    assert len(set(a.chem_labels[b_members])) == 1 and len(set(a.chem_labels[a_members])) > 1
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** In `recluster`, replace `--shape-criterion/--shape-threshold/--chem-criterion/--chem-threshold` with `--shape-percentile` / `--chem-percentile` (floats); docstring example `qpocket recluster runs/7KG6 --shape-percentile 15 --chem-percentile 20`. Update the listed test files from heights/fractions to percentiles.
- [ ] **Step 4: Run the full suite.** Expected: all pass except explorer/viz tests owned by Tasks 5–7.
- [ ] **Step 5: Commit** `feat(clustering): pipelines and recluster cut by site percentile`.

### Task 4: Figure style and cluster colours

**Files:** Create `src/qpocket/viz/style.py`; Modify `src/qpocket/viz/__init__.py`; Test `tests/test_viz_style.py` (new)

**Interfaces (Produces):**
- `TEMPLATE: plotly.graph_objects.layout.Template` — font Helvetica/Arial 10, white paper/plot, `showgrid=False`, `showline=True`, `linecolor="#222"`, `ticks="outside"`, `zeroline=False`.
- `apply(fig: go.Figure) -> go.Figure` — sets `template=TEMPLATE` and tight margins.
- `GREY = "#9aa3ad"`; `HUES: Tuple[str, ...]` — ≥ 10 colour-blind-safe hues, the first four `#1f5f8b, #2a9d8f, #e9a03b, #c4566a` (the approved mockup's).
- `cluster_colours(shape_labels: Sequence[int], chem_labels: Optional[Sequence[int]], min_size: int) -> Dict[Tuple[int, int], str]` — key `(shape, chem)` (chem `0` when there is none); shape clusters ordered by size desc, then label, take `HUES` in order, cycling; chemistry subclusters take tints of their shape hue in 15 % lightness steps, largest darkest; any cluster smaller than `min_size` → `GREY`.

- [ ] **Step 1: Failing tests**

```python
def test_largest_cluster_gets_the_first_hue(): ...
def test_subclusters_are_tints_of_their_shape_hue(): ...      # same hue, increasing lightness
def test_small_clusters_are_grey(): ...                        # size 3 with min_size 4 -> GREY
def test_colours_cycle_stably(): ...                           # 14 clusters, 10 hues; two calls equal
def test_template_has_no_gridlines():
    assert TEMPLATE.layout.xaxis.showgrid is False and TEMPLATE.layout.yaxis.showgrid is False
```

- [ ] **Step 2–4:** run (fail), implement, run (pass).
- [ ] **Step 5: Commit** `feat(viz): one figure style and stable cluster colours`.

### Task 5: Collapsed trees and the cluster table

**Files:** Modify `src/qpocket/viz/two_stage.py`; Create `src/qpocket/viz/clusters.py`; Test: update `tests/test_viz_two_stage.py`, `tests/test_two_stage_figure.py`; new `tests/test_viz_clusters.py`

**Interfaces:**
- Consumes Tasks 1 and 4.
- Produces in `two_stage.py`:
  - `settings_for(base, method, combined_cut=None, shape_cut=None, chem_cut=None, damping=None, min_cluster_size=None)` — cuts are percentiles;
  - `shape_only_settings(settings, shape_cut: float)`;
  - `percentile_settings(settings, shape_cut: float, chem_cut: Optional[float], min_cluster_size=None)` (replaces `threshold_settings`);
  - `collapsed_figure(m: RunMatrices, preview_result, shape_cut: float, chem_cut: Optional[float], linkage: str, colours: Dict) -> go.Figure` — shape tree drawn down to the cut, one leaf per shape cluster (size label, coloured ribbon), small clusters pooled into one grey leaf "small clusters (n)", y axis "Shape distance (percentile)" via `to_percentile`, branches coloured below the cut and grey above, a dashed cut line labelled `p15`; with chemistry, each shape cluster's chemistry subtree mirrored below on a "Physicochemical distance (percentile)" axis with one dashed line; leaf marker points carry `customdata` = cluster id; hover on merges shows raw distance and percentile;
  - `subtree_figure(m: RunMatrices, member_ids: Sequence[str], kind: str, linkage: str) -> go.Figure` — member-level tree of one cluster, no leaf labels, ID on hover.
- Produces in `clusters.py`:
  - `shared_volume(a: np.ndarray, b: np.ndarray, spacing: float) -> float` — points rounded to `spacing`; |A∩B| / |A∪B|;
  - `cluster_table(active_info: pd.DataFrame, m: RunMatrices, clouds: Dict[str, np.ndarray], min_size: int, max_pairs: int = 200, seed: int = 0) -> pd.DataFrame` — columns `cluster, level ("shape"|"chem"), parent, n, holo, shape_within, chem_within, shared_volume, representative`; small clusters pooled into one row with `cluster == "small"`.

- [ ] **Step 1: Failing tests**

```python
def test_collapsed_tree_has_one_leaf_per_cluster(): ...       # 3 clusters + pooled small -> 4 leaf markers
def test_collapsed_tree_axis_is_percentile(): ...             # y within [0, 100]; title mentions percentile
def test_collapsed_tree_two_leaves(): ...                     # 2 structures render
def test_chemistry_line_is_one_height_across_subtrees(): ...  # the chem cut is a single shape at one y
def test_shared_volume_identical_and_disjoint(): ...          # 1.0 and 0.0
def test_cluster_table_columns_and_pooling(): ...             # expected columns; a size-2 cluster in "small"
def test_within_is_a_site_percentile(): ...                   # tight cluster's within < loose cluster's
```

- [ ] **Step 2–4:** run (fail), implement, run (pass).
- [ ] **Step 5: Commit** `feat(viz): collapsed trees and cluster table`.

### Task 6: Cluster-distance map, landscape and cloud picker

**Files:** Modify `src/qpocket/viz/matrix.py`, `src/qpocket/viz/grid.py`, `src/qpocket/viz/clusters.py`; Test `tests/test_viz_matrix.py` (new), `tests/test_viz_grid.py`

**Interfaces (Produces):**
- `matrix.cluster_map(M: pd.DataFrame, labels: pd.Series) -> pd.DataFrame` — k × k median between-cluster distance as a site percentile (`to_percentile` on M's own pairs); diagonal = median within.
- `matrix.cluster_map_figure(cmap: pd.DataFrame, colours: Dict) -> go.Figure` — styled; cell hover shows both cluster sizes and the percentile.
- `matrix.member_block(M: pd.DataFrame, labels: pd.Series, a, b) -> pd.DataFrame` — rows and columns of clusters a and b, ordered by cluster then leaf order.
- `clusters.landscape_figure(coords: pd.DataFrame, colour_by: str, colours: Dict) -> go.Figure` — markers only (no text mode), size 6, opacity 0.7, hover `pdb_id · cluster · holo|apo`.
- `grid.cloud_options(active_info: pd.DataFrame) -> Dict[str, str]` — label `f"{pdb_id} · {cluster} · {'holo' or 'apo'}"` → pdb_id, every structure of the site.
- `grid.default_cloud_selection(active_info: pd.DataFrame, selected_cluster, limit: int = 40) -> List[str]` — representatives first, then the selected cluster's members in master order, deduplicated, at most `limit`.
- `grid.capped(selection: Sequence[str], limit: int = 40) -> Tuple[List[str], str]` — the first `limit` and a note (`""` when nothing is dropped).

- [ ] **Step 1: Failing tests**

```python
def test_cluster_map_is_k_by_k_in_percentiles(): ...
def test_member_block_holds_only_the_two_clusters(): ...
def test_landscape_has_no_text_labels(): ...                  # every trace mode == "markers"
def test_default_selection_representatives_first(): ...
def test_cloud_selection_capped_with_note(): ...              # 55 ids -> 40 kept; note mentions 15
```

- [ ] **Step 2–4:** run (fail), implement, run (pass).
- [ ] **Step 5: Commit** `feat(viz): cluster-distance map, label-free landscape, cloud picker`.

### Task 7: Explorer wiring

**Files:** Modify `src/qpocket/viz/explorer.py`; Test `tests/test_notebook_renders.py`

**Interfaces:** Consumes Tasks 4–6.

- [ ] **Step 1: Failing render assertions** in `test_a_site_run_leaves_no_error_boxes`: the shown text contains `"percentile"`, `"Pockets shown"` and the table header `"shared volume"`.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** in the existing cells, keeping their order and data flow:
  - cut controls: `shape_cut`, `chem_cut`, `combined_cut` become `mo.ui.slider(start=0.5, stop=99.5, step=0.5, value=<the run's percentile>, label="… cut (percentile)")`; the `cut_by` radio goes; the preview recomputes on every slide; "Apply to this run" unchanged;
  - Clustering section: `two_stage.collapsed_figure` and the cluster table (`mo.ui.table(..., selection="single")`); a row selection (or a leaf click, if `mo.ui.plotly` reports one on the leaf markers) selects the cluster and shows `subtree_figure` below; when the preview differs from the applied clustering, the line "Preview — apply to update the page";
  - distance-matrix section → `cluster_map_figure` with a shape / chemistry / combined switch, and `member_block` for a selected pair;
  - landscape → `landscape_figure`; the more recent of a lasso and a table-row selection sets the clouds' starting cluster;
  - pocket clouds → `mo.ui.multiselect(options=cloud_options(...), value=default_cloud_selection(...), label="Pockets shown")` plus the cap note; panel modes and colour modes unchanged and computed over whole clusters;
  - every Plotly figure passes through `style.apply`; section headings stay; no panel letters.
- [ ] **Step 4: Run** the render tests, then the full suite. Expected: all pass.
- [ ] **Step 5: Commit** `feat(explorer): percentile sliders, collapsed trees, cluster map, cloud picker`.

### Task 8: Docs and the real-run check

- [ ] Update `docs/user-guide/distance-analysis.md` (percentile cuts, collapsed tree, cluster table, cluster map), the recluster examples in `README.md` / `RUNBOOK.md`, and `tests/golden/CHANGES.md` (cuts expressed as percentiles; PARG data absent).
- [ ] Full suite green; commit `docs: percentile cuts and explorer navigation`.
- [ ] On `~/qpocket_runs/EV71_2A_cavity`: `qpocket site shape ~/qpocket_runs/EV71_2A_cavity --site S1-G127` (shape only, no APBS), then serve the explorer headless and confirm the site renders without error boxes; report the cluster count at p15 and the render time.
