# Binding-Site Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `qpocket run` discovers every ligand-occupied binding site across all aligned structures and writes one unfeaturised point cloud per site per structure; the explorer then runs shape clustering, featurisation, two-stage clustering and cross-docking per selected site.

**Architecture:** The per-structure chain becomes align → clean → find ligands → protonate (all ligands) → contacts per ligand → cavity detection (whole protein, saved). A global step groups ligands into sites by single-linkage heavy-atom distance. A second parallel pass cuts each site's cloud out of each structure's saved cavities and caches geodesics. Each `sites/site_NN/` is a run-root-shaped workspace, so existing distance, recluster and screening code runs on it.

**Tech Stack:** Python 3.11, numpy/scipy (cKDTree), pandas, Biopython, RDKit, ProLIF, marimo, click, OmegaConf, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-binding-site-discovery-design.md`

## Global Constraints

- Layout version 3 (`io/layout.py` `LAYOUT_VERSION = 3`); layout-2 runs are not readable and must be re-run.
- `reference.ligand_id`, `reference.ligand_chain_id`, `reference.pocket_mode`, `reference.bounding_box` are obsolete: warn and ignore (via `LEGACY_KEYS` → `None`).
- A ligand: non-polymer, non-water HET residue, not in `cleaning.allowed_cofactors` (the per-run cleaning set), with ≥ `sites.min_heavy_atoms` heavy atoms. Carbon-free groups (ions, SO4, PO4, NO3, …) are never ligands. No built-in name list (`constants.HET_TO_REMOVE` is not used for sites) and no `sites.include`/`sites.exclude` (user decision 2026-10-26).
- Allowed cofactors stay in the protein and never define a site.
- Every site seen at least once is kept.
- Every structure gets a cloud at every site; `protein_state` per site is `BOUND_STATE` (≥1 ligand of the site) or `UNBOUND_STATE`, or `CAVITY_CLOSED` when the box holds no cavity points.
- `qpocket run` never runs feature providers (APBS etc.) and never computes distance matrices.
- One protonated complex per structure holding every ligand.
- The "Interactions" crop is spatial: distance to the aligned-frame atom coordinates of the site's contact residues.
- Site ids `site_01, site_02, …` ordered by number of holo structures (descending), ties by first appearance in input order.
- Alignment frame: unchanged sources. `pdb_query` needs `reference.pdb_id` (it is the query); `fragment_screen` uses `reference_path`, else the existing `reference_index` fallback — that fallback is the spec's "auto-pick". No new picking code.
- Run tests with the venv on PATH: `export PATH=$PWD/.venv/bin:$PATH`.
- Reuse `pipe`, `parallel.run_map`, `guarded`, fingerprinted resume; add no parallel framework or dependency.

`sites` config block, exact defaults:

```yaml
sites:
  min_heavy_atoms: 6
  link_cutoff: 4.0     # Å
  box_padding: 4.0     # Å
```

## Review Focus

1. **No ligand in any structure.** The run finishes, `sites/sites.csv` has a header and no rows, and the explorer says "No ligand-bound site was found in this run". Test in Task 8.
2. **Closed site in some structures.** No cavity point in the box gives a `CAVITY_CLOSED` row in the site's `master.csv` and no `binding_site.csv`. Shape clustering skips it. Fewer than 2 clouds raises `ValueError("site_NN has fewer than 2 pockets to compare")`. Tests in Tasks 7 and 10.
3. **The same resname twice in one structure** (homodimer, two copies of a ligand). Expect two `Ligand` rows, two files, and contacts that are not merged. Tests in Tasks 3 and 5.
4. **Re-running on a finished run.** Structures are skipped by fingerprint, the site step is skipped when `grouping.json` matches, and per-site results survive. Test in Task 8.
5. **Old inputs.** A config with `reference.ligand_id` warns and runs. A layout-2 run directory does not show in the explorer's run list. Tests in Tasks 1 and 14.

---

## Milestone 1 — the run discovers sites and writes clouds

Execution order: 1, 2, 3, 4, 5, 6, **9**, 7, 8, then 10 onward. Task 7 needs Task 9's unfeaturised geodesics, but Task 9 lives with the other distance work.

### Task 1: `sites` config and obsolete reference keys

**Files:**
- Modify: `src/qpocket/config.py` (`ReferenceParams` :29-48, `QPocketConfig` :334, `LEGACY_KEYS` :418, `validate_config` :525)
- Modify: `conf/config.yaml`, `src/qpocket/conf/config.yaml` (identical comments, see commit b002021)
- Modify: `src/qpocket/pipelines/run.py` (`run_qpocket_pipeline` drops `ligand_id`), `src/qpocket/cli/run.py` (drop `--ligand-id` option)
- Test: `tests/test_config_migration.py`, `tests/test_config_validation.py`

**Interfaces:**
- Produces: `SitesConfig(min_heavy_atoms: int = 6, include: List[str] = [], exclude: List[str] = [], link_cutoff: float = 4.0, box_padding: float = 4.0)` frozen dataclass; `QPocketConfig.sites: SitesConfig`.

- [ ] **Step 1: Write failing tests**

```python
def test_ligand_id_is_obsolete(tmp_path):
    cfg = OmegaConf.create({"reference": {"ligand_id": "LKQ", "ligand_chain_id": "A",
                                          "pocket_mode": "bounding_box"}})
    warnings = migrate_legacy(cfg)
    assert any("reference.ligand_id is obsolete" in w for w in warnings)
    assert any("reference.pocket_mode is obsolete" in w for w in warnings)

def test_sites_defaults():
    s = QPocketConfig().sites
    assert (s.min_heavy_atoms, s.link_cutoff, s.box_padding) == (6, 4.0, 4.0)
    assert s.include == [] and s.exclude == []

@pytest.mark.parametrize("key,value", [("link_cutoff", 0.0), ("box_padding", -1.0), ("min_heavy_atoms", 0)])
def test_sites_validation(key, value):
    with pytest.raises(ValueError, match=f"sites.{key}"):
        validate_config(evolve(QPocketConfig(), f"sites.{key}", value))
```

- [ ] **Step 2: Run them and confirm they fail.** Run `pytest tests/test_config_migration.py tests/test_config_validation.py -v`. Expected: FAIL (no `sites`, no warning).
- [ ] **Step 3: Implement.** Add `SitesConfig` and `QPocketConfig.sites`. Delete `ligand_id`, `ligand_chain_id`, `pocket_mode` and `bounding_box` from `ReferenceParams` (also delete `BoundingBoxParams`). Add these four `reference.*` keys to `LEGACY_KEYS` with value `None`. In `validate_config`, require `link_cutoff > 0`, `box_padding >= 0` and `min_heavy_atoms >= 1`. Each error message names the key, e.g. `sites.link_cutoff`. Add the YAML block from Global Constraints to both config files, with comments.
- [ ] **Step 4: Run** `pytest tests/test_config_*.py tests/test_cli.py -v`. Expected: PASS. Fix the callers that grep finds for `ligand_id` in config or CLI code. Leave the structure-chain uses, because Task 5 removes them.
- [ ] **Step 5: Commit** with the message `feat: sites config; the reference ligand is obsolete`.

### Task 2: Layout version 3 and site workspaces

**Files:**
- Modify: `src/qpocket/io/layout.py`
- Test: `tests/test_layout.py`

**Interfaces:**
- Produces:
  - `Workspace` gains `sites: Path` (`<run>/sites`) and `cavities: Path` (`<run>/structures/cavities`).
  - `workspace(root)` recognises a site directory: when `root.parent.name == "sites"`, `structures/original/aligned/cleaned/extracted_ligands/cavities` resolve under `root.parent.parent`; `pockets/metadata/screening/decoys/actives` resolve under `root`; `sites` is `root.parent`.
  - `site_workspace(ws: Workspace, site_id: str) -> Workspace` builds the site's workspace and creates its directories.
  - `run_root(ws: Workspace) -> Path` returns the run directory for both kinds of workspace.
  - `LAYOUT_VERSION = 3`.

- [ ] **Step 1: Write failing tests**

```python
def test_site_workspace_keeps_run_structures(tmp_path):
    ws = layout.create(str(tmp_path), "run")
    site = layout.site_workspace(ws, "site_01")
    assert site.root == ws.root / "sites" / "site_01"
    assert site.original == ws.original and site.cleaned == ws.cleaned
    assert site.pockets == site.root / "pockets" and site.metadata.is_dir()
    assert layout.workspace(site.root) == site
    assert layout.run_root(site) == ws.root

def test_layout_version_is_3(tmp_path):
    ws = layout.create(str(tmp_path), "run")
    assert layout.read_run_record(ws.root)["layout_version"] == 3
```

- [ ] **Step 2: Run** `pytest tests/test_layout.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the interface above. The one `workspace()` branch fixes every caller that is handed a site directory. That includes `crossdock._sequences` (`arms/screen/crossdock.py:30-37`), which then finds `structures/original`.
- [ ] **Step 4: Run** `pytest tests/test_layout.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: layout 3 with per-site workspaces`.

### Task 3: Ligand finding (pure)

**Files:**
- Create: `src/qpocket/core/sites.py`
- Modify: `src/qpocket/core/types.py` (add `Ligand`), `src/qpocket/core/pocket.py:13` (`is_residue_ligand` gains include/exclude)
- Test: `tests/test_sites_core.py`

**Interfaces:**
- Produces:
  - `Ligand(NamedTuple)` with fields `pdb_id: str, chain: str, resname: str, resseq: int, icode: str, n_heavy: int, file: str = "", smiles: str = ""`.
  - Property `key -> str`, formatted `f"{pdb_id}:{chain}:{resname}:{resseq}{icode.strip()}"`.
  - Methods `to_json()` and `from_json()` in the style of `Structure`.
  - `is_residue_ligand(residue, allowed_cofactors, include=(), exclude=()) -> bool`. `include` overrides `HET_TO_REMOVE` but never `allowed_cofactors`.
  - `find_ligands(structure, pdb_id: str, allowed_cofactors: Sequence[str], min_heavy_atoms: int, include: Sequence[str] = (), exclude: Sequence[str] = ()) -> List[Tuple[Ligand, residue]]`, in chain then residue order. Heavy atoms are atoms whose element is not H or D.

- [ ] **Step 1: Write failing tests.** They use `tests/fixtures/complexes/1STP.pdb` (biotin, BTN) and `3PTB.pdb` (benzamidine, BEN).

```python
def test_finds_biotin():
    found = find_ligands(_parse("1STP"), "1STP", [], 6)
    assert [l.resname for l, _ in found] == ["BTN"]
    assert found[0][0].key.startswith("1STP:A:BTN:")

def test_min_heavy_atoms_and_lists():
    s = _parse("3PTB")
    assert find_ligands(s, "3PTB", [], 10) == []          # BEN has 9 heavy atoms
    assert find_ligands(s, "3PTB", [], 6, exclude=["BEN"]) == []
    assert find_ligands(s, "3PTB", ["BEN"], 6, include=["BEN"]) == []   # cofactor wins

def test_same_resname_twice_gives_two_ligands():
    s = _two_copies("1STP")      # helper: copy BTN into chain B with resseq+1
    found = find_ligands(s, "1STP", [], 6)
    assert len({l.key for l, _ in found}) == 2
```

- [ ] **Step 2: Run** `pytest tests/test_sites_core.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `Ligand`, the extended `is_residue_ligand` (existing callers keep their behaviour through the defaults) and `find_ligands`.
- [ ] **Step 4: Run** `pytest tests/test_sites_core.py tests/test_pocket_core.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: find every drug-like ligand in a structure`.

### Task 4: Site grouping and site boxes (pure)

**Files:**
- Modify: `src/qpocket/core/sites.py`
- Test: `tests/test_sites_core.py`

**Interfaces:**
- Produces:
  - `group_ligands(coords: Sequence[np.ndarray], cutoff: float) -> np.ndarray`. It returns integer labels, one per ligand: single linkage, where two ligands link if any heavy-atom pair is within `cutoff`. Use `scipy.spatial.cKDTree` per pair or across all atoms, then `scipy.sparse.csgraph.connected_components`.
  - `site_ids(labels: np.ndarray, pdb_ids: Sequence[str]) -> List[str]` maps each ligand to `site_NN`. Order is by distinct holo `pdb_id` count descending, ties by first appearance; the number is zero-padded to 2 digits, or to 3 when there are more than 99 sites.
  - `site_box(coords: Sequence[np.ndarray], padding: float) -> Tuple[np.ndarray, np.ndarray]` reuses `core.geometry.compute_pocket_bounding_box(np.vstack(coords), padding)`.
  - `points_in_box(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write failing tests**

```python
A = np.array([[0, 0, 0], [1, 0, 0]]); B = A + [4.5, 0, 0]; C = A + [9.0, 0, 0]; FAR = A + [50, 0, 0]

def test_overlapping_join_far_split():
    labels = group_ligands([A, A + 0.5, FAR], cutoff=4.0)
    assert labels[0] == labels[1] != labels[2]

def test_bridge_chains_sites():          # accepted per spec decision 13
    labels = group_ligands([A, B, C], cutoff=4.0)
    assert len(set(labels)) == 1

def test_site_ids_by_holo_count():
    ids = site_ids(np.array([0, 1, 1, 1]), ["P1", "P1", "P2", "P3"])
    assert ids == ["site_02", "site_01", "site_01", "site_01"]

def test_points_in_box():
    lo, hi = site_box([A], padding=1.0)
    pts = np.array([[0.5, 0, 0], [5, 0, 0]])
    assert points_in_box(pts, lo, hi).tolist() == [[0.5, 0, 0]]
```

- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** them. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: group ligands into binding sites`.

### Task 5: Per-structure chain (ligands, protonation, contacts, cavities)

**Files:**
- Modify: `src/qpocket/pipelines/structure.py`, `src/qpocket/core/types.py` (`Structure`), `src/qpocket/io/interactions.py` (`write_complex` :222, `interactions_for` :142), `src/qpocket/plugins/contacts/prolif.py`
- Modify: `src/qpocket/pipelines/reference.py` (no pocket: drop `reference_pocket_box`, `_reference_pocket`, centroid and box)
- Delete: `src/qpocket/plugins/pockets/` and the `POCKETS` port, if nothing else uses them. Check with `grep -rn POCKETS src tests`, then update `test_stage_plugins.py`, `test_registry.py` and doctor.
- Test: `tests/test_structure.py`, `tests/test_interactions_io.py`, `tests/test_reference_pocket.py` (delete what tests the removed functions)

**Interfaces:**
- Consumes: `find_ligands`, `Ligand` (Task 3); `SitesConfig` (Task 1).
- Produces:
  - `Structure` drops `points`, `ligand_file`, `ligand_id` and `smiles`.
  - `Structure` gains `ligands: Tuple[Ligand, ...] = ()`, `contacts: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()` (ligand key → `RES_NUM_CHAIN|Type` entries) and `cavities_file: Optional[str] = None`.
  - Run-level `protein_state` is `BOUND_STATE` if any ligand, `UNBOUND_STATE` otherwise, or `FAILED` / `NO_ANY_CAVITY_FOUND`.
  - `master_row()` columns: `pdb_id, rmsd_to_ref, aligned_file, [aux_chain_mapping], cleaned_file, protonated_file, cavities_file, n_ligands, protein_state`.
  - `StructureParams` drops `pocket`, `pocket_params`, `reference_ligand_id`, `pocket_definition` and `interaction_padding` (the last two move to the site step). It gains `min_heavy_atoms`.
  - Steps:
    - `find_ligands_step(p, ws, record) -> Structure` writes `extracted_ligands/{pdb_id}_{chain}_{resname}_{resseq}{icode}.pdb` and fills `Ligand.file` and `Ligand.smiles` using the existing `ligand_smiles`.
    - `protonate(ws, record)` writes every ligand into one complex.
    - `read_contacts(p, ws, record)` reads contacts per ligand.
    - `detect_cavities(ws, record) -> Structure` writes `structures/cavities/{pdb_id}.csv` with columns `x,y,z,cavity`, where `cavity` is the detection index.
  - `write_complex(path, protein_pqr, ligands: Sequence[Tuple[Chem.Mol, str]])` writes ligand i at chain "A", resseq `900 + i`.
  - CONTACTS plug-in signature: `fn(complex_path, ligand_id: str, params, ligand_resseq: Optional[int] = None)`. `interactions_for(..., ligand_resseq=None)` splits out only the residue with that resname and resseq.
  - `process()` pipe: `align, clean, find_ligands_step, protonate, read_contacts, detect_cavities`. `locate`, `first_ligand`, `refine_and_featurize`, `geodesics` and `finish` leave the chain. `featurize()` stays, because Task 11 uses it.

- [ ] **Step 1: Write failing tests**

```python
def test_chain_records_all_ligands_and_cavities(tmp_path, one_structure_run):
    rec = process(ref, params, ws, raw_1stp)
    assert [l.resname for l in rec.ligands] == ["BTN"]
    assert Path(rec.cavities_file).exists()
    cav = pd.read_csv(rec.cavities_file)
    assert list(cav.columns) == ["x", "y", "z", "cavity"] and len(cav) > 0
    assert dict(rec.contacts)[rec.ligands[0].key]           # non-empty contacts
    assert not (ws.pockets / "1STP").exists()               # no per-structure pocket any more

def test_two_same_name_ligands_keep_separate_contacts(tmp_path):
    out = write_complex(tmp_path / "c.pdb", pqr, [(mol_a, "BTN"), (mol_b, "BTN")])
    a = interactions_for(out, "BTN", ligand_resseq=900)
    b = interactions_for(out, "BTN", ligand_resseq=901)
    assert a != b
```

- [ ] **Step 2: Run** `pytest tests/test_structure.py tests/test_interactions_io.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the interface above. `Structure.from_json` must rebuild `Ligand`s. The fingerprint in `fingerprint()` already covers `StructureParams`.
- [ ] **Step 4: Run** `pytest tests -q -m "not golden"`. Expected: PASS. Rewrite or delete tests of removed steps. Leave tests of explorer, crossdock and distance that need site folders failing only if they belong to later tasks, and list them in the commit message.
- [ ] **Step 5: Commit** with the message `feat: structures record every ligand, its contacts, and all cavities`.

### Task 6: Site discovery step

**Files:**
- Create: `src/qpocket/pipelines/sites.py`
- Test: `tests/test_sites_pipeline.py`

**Interfaces:**
- Consumes: `Structure.ligands`, `Structure.contacts` (Task 5); `group_ligands`, `site_ids` (Task 4).
- Produces:
  - `discover_sites(cfg: QPocketConfig, ws: Workspace, records: Sequence[Structure]) -> pd.DataFrame`. It returns and writes `sites/sites.csv` with columns `site_id, n_ligands, n_holo_structures, centre_x, centre_y, centre_z, extent, resnames, contact_residues`:
    - `extent` is the maximum pairwise heavy-atom distance in Å.
    - `resnames` and `contact_residues` are `;`-joined and sorted.
  - It also writes:
    - `structures/ligands.csv`: `Ligand` fields plus `site_id`.
    - `sites/grouping.json`: `{"link_cutoff", "box_padding", "ligands_hash"}`.
    - `sites/site_NN/ligands.pdb`: the union of the site's ligand files.
  - It skips itself when `grouping.json` matches the cutoff, the padding and the sha256 of `structures/ligands.csv`.
  - `read_sites(ws) -> pd.DataFrame`.

- [ ] **Step 1: Write failing tests.** Use synthetic `Structure` records with `Ligand.file` PDBs written into `tmp_path`.

```python
def test_two_sites_from_three_structures(tmp_path):
    sites = discover_sites(cfg, ws, [rec("P1", [lig_a]), rec("P2", [lig_a2, lig_far]), rec("P3", [])])
    assert list(sites.site_id) == ["site_01", "site_02"]
    assert sites.set_index("site_id").loc["site_01", "n_holo_structures"] == 2
    assert (ws.sites / "site_01" / "ligands.pdb").exists()

def test_no_ligands_gives_empty_table(tmp_path):
    sites = discover_sites(cfg, ws, [rec("P1", [])])
    assert sites.empty and (ws.sites / "sites.csv").read_text().startswith("site_id,")

def test_rerun_with_same_inputs_is_skipped(tmp_path):
    discover_sites(cfg, ws, recs); stamp = (ws.sites / "sites.csv").stat().st_mtime_ns
    discover_sites(cfg, ws, recs)
    assert (ws.sites / "sites.csv").stat().st_mtime_ns == stamp
```

- [ ] **Step 2: Run** `pytest tests/test_sites_pipeline.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Ligand heavy-atom coordinates come from `Ligand.file`, read with biopandas.
- [ ] **Step 4: Run** the tests again. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: discover binding sites from all ligands`.

### Task 7: Per-site cloud extraction

**Files:**
- Modify: `src/qpocket/pipelines/sites.py`, `src/qpocket/pipelines/collect.py` (`collect_run` :121-160, actives with a ligand file), `src/qpocket/arms/screen/ligands.py:19-45` (`distinct_actives`)
- Test: `tests/test_sites_pipeline.py`, `tests/test_actives_smi.py`

**Interfaces:**
- Consumes: `site_workspace` (Task 2); `points_in_box`, `site_box` (Task 4); `discover_sites` output (Task 6); `compute_geodesic` (still featurized-based until Task 9, so call it after Task 9 or pass `binding_site.csv`; see Step 3).
- Produces:
  - `SiteRecord(NamedTuple)` with fields `pdb_id, rmsd_to_ref, aligned_file, cleaned_file, protonated_file, protein_state, ligand_file, ligand_id, potential, observed, actives`. `actives` is a tuple of `(smiles, ligand_name, ligand_file)`, one per site ligand of that structure. `ligand_file` and `ligand_id` hold the site's largest ligand in that structure (for cross-docking). `master_row()` and `to_json()` work as in `Structure`.
  - `extract_site(cfg, ws, site: pd.Series, record: Structure) -> SiteRecord`:
    - Writes `sites/<id>/pockets/<pdb_id>/binding_site.csv` with all cavity points inside the site box.
    - In `Interactions` mode it crops with `crop_pocket_by_interactions(points, contact_xyz, padding=cfg.reference.interaction_padding)`. `contact_xyz` is the aligned-frame atom coordinates of `contact_residues`, taken from every holo member's `aligned_file` and saved once as `sites/<id>/contact_atoms.csv`.
    - Then calls `compute_geodesic(binding_site.csv, radius)`.
  - `extract_all_sites(cfg, ws, sites: pd.DataFrame, records) -> None` runs `parallel.run_map` over (site × structure) and then `collect_run(site_ws, site_records, failures, features=(), compression)` per site.
  - `collect_run` writes `actives.smi` with the header `SMILES\tPDB_ID\tLigand_Name\tLigand_File`, one row per entry of `record.actives`.
  - `distinct_actives` takes crystal files from `Ligand_File` when that column exists, otherwise from `master.csv` as today.

- [ ] **Step 1: Write failing tests**

```python
def test_holo_and_apo_clouds_cut_by_same_box(tmp_path):
    out = extract_site(cfg, ws, site_row, apo_record)
    assert out.protein_state == "UNBOUND_STATE"
    pts = pd.read_csv(ws.sites / "site_01" / "pockets" / "APO1" / "binding_site.csv")
    lo, hi = box_of(site_row); assert ((pts[["x","y","z"]] >= lo) & (pts[["x","y","z"]] <= hi)).all().all()

def test_closed_site(tmp_path):
    out = extract_site(cfg, ws, site_row, record_with_no_points_in_box)
    assert out.protein_state == "CAVITY_CLOSED"
    assert not (ws.sites / "site_01" / "pockets" / out.pdb_id / "binding_site.csv").exists()

def test_two_ligands_at_one_site_are_two_actives(tmp_path):
    out = extract_site(cfg, ws, site_row, record_with_two_btn)
    assert len(out.actives) == 2 and len({a[2] for a in out.actives}) == 2

def test_distinct_actives_reads_ligand_file_column(tmp_path):
    df = distinct_actives(actives_with_file_column, master_csv)
    assert df.loc[0, "crystal_files"] == "lig_a.pdb;lig_b.pdb"
```

- [ ] **Step 2: Run** `pytest tests/test_sites_pipeline.py tests/test_actives_smi.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Do Task 9 Step 3 first (geodesics from `binding_site.csv`) so that `compute_geodesic` accepts an x,y,z-only CSV. Task 9's tests can follow afterwards. `observed` for a holo structure is the union of its site ligands' contacts. For an apo structure it is `("no_ligand_found",)`, which `viz/interactions.py:137` relies on. `potential` is computed as in the old `_potential` (residues within 4 Å of the site cloud).
- [ ] **Step 4: Run** the tests again. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: one cloud per site per structure, collected per site`.

### Task 8: Wire the run; snapshot the config; golden

**Files:**
- Modify: `src/qpocket/pipelines/run.py:52-81`, `src/qpocket/pipelines/collect.py` (run-level: `master.csv` and `failures.csv` only)
- Modify: `src/qpocket/config.py`. Add `save_run_config(cfg, ws)`, which writes `metadata/config.yaml` via `str(cfg)`, and `load_run_config(run_dir) -> QPocketConfig`.
- Modify: `tests/golden/test_golden.py`, `tests/golden/CHANGES.md`, `scripts/make_golden.py` (only if it names old paths)
- Test: `tests/test_orchestrators.py`

**Interfaces:**
- Consumes: Tasks 5–7.
- Produces: `run(cfg) -> Path`:
  1. ingest, then `build_reference` (alignment only);
  2. `run_map(process)`;
  3. run-level `master.csv` and `failures.csv`;
  4. `discover_sites`;
  5. `extract_all_sites`;
  6. `save_run_config`.

  `compute_spectral_distance_matrix` is no longer called.

- [ ] **Step 1: Write failing tests** in `test_orchestrators.py`, monkeypatching cavity detection and protonation the way the existing orchestrator tests do.

```python
def test_run_writes_sites_and_no_matrices(tmp_path, fake_tools):
    root = run(cfg_two_structures(tmp_path))
    assert (root / "sites" / "sites.csv").exists()
    assert (root / "sites" / "site_01" / "metadata" / "master.csv").exists()
    assert not list(root.rglob("*distance_matrix.csv"))
    assert not list(root.rglob("featurized.csv"))
    assert load_run_config(root).sites.link_cutoff == 4.0

def test_rerun_keeps_site_results(tmp_path, fake_tools):
    root = run(cfg); marker = root / "sites" / "site_01" / "metadata" / "keep.txt"; marker.write_text("x")
    run(cfg); assert marker.exists()

def test_run_with_no_ligands(tmp_path, fake_tools_apo):
    root = run(cfg); assert pd.read_csv(root / "sites" / "sites.csv").empty
```

- [ ] **Step 2: Run** `pytest tests/test_orchestrators.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `run`. Then regenerate the golden output with `python scripts/make_golden.py tests/golden/expected`. Add a `CHANGES.md` entry: "layout 3: per-site clouds, no features or matrices in `qpocket run`". Point `test_golden.py`'s file list at the new layout.
- [ ] **Step 4: Run** `pytest tests -q` (including `-m golden`). Expected: PASS, except explorer, crossdock and slurm tests owned by Tasks 13–15, which are listed in the commit.
- [ ] **Step 5: Commit** with the message `feat: qpocket run discovers sites and stops after clouds`.

---

## Milestone 2 — per-site analysis steps (library and CLI)

### Task 9: Geodesics from `binding_site.csv`; features joined at fingerprint time

**Files:**
- Modify: `src/qpocket/pipelines/distance.py` (`compute_geodesic` :35, `data_processing_spectral` :135, `pocket_manifest` :220), `src/qpocket/io/cache.py`, `src/qpocket/pipelines/tasks.py:181-220`
- Test: `tests/test_geodesic_cache.py`, `tests/test_spectral_distance.py`

**Interfaces:**
- Produces:
  - `compute_geodesic(csv_path, neighbor_radius)` reads only x,y,z and keys the cache on that file's hash.
  - `cache.save_geodesic(path, d_geo, source_hash, radius)` no longer stores features. `load_geodesic(path) -> np.ndarray` returns D only.
  - `pocket_manifest(cfg, ws, require_features: bool) -> (List[Path], List[str])` lists `binding_site.csv` paths for pockets that have one (and, when `require_features`, also a `featurized.csv`).
  - `data_processing_spectral` reads F from `featurized.csv` beside each cache, minus x,y,z. It raises `ValueError` if the row count differs from D.

- [ ] **Step 1: Write failing tests.**
  - `test_geodesic_from_unfeaturised_cloud`: a CSV with only x,y,z produces a cache, and `load_geodesic` returns an (N, N) matrix.
  - `test_cache_survives_featurisation`: writing `featurized.csv` afterwards leaves the cache current.
  - `test_feature_rows_must_match`: a mismatched `featurized.csv` raises `ValueError`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** `structure.featurize` writes rows in `binding_site.csv` order (`tables.read_point_table`). Assert that, don't reorder.
- [ ] **Step 4: Run** `pytest tests/test_geodesic_cache.py tests/test_spectral_distance.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: geodesics are computed before featurisation`.

### Task 10: Shape-only distances and clustering; readers accept shape alone

**Files:**
- Modify: `src/qpocket/pipelines/distance.py`, `src/qpocket/pipelines/clustering.py` (`cluster_pockets` :88, `write_clustering` :190), `src/qpocket/arms/recluster.py:47-80`, `src/qpocket/viz/two_stage.py:15-48`
- Test: `tests/test_spectral_distance.py`, `tests/test_recluster.py`, `tests/test_clustering_two_stage.py`

**Interfaces:**
- Produces:
  - `shape_fingerprints(cfg, cache_paths) -> Dict[str, np.ndarray]` gives the eigenvalues only, via `sp.spectral_fingerprint(D, np.zeros((n, 0)), k, sigma)[0]`.
  - `compute_shape_matrix(cfg, ws) -> Path` writes `metadata/shape_distance_matrix.csv` (Euclidean on eigenvalues, the same as `distance_matrices`' `d_shape`). It deletes any stale `feature_distance_matrix.csv` and `spectral_distance_matrix.csv`, and raises `ValueError(f"{site} has fewer than 2 pockets to compare")` when fewer than 2 pockets remain.
  - `cluster_shape_only(stage: StageClustering, d_shape) -> ClusterAssignment` produces `params["mode"] = "shape_only"` and shape-medoid representatives. `write_clustering(meta, ids, assignment, combined=d_shape, file_by_id)`.
  - `compute_spectral_distance_matrix(cfg, ws)` (two-stage) uses `pocket_manifest(require_features=True)`.
  - `RunMatrices.chem` and `.combined` become `Optional`. `load_matrices` returns `chem=None` and `combined=shape` when only the shape matrix exists.
  - `recluster.run` accepts `settings.mode == "shape_only"` and raises `ValueError("two-stage clustering needs featurised pockets; run Featurise first")` when chem is missing.
  - `ClusteringParams.mode` gains the literal `"shape_only"`.

- [ ] **Step 1: Write failing tests.**
  - `test_shape_matrix_equals_two_stage_shape`: on the fixture caches, shape-only `d_shape` equals `build_distance_matrices(...)[0]` (`atol 1e-10`).
  - `test_shape_only_needs_two_pockets`: one pocket raises the `ValueError`.
  - `test_load_matrices_shape_only`: `chem is None`.
  - `test_recluster_two_stage_without_chem_raises`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/test_spectral_distance.py tests/test_recluster.py tests/test_clustering_two_stage.py tests/test_apply_clustering.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: shape-only clustering without featurisation`.

### Task 11: Site steps library (featurise, shape, two-stage, regroup)

**Files:**
- Create: `src/qpocket/arms/sites.py`
- Test: `tests/test_site_steps.py`

**Interfaces:**
- Consumes: `load_run_config` (Task 8); `structure.featurize`; `compute_shape_matrix`, `cluster_shape_only`, `compute_spectral_distance_matrix` (Task 10); `discover_sites`, `extract_all_sites` (Tasks 6–7).
- Produces (each takes `run_dir: Path` and resolves the site with `layout.site_workspace`):
  - `featurize_site(run_dir, site_id, workers: Optional[int] = None) -> List[str]`. It featurises each pocket with `binding_site.csv` using `structure.featurize(csv, cleaned_file, cfg.features.enabled, cfg.charges.use, options, charge_params)` through `parallel.run_map`. A pocket whose `featurized.csv` is newer than its `binding_site.csv` is skipped. It returns the pdb_ids featurised.
  - `shape_cluster_site(run_dir, site_id) -> Path` runs `compute_shape_matrix`, then `cluster_shape_only` with `cfg.distance.clustering.shape`, then writes the clustering. It archives any previous clustering into `clustering_history/`, reusing recluster's archive code (extract `recluster._archive(meta)` and call it from both).
  - `two_stage_site(run_dir, site_id) -> Path` runs `compute_spectral_distance_matrix` for the site and raises if Featurise has not run.
  - `regroup_sites(run_dir, link_cutoff: float) -> pd.DataFrame` re-runs `discover_sites` with the new cutoff, using `Structure` records from the run's sidecars (`layout.read_part`). A new site whose ligand-key set equals an old site's keeps that old folder, renamed to its new id via a temp name. Every other old site folder is deleted. `extract_all_sites` then runs for the new or changed sites only.
  - `site_status(run_dir, site_id) -> Dict[str, bool]` with keys `clouds, geodesics, shape_clustered, featurised, two_stage, crossdock`, each read from files.

- [ ] **Step 1: Write failing tests.** They run on a small finished run built as in Task 8 with fake tools.
  - `test_shape_then_featurise_then_two_stage` checks the status flips in order.
  - `test_two_stage_before_featurise_raises`.
  - `test_regroup_keeps_unchanged_site`: put a marker in a site that keeps its ligands, regroup with a smaller cutoff that splits a different site, and the marker survives under the new id.
  - `test_featurise_skips_done`.
- [ ] **Step 2: Run** `pytest tests/test_site_steps.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests again. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: per-site analysis steps`.

### Task 12: `qpocket site` CLI

**Files:**
- Create: `src/qpocket/cli/site.py`; modify `src/qpocket/cli/__init__.py` (or wherever commands are registered) to add the group
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 11.
- Produces: a click group `site` with these subcommands:
  - `qpocket site list RUN_DIR` prints `sites.csv` plus the `site_status` columns.
  - `qpocket site featurize RUN_DIR --site site_01 [--workers N]`.
  - `qpocket site shape RUN_DIR --site site_01`.
  - `qpocket site cluster RUN_DIR --site site_01` (two-stage).
  - `qpocket site regroup RUN_DIR --cutoff 3.0`.

  The existing `qpocket recluster` takes a site directory unchanged, thanks to Task 2.

- [ ] **Step 1: Write failing tests.** Use `CliRunner` with each subcommand on the Task 11 fixture run. Expect exit code 0 and the expected files. A missing `--site` exits non-zero with `Missing option '--site'`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement** thin wrappers.
- [ ] **Step 4: Run** `pytest tests/test_cli.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: qpocket site commands`.

### Task 13: SLURM stages for the new run and per-site featurise

**Files:**
- Modify: `src/qpocket/pipelines/slurm.py:44-90` (`STAGES`), `src/qpocket/pipelines/tasks.py` (`task_*`, `manifest_ids`, `run_task`), `src/qpocket/cli/slurm.py`
- Test: `tests/test_slurm_render.py`, `tests/test_parallel_stages.py`

**Interfaces:**
- Produces:
  - Run stages: `prepare` (artifact `metadata/reference.json`), `structure` (array, artifact `structures/cavities/{pdb_id}.csv`), `sites` (single: discover, extract, collect and snapshot config; artifact `sites/sites.csv`).
  - Per-site stages, generated by `qpocket slurm generate RUN --site site_01`: `site_featurize` (array over the site's `master.csv` pdb_ids with a `binding_site.csv`; artifact `sites/{site}/pockets/{pdb_id}/featurized.csv`) and `site_cluster` (single; artifact `sites/{site}/metadata/spectral_distance_matrix.csv`).
  - `apbs` split stage (`split_apbs_stage`) is removed. Note it in the commit as the builder's call.

- [ ] **Step 1: Write failing tests.**
  - The rendered scripts for a run list stages `prepare, structure, sites` in that order.
  - `--site site_01` renders `site_featurize, site_cluster` with `--site site_01` in each command line.
  - The manifest for `site_featurize` lists only pocketed pdb_ids.
- [ ] **Step 2: Run** `pytest tests/test_slurm_render.py tests/test_slurm_resources.py tests/test_parallel_stages.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same tests. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: slurm stages for site discovery and per-site featurisation`.

---

## Milestone 3 — explorer and screening per site

### Task 14: Run discovery and the explorer's binding-site section

**Files:**
- Modify: `src/qpocket/viz/runs.py:13-48`, `src/qpocket/viz/explorer.py` (cells at :88-150, 164, 182, 344-351, 359-370, 579-590, 1115, 1200, 1213), `src/qpocket/viz/pockets.py` / `viz/ligands.py` (only if they read run-root paths: `grep -n "pockets\|metadata\|screening" src/qpocket/viz/*.py`)
- Test: `tests/test_notebook_renders.py`, `tests/test_explorer_ships.py`, new `tests/test_viz_runs.py`

**Interfaces:**
- Consumes: `read_sites` (Task 6); `site_status`, `featurize_site`, `shape_cluster_site`, `two_stage_site` (Task 11); `load_matrices` with optional chem (Task 10).
- Produces:
  - `is_run(path)` requires `layout_version == 3`, `metadata/master.csv` and `sites/`.
  - `is_complete(path)` means `sites/sites.csv` exists.
  - `PRUNE` gains `"sites"`.
  - Explorer, after the `run_dir` cell:
    - A cell builds `sites_df` and a `site_picker` (a `mo.ui.dropdown` of site ids labelled `site_01 · 12 holo · BTN;BEN`), plus a site table.
    - Next cell: `site_dir = run_dir / "sites" / site_picker.value`. With no sites it shows "No ligand-bound site was found in this run" and stops the downstream cells with `mo.stop`.
    - Every listed cell reads `site_dir` instead of `run_dir`, except run discovery.
    - The pocket-cloud readers use `featurized.csv` when present and otherwise `binding_site.csv`, with feature colouring disabled.
    - A 3D view of `sites/<id>/ligands.pdb` over the frame protein reuses the existing ligand/pocket viewer helpers.
    - A steps cell shows `site_status` and offers `mo.ui.run_button`s: "Shape clustering", "Featurise", "Two-stage clustering" (disabled until featurised, with the label hint "needs Featurise"). Each runs the Task 11 function inside `mo.status.spinner` and prints the equivalent `qpocket site …` and `qpocket slurm generate … --site …` commands. A "Regroup sites" control (`mo.ui.number` cutoff + run button) calls `regroup_sites` after a callout warning that per-site results of changed sites are discarded.

- [ ] **Step 1: Write failing tests.**
  - `test_layout2_run_not_listed`: a directory with `run.json` layout 2 is not a run.
  - `test_v3_run_listed`.
  - `test_notebook_renders` is extended to render the explorer on the Task 11 fixture run in two states: clouds only, and after shape clustering. Both render with no cell errors.
- [ ] **Step 2: Run** `PATH=$PWD/.venv/bin:$PATH pytest tests/test_viz_runs.py tests/test_notebook_renders.py -v`. Expected: FAIL. The render test also needs a dummy `uv` on PATH; see memory note `tests-need-venv-path`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS. Then open it by hand with `qpocket explore <fixture run>`, click Shape clustering, and confirm the landscape section fills.
- [ ] **Step 5: Commit** with the message `feat: explorer picks a binding site and runs its steps`.

### Task 15: Cross-docking and screen modes per site

**Files:**
- Modify: `src/qpocket/arms/screen/crossdock.py:111-160` (snapshot representatives), `src/qpocket/arms/screen/campaign.py:89-160` (use `ws` paths, not `base / "structures" / ...`), `src/qpocket/arms/screen/analysis.py:392`, `src/qpocket/cli/screen.py` (`--site`), `src/qpocket/viz/explorer.py:1189-1230`
- Test: `tests/test_crossdock_prepare.py`, `tests/test_crossdock_analyze.py`, `tests/test_screening.py`

**Interfaces:**
- Consumes: site workspaces (Task 2); per-site `actives.smi` with `Ligand_File` (Task 7); `selected_representatives.csv` from either clustering (Task 10).
- Produces:
  - `crossdock.prepare(site_dir, …)` copies `metadata/selected_representatives.csv` and `metadata/clustering_params.json` into `screening/crossdock/representatives_used/` and records `"clustering_mode"` in `box.json`.
  - `crossdock.analyze` reads representatives from that snapshot.
  - `campaign.prepare(site_dir)` and `run_full_analysis` resolve ligands, pockets and screening through `layout.workspace(site_dir)`.
  - `qpocket screen prepare|analyze RUN_DIR --site site_01` resolves to the site dir.
  - The explorer's cross-docking buttons pass `site_dir`.

- [ ] **Step 1: Write failing tests.**
  - `test_crossdock_prepare_on_site_dir`: receptor repair gets sequences, so `_sequences` is not None for a fixture with an original file.
  - `test_reclustering_after_prepare_does_not_change_docking`: re-cluster after prepare and `analyze` still uses the snapshot.
  - `test_campaign_prepare_on_site_dir` uses the site's `extracted_ligands` (via ws).
- [ ] **Step 2: Run** `pytest tests/test_crossdock_*.py tests/test_screening.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests -q`, including `-m golden`. Expected: all PASS.
- [ ] **Step 5: Commit** with the message `feat: cross-docking and screening run per site`.

### Task 16: Docs

**Files:**
- Modify: `docs/quickstart.md`, `docs/configuration.md`, `docs/user-guide/*` (the reference-ligand passages), `README.md`, `RUNBOOK.md`

- [ ] **Step 1:** Run `grep -rn "ligand_id\|pocket_mode\|bounding_box\|featurized.csv\|spectral_distance_matrix" docs README.md RUNBOOK.md` and rewrite each hit for the site flow: `qpocket run`, then `qpocket explore` or `qpocket site …`. Document the `sites` block with the defaults from Global Constraints.
- [ ] **Step 2:** Run `mkdocs build --strict`. Expected: no warnings.
- [ ] **Step 3:** Commit with the message `docs: binding-site discovery workflow`.
