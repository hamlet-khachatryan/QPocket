# Binding-site discovery: every ligand-occupied pocket, not one reference ligand

Date: 2026-10-25

## Intent

Today a run is anchored on one reference ligand (`reference.ligand_id`): its
pocket is located in the reference, and every structure's cloud is cut at that
place. Anything bound elsewhere — allosteric sites, cofactor sites, fragment
hot spots — is never seen.

The run should instead discover every binding site where a ligand was found in
at least one structure, extract a point cloud for each site from every
structure, and stop. The user then works site by site in the marimo explorer,
choosing which expensive or analytical step to run next: shape-only
clustering, featurisation, two-stage clustering, cross-docking. Each step is a
button, so the notebook controls the order of consecutive steps.

## Goals

- `qpocket run` finds all binding sites automatically and needs no ligand id.
- Every aligned structure gets a cloud at every site, flagged holo or apo, so
  conformations of a site are compared across ligand-bound and empty structures.
- Clouds are stored per site in a layout a person can browse.
- The explorer gets a binding-site picker; every analysis below it applies to
  the selected site.
- Shape-only clustering works without featurisation.
- Heavy steps run from the notebook locally (parallel, cached), and the same
  steps are reachable from the CLI and as generated SLURM jobs.

## Non-goals

- Pooling several sites into one distance matrix (cross-site similarity). One
  site at a time.
- Keeping the old single-reference-ligand path alongside the new one.
- New science in featurisation, fingerprints or clustering algorithms — the
  existing ones are reused per site.
- Backward compatibility of old run folders (layout version bumps; old runs
  must be re-run, as with layout 2).

## Anti-goals

- A site silently dropped because few structures bind there. Every site seen
  at least once is kept; the count is shown so the user filters.
- Apo and holo clouds of one site cut by different rules — they must be
  comparable.
- Featurisation (APBS) running during `qpocket run`.

## Decisions taken by the user

1. **Alignment frame.** Reference is optional. If `reference.pdb_id`/`chain_id`
   (pdb_query) or `fragment_screen.reference_path` is set, it is the frame.
   Otherwise one is picked automatically.
2. **What is a ligand.** Non-polymer, non-water HET groups not kept by the
   run's cleaning set (`cleaning.allowed_cofactors`), with a minimum heavy-atom
   count. No built-in name list and no include/exclude lists (revised by the
   user 2026-10-26: additives that pass the size cut show up as small sites).
3. **Grouping ligands into sites.** After alignment, two ligands belong to the
   same site if any of their heavy atoms lie within a cutoff (Å); sites are the
   connected components (single linkage).
4. **Support.** Keep every site seen at least once; record how many ligands and
   how many holo structures it has.
5. **Coverage.** Every structure gets a cloud at every site, flagged holo/apo.
6. **Cutting the cloud.** Existing cavitomix cavity detection; keep the cavity
   points inside a box around the union of the site's ligands plus padding.
7. **Old mode.** Replaced. `reference.ligand_id` is removed; a config still
   carrying it gets a migration warning. A single-ligand study becomes "pick
   that site in the notebook".
8. **Storage.** Per-site folders:
   `sites/site_NN/<pdb_id>/binding_site.csv`, a run-wide `sites/sites.csv`, and
   a per-site union-ligand PDB for viewing.
9. **Where `qpocket run` stops.** After clouds **and** protonation **and**
   geodesics. Includes ProLIF contacts for every ligand found (right after
   protonation). No APBS / feature providers.
10. **Execution of notebook steps.** In the notebook kernel by default, in
    parallel with `compute.workers`, cached on disk so a re-click skips finished
    work. The same steps also exist as CLI commands and as SLURM jobs.
11. **Buttons, per selected site:** Shape-only clustering, Featurise, Two-stage
    clustering (needs Featurise), Cross-docking (existing prepare/analyse,
    scoped to the site).
12. **One site at a time.**
13. **Chaining** (a long ligand bridging two pockets into one site): accepted.
    `sites.csv` shows site extent; the user can re-run site grouping with a
    tighter cutoff from the notebook.
14. **"Interactions" pocket definition** becomes site-level: every structure's
    cloud at a site is cropped to the union of residues contacted (ProLIF) by any
    ligand of that site, so apo and holo are cut identically. "Solvent" stays
    the cavity-in-box cut.
15. **Notebook.** Extend the existing `qpocket explore` notebook with a
    "Binding sites" section at the top (site table, 3D view, site picker); the
    existing sections read the selected site.
16. **Cofactors** kept by `cleaning.allowed_cofactors` stay part of the protein
    (receptor wall for every site and for cross-docking) and never define a
    site. Only drug-like ligands define sites.
17. **Protonation** writes one complex per structure containing every ligand
    found in it; ProLIF reads each ligand's contacts from that one file.
18. **Site-level residue crop is spatial.** The union of a site's contact
    residues is turned into aligned-frame atom coordinates (from the holo
    structures that contact them); every structure's cloud is cropped by
    distance (`interaction_padding`) to those coordinates. No chain-ID or
    residue-number matching across structures.
19. **Cross-docking representatives** come from whichever clustering ran last
    for the site (shape-only or two-stage); both write
    `metadata/selected_representatives.csv`. Cross-docking records which
    clustering it used and snapshots the representatives, so a later
    re-cluster never silently changes a finished docking.
20. **Several ligands of one structure at one site** each count: each is its
    own active with its own crystal pose (RMSD ground truth). Holo = at least
    one ligand at the site.
21. **Other screen modes** (virtual screening, redocking) are ported per site,
    like cross-docking: actives, box and receptors come from the site folder.

## The flow

```
qpocket run
  ingest
  choose frame (given reference, or auto-pick)
  per structure (parallel):   align → clean → find ligands → protonate
                              → ProLIF per ligand → cavity detection (whole protein)
  global:                     group ligands into sites → sites.csv, union PDBs
  per site × structure (parallel):
                              cut cloud from that structure's cavities
                              (+ site-level residue crop in Interactions mode)
                              → write cloud → geodesics
  collect:                    per-site master tables, holo/apo flags

qpocket explore  (per selected site)
  [Shape-only clustering]     spectral fingerprint on cached geodesics → shape distance → cluster
  [Featurise]                 feature providers (APBS, cp_lp, hbond, aromatic) on that site's clouds
  [Two-stage clustering]      existing two-stage on shape + chem   (disabled until Featurise done)
  [Cross-docking]             existing prepare / analyse, scoped to the site
  [Regroup sites]             re-run site grouping with a new cutoff, then re-cut clouds
```

Cavity detection runs once per structure over the whole protein and its cavity
points are kept; a site only selects from them. This makes regrouping sites
cheap: no structure is re-aligned, re-protonated or re-detected. *(My call.)*

## Data on disk

Layout version bumps to 3.

```
<run>/
  structures/{original,aligned,cleaned,extracted_ligands,protonated}/   as today
  structures/cavities/<pdb_id>.csv          all cavity points of a structure
  structures/ligands.csv                    one row per ligand found:
                                            pdb_id, chain, resname, resseq,
                                            n_heavy, centroid, file, site_id
  sites/
    sites.csv                               site_id, n_ligands, n_holo_structures,
                                            centre, extent (max diameter, Å),
                                            ligand resnames, contact residues
    grouping.json                           cutoff and parameters the grouping used
    site_01/
      ligands.pdb                           union of the site's ligands
      pockets/<pdb_id>/binding_site.csv     the cloud (no features)
      pockets/<pdb_id>/geodesic.*           cached geodesics
      pockets/<pdb_id>/featurized.csv       only after Featurise
      metadata/                             per-site master.csv (with holo/apo,
                                            ligand id), interactions h5,
                                            distance matrices, clustering,
                                            clustering_history/
      screening/                            per-site cross-docking
  metadata/                                 run-wide: run.json, failures.csv,
                                            logs
```

Each `sites/site_NN/` deliberately looks like today's run root (`pockets/`,
`metadata/`, `screening/`), so the existing distance, recluster, explorer and
cross-docking code can take a site folder where it takes a run folder today.
*(My call.)*

Site ids are ordered by number of holo structures, descending (`site_01` is
the most populated). *(My call.)*

## Config

- `reference.ligand_id`, `reference.ligand_chain_id`: removed; warn on load.
- `reference.pdb_id` / `fragment_screen.reference_path`: optional.
- New `sites` block, for example:
  ```yaml
  sites:
    min_heavy_atoms: 6           # smaller HET groups are not ligands
    exclude: []                  # extra resnames added to the built-in additive list
    include: []                  # resnames forced in (overrides the built-in list)
    link_cutoff: 4.0             # Å, heavy-atom distance joining two ligands into a site
    box_padding: 4.0             # Å around the union of a site's ligands
  ```
  Default values are the builder's to tune; these are placeholders.
- `reference.pocket_definition` (`Solvent` / `Interactions`) and
  `interaction_padding` keep their meaning, applied per site.
- `reference.pocket_mode` / `bounding_box`: a user-given box no longer makes
  sense run-wide; builder decides whether to drop it or treat a given box as
  one extra manual site.

## Notebook

A "Binding sites" section at the top of the explorer:

- Site table from `sites.csv` (sortable; n holo, n ligands, extent, resnames).
- 3D view: aligned frame protein plus the selected site's union ligands; all
  sites shown faintly for orientation.
- Site picker (single select). Every section below reads the selected site
  folder.
- Status per site: which steps are done (geodesics, shape clustering,
  featurised, two-stage, cross-docking), read from files on disk.
- Buttons as listed above; a button whose prerequisite is missing is disabled
  with a one-line reason.
- Long steps show progress and can be run instead via a printed CLI command or
  a generated SLURM script for that site.
- "Regroup sites" takes a new link cutoff, rewrites `sites/` and warns that
  per-site results (features, clustering, docking) under changed sites are
  discarded.

## CLI

Every notebook step has a CLI equivalent taking `--site site_NN` (my naming):
featurise, shape clustering, two-stage clustering (existing `recluster`),
cross-docking, regroup. `qpocket slurm generate` gains per-site stages for
featurise (the APBS-heavy one). Exact command names are the builder's.

## Constraints

- Reuse the plug-in registries (sources, pockets, charges, contacts, features,
  clustering) and the `pipe`/`parallel.run_map`/fingerprinted-resume pattern;
  no parallel framework is added.
- Alignment, cleaning and per-structure processing stay as they are, minus the
  reference-ligand dependence.
- Per-structure resume by fingerprint must still work; the site step and
  per-site steps get the same "same inputs → skip" behaviour.
- The explorer must still open a run where only `qpocket run` has finished.
- Geodesics are computed today from `featurized.csv`; they must read
  `binding_site.csv` so they can run before featurisation.

## Known code changes (from the final code review)

The "site folder mirrors a run root" call holds for `recluster`,
`write_clustering`, `pocket_manifest`, cross-docking `analyze`,
`two_stage.load_matrices` and `load_run_interactions` (they read only
`metadata/` and `pockets/`, and `master.csv` holds absolute paths). These
must change:

- **Run discovery:** `viz/runs.py:42-48` `is_run`/`is_complete` look for
  root `metadata/master.csv`, `pockets/` and matrices; `PRUNE`
  (`viz/runs.py:13`) adds `sites`.
- **Layout:** `io/layout.py:10` → version 3; a site workspace that keeps
  run-level `structures/*` paths (plain `workspace(site_dir)` points
  `structures/original` inside the site).
- **Cross-docking repair:** `viz/crossdock.py:31-37` `_sequences` uses
  `layout.raw_structure(ws, rep)` with a site path and swallows the error —
  pass the run workspace.
- **Distance:** drop `compute_spectral_distance_matrix` from `run.py:76` and
  SLURM `tasks.py:234`; `compute_geodesic` (`distance.py:35-110`,
  `structure.py:444-453`) reads `binding_site.csv` and keys its cache on it;
  a shape-only path giving eigenvalues without features or scaler
  (`distance.py:135-217`; shape distance is eigenvalues only,
  `core/spectral.py:173-175`).
- **Matrix readers** (`arms/recluster.py:56-60`, `viz/two_stage.py:41-42`,
  explorer `dist_df` `explorer.py:105`) must accept shape-only output alone.
- **Explorer:** clouds read only `featurized.csv` (`explorer.py:124,
  128-133, 579-590, 1115`) — fall back to `binding_site.csv`; `run_dir` →
  selected `site_dir` in cells at `explorer.py:88-150, 164, 182, 344-351,
  370, 1200, 1213`; `actives.smi` per site (`explorer.py:103`).
- **Config snapshot:** the run saves its resolved config so notebook/CLI
  steps get `features`, `charges`, `distance`, `compute`, and caches can
  fingerprint it.
- **Several ligands per structure:** `protonate` writes one ligand
  (`structure.py:337-341`); `read_contacts` selects by resname
  (`structure.py:383-388`), merging same-named ligands — select by
  chain+resseq; `extracted_ligands/{pdb_id}_{resname}.pdb`
  (`structure.py:206`) collides — include chain and resseq;
  `master_row` (`core/types.py:293`), `actives.smi`
  (`collect.py:139-143`) and `distinct_actives` (`screen/ligands.py:30,39`)
  assume one ligand per `pdb_id`.
- **Per-site interactions:** `observed_interactions.h5` per site holds only
  that site's contacts (holo = has rows, `viz/interactions.py:137`; the
  cross-docking interacting-residues box reads it, `crossdock.py:66-74`).
- **Screen modes:** `campaign.py:121`, `analysis.py:392` hard-code
  `base/structures/extracted_ligands` — take the site.
- **SLURM:** per-site featurise and distance stages take site paths
  (`tasks.py:234`).

## Left to the builder

- Auto-pick rule for the frame when no reference is given (suggestion: best
  resolution, then input order).
- The built-in additive/ion exclusion list.
- Default numbers in `sites`.
- What to do with `pocket_mode: bounding_box`.
- Whether the APBS grid is computed once per structure and shared by all sites
  (it should be, if the grid covers the protein).
- How the notebook runs local work without blocking the UI (marimo threads vs
  subprocess).
- Exact CLI names and SLURM stage split.
- How existing tests and the golden run are updated for the new layout.

## Calls I made (not discussed)

- Cavity detection once per structure, kept on disk, so sites only select.
- Each site folder mirrors a run root, so existing per-run code is reused per site.
- Site ids ordered by holo count.
- Layout version 3; old runs re-run.
- Regrouping discards per-site downstream results for sites that changed.
- Matching sites before/after a regroup, multi-residue ligands and glycans,
  and cross-docking scope (from the site's `master.csv`) are the builder's.
