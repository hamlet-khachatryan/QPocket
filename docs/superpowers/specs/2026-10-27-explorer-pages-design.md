# Explorer as three pages: Site, Exploration, Screening

Date: 2026-10-27
Status: approved 2026-10-27

## Why

The explorer is one long notebook (`src/qpocket/viz/explorer.py`, ~1,400 lines:
Binding sites → Clustering → Structures → Landscape → Pockets → Ligands → Cluster
distances → Interaction fingerprint → Screening). The user wants it organised as
the three stages of the work, each with its own controls, and the screening
stage made usable. Cross-docking preparation currently fails on the EV-A71 2A
laptop copy.

### The cross-docking failure (found 2026-10-27)

`qpocket screen prepare --workdir ~/qpocket_runs/EV71_2A_cavity --site S1-G127`
ends with `ValueError: no coordinates for box mode 'representative_pockets'
among []`. Root cause: `prepare_receptor` fails for every one of the 36
representatives with `No module named 'openmm'` — the laptop environment was
synced without the project's `screen` extra (`pdbfixer`, `openmm`). The error is
logged and swallowed (`arms/screen/receptors.py:183`), so the user only sees the
misleading box error. APBS works on the Mac (`apbs-binary` macOS arm64 wheel;
`qpocket doctor` runs it).

## Decisions (the user's, from 20 questions)

Navigation and flow
1. **Tabs in one notebook**: Site · Exploration · Screening, shared state.
2. **Free navigation with guards**: any tab opens; a tab missing a prerequisite
   says what to do first.
3. **Compute per step, local or SLURM**: every heavy step (re-siting,
   featurisation, docking) offers "run here" (workers) or "submit to SLURM". The
   Mac has everything needed, APBS included.

Site page
4. **Site redefinition**: regroup parameters (cavity overlap, ligand fraction,
   box padding, min cavity points, min structure share); **manual merge** of
   sites; **manual split / drop** of a site; a **3D site overview**.

Exploration page
5. **A "Finalise site" button**: Apply still previews and saves clustering;
   Finalise freezes clustering and representatives and writes the final site
   metadata that Screening reads.
6. **Representatives automatic, overridable**: medoid per cluster by default;
   any cluster's representative can be swapped in the cluster table.
7. **Final metadata**: per-pocket cluster + representative table; clustering
   settings used; per-cluster residue profile (contacted and lining
   frequencies); per-cluster ligand list (the actives).
8. **Panels**: existing interaction fingerprint and 2D ligand panels, restyled,
   plus a clusters × residues contact heatmap. Featurisation runs from this page.

Screening page
9. **Two box modes**: (a) bounding box of the site's **interacting residues**
   plus padding, with a viewer of up to 40 superposed ligands and the interacting
   residues highlighted; (b) bounding box of the **representatives' point
   clouds** plus padding, with a viewer showing the clouds coloured by cluster,
   the **extremum points** that set each face, the tight box and the padded box,
   and the reference protein cartoon.
10. **The 40 ligands**: searchable picker; default = representatives' ligands,
    then one per cluster, then most frequent.
11. **Box editing**: mode + padding slider, live preview, **Save box** writes the
    site's `screening/box.json`, which preparation uses.
12. **Engines**: QuickVina 2, AutoDock Vina, GNINA when installed, custom engines
    from config.
13. **Submission**: run here in the background (progress), SLURM array, remote
    VM over SSH.
14. **Pairs**: all distinct compounds × all representatives (redocking
    included); RMSD of each pose against every crystal pose of that compound.
15. **Active/decoy benchmark**: shown as a disabled option "not implemented yet".
16. **Analysis**: success matrix (receptor × ligand, best-pose RMSD) and top-k
    success per receptor; success threshold 2.0 Å, configurable.
17. **Pose viewer**: pick receptor and ligand; receptor, crystal pose and top-N
    docked poses coloured by RMSD, interacting residues highlighted.

## Answers to the review (the user's)

18. **"Submit to SLURM" only writes scripts**: the notebook writes the sbatch
    scripts and shows the commands; moving the run and submitting is the
    user's. "Run on VM" stays the one remote-execution path (SSH).
19. **Overrides follow member cavities**: merge/split/drop are stored as sets
    of `(pdb_id, cavity)` and re-applied to whichever sites hold those cavities
    after a regroup. The strict cavity groups a split needs are saved in
    `grouping.json`.
20. **Changed sites: confirm, then delete**: before a merge, split, drop or
    regroup that changes a site's members, the page lists what will be lost
    (clustering, final metadata, docking results) and asks; on confirm the old
    folder is deleted as today.
21. **CLI prepare prefers final + saved box**: `qpocket screen prepare` uses
    `final/` and a saved `screening/box.json` when present (a box older than the
    last Finalise is stale and ignored with a warning); otherwise it falls back
    to today's behaviour with a warning.
22. **Interacting-residues box = every residue contacted at least once in the
    site** (any holo structure), in the reference frame; the 40-ligand viewer
    highlights exactly those residues.
23. **Shape-only sites can be finalised and screened**: `settings.json` records
    `shape_only` and an empty feature set.

## Phases (my call: order and boundaries)

Four deliverables, built and merged in order; each gets its own plan.

### Phase 0 — cross-docking preparation fails clearly, and works

- Receptor failures are collected; when **no** receptor could be prepared,
  `prepare` raises one error naming the reasons (e.g. "No module named 'openmm'
  — install the screen extra: `uv sync --extra screen`") instead of the box
  error. When some fail, the rest proceed and a warning lists the failed ones.
- `qpocket doctor` checks the `screen` extra (pdbfixer, openmm) as an optional
  group, like docking engines.
- `docs/installation.md` says to sync with `--extra screen` for cross-docking.
- Then, on the laptop: `uv sync --extra viz --extra screen`, rerun prepare for
  S1-G127, confirm receptors, ligands and box.

### Phase 1 — tabbed shell and the Site page

- `explorer.py` stays one notebook; its top cell is `mo.ui.tabs` with three
  pages. Page logic moves into `viz/pages/{site,exploration,screening}.py`
  helpers (*my call*: one notebook for shared state, as decided; logic moved out
  so the file stays navigable).
- Site page: run picker, site table, 3D site overview (all site boxes and a
  capped set of ligands on the reference protein), regroup controls (all five
  parameters), and manual operations:
  - **merge** ≥ 2 sites → one site whose members are the union of their member
    cavities;
  - **drop** a site (its ligands become misses with reason `site_dropped`);
  - **split** a site by its strict cavity groups (the complete-linkage cores
    before the site merge).
  Overrides live in the run config as `sites.overrides` (*my call*, so they are
  reproducible and survive regroup), are applied after automatic grouping,
  recorded in `grouping.json`, and listed on the page with an undo each.
- Guards: Exploration and Screening say "pick a site on the Site page".

### Phase 2 — the Exploration page and final metadata

- Holds today's clustering, structures, landscape, pockets, ligands, cluster
  distances and interaction sections (restyled), featurisation controls (local
  workers or SLURM), and the clusters × residues heatmap.
- Representative override: the cluster table gains a representative picker per
  cluster; overrides are saved with the clustering.
- **Finalise site** writes `sites/<id>/final/` (*my call*: one folder Screening
  reads): `clusters.csv` (pdb_id, cluster, is_representative,
  dist_to_representative, holo), `settings.json` (method, percentiles, linkage,
  feature set, matrix hashes, ISO date), `residue_profile.csv` (cluster, residue,
  contacted %, lining %), `ligands.csv` (cluster, pdb_id, ligand key, resname,
  SMILES). Finalising again keeps the previous one in `final/history/<ISO time>/`.
- Guard: Screening says "finalise this site on the Exploration page" when
  `final/` is missing or older than the applied clustering.

### Phase 3 — the Screening page

- Run type: Cross-docking; Active/decoy benchmark (disabled, "not implemented
  yet").
- Box: the two modes, their viewers, padding, Save box → `box.json`. Interacting
  residues come from the final residue profile; clouds from the final
  representatives.
- Docking: engine (installed ones selectable, GNINA greyed when absent),
  exhaustiveness, modes, CPUs; pair counts shown before submission.
- Submission: Run here (background, progress from finished job files); Submit to
  SLURM (writes the existing runner scripts and shows the commands, decision 18); Run on VM (*my call*: SSH host and directory
  from a `remote:` block in `screen_config.yaml`; rsync the task folder, start
  the runner in `tmux`, poll, pull results back; never starts or stops the VM —
  the page shows the VM power state via `az` when available and warns when other
  jobs are running there).
- Analysis: success matrix and top-k success per receptor, threshold field.
- Pose viewer: receptor/ligand pickers, cartoon, crystal pose, top-N poses by
  RMSD colour, interacting residues highlighted.

## Goals

- Each stage is one tab with only its controls; EV-A71 2A goes from site
  selection to a cross-docking success matrix inside the notebook.
- Cross-docking preparation never fails with a misleading message.

## Non-goals

- The active/decoy benchmark itself; starting/stopping VMs; changing engines.

## Anti-goals

- A tab silently showing results from before the last Finalise or regroup.
- Heavy computation started by merely opening a tab.

## Acceptance (per phase)

- Phase 0: all-receptors-fail test yields the explicit error; doctor test for the
  screen extra; S1-G127 prepares on the laptop.
- Phase 1: render tests per tab; tests for merge / drop / split and undo;
  regroup keeps overrides.
- Phase 2: representative override, Finalise outputs and history, the guard;
  render test of the Exploration tab with chemistry.
- Phase 3: both box modes (extremum points, padding), Save box, pair counting,
  local progress, SLURM scripts, remote command construction (no real SSH in
  tests), analysis tables, pose viewer data; render test of the Screening tab.

## Left to the builder

Module layout inside `viz/pages/`; widget layout within the approved style; how
background runs report progress; SSH/rsync details.
