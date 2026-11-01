# Screening: cross-docking

Cross-docking asks how well a site's cluster representatives reproduce the crystal
binding modes of the ligands seen at that site: every distinct ligand is docked into
every representative, and each pose is scored by its RMSD to the crystal ligand.

Screening works per site and only from that site's **final** metadata
(`sites/<site>/final/`, written by **Finalise** on the Exploration page). A site
that was never finalised, or whose clustering or representatives changed since, is
refused until it is finalised again.

The explorer's **Screening** tab runs everything below; the commands do the same
from a terminal.

## 1. Box

Two ways to place the search box:

| Mode | Built from | Use when |
|---|---|---|
| Interacting residues | every residue a ligand contacts at least once (from the final residue profile), in the frame structure | contacts are known; the box hugs where ligands actually bind |
| Representatives' clouds | the pocket point clouds of every final representative | contacts are missing, or you want the whole pocket |

The page draws both boxes: the tight one around the points, and the padded one
docking uses (`crossdock.box_padding`, 5 Å by default). **Save box** writes
`sites/<site>/screening/box.json`. Preparation uses a saved box only while it is
still current: it goes stale when the site is finalised with a different clustering
and, for an interacting box, when the contacted residues change. A stale, missing
or unreadable box falls back to the configured box, with a warning.

## 2. Prepare

```bash
qpocket screen prepare --workdir runs/7KG6 --site S1-H21
```

Preparation reads the final representatives, prepares each receptor (pdb2pqr,
PROPKA pH 7.4; with the `screen` extra, PDBFixer first rebuilds missing side chains
and loops within `receptor.loop_radius` of the pocket) and every distinct ligand,
and writes the task folder `sites/S1-H21/screening/crossdock/`:
`docking_jobs.csv` (one row per receptor × ligand), `box.json`, `receptors/`,
`ligands/`, and `representatives_used/`. Receptors that fail are left out and named
in a warning; if none can be prepared, preparation stops and says why.

Preparing again with a different job list moves the old poses to
`poses_previous/`. Preparation refuses while a run started from that folder is
still going.

Engines: QuickVina (default, ships with the package), Vina and GNINA, plus any
binary declared under `docking.custom_engines`. The page greys out engines that are
not installed and shows the number of dockings before you prepare.

## 3. Run

Three ways; nothing runs until you choose one.

**Here.** On the page, **Start** runs every job in the background on this machine;
**Refresh progress** counts finished poses and **Stop** ends the run (and the
dockings it started). From a terminal:

```bash
cd runs/7KG6/sites/S1-H21/screening/crossdock
qpocket screen run-local docking_jobs.csv --workers 8 --cpus 4
```

`--workers` defaults to half the cores; `--cpus` sets threads per docking.

**SLURM.** Preparation also writes `run_docking.sbatch` into the task folder; copy
the folder to the cluster and submit it from inside:

```bash
cd runs/7KG6/sites/S1-H21/screening/crossdock && sbatch run_docking.sbatch
```

Each array task runs one job through `qpocket screen run-task`; long job lists are
split under the array-size cap.

**A VM over SSH.** Add a `remote:` block to the screening config:

```yaml
remote:
  host: my-vm                 # an ssh alias or user@host; key-based login
  directory: ~/qpocket_crossdock
  setup: ["source ~/QPocket/.venv/bin/activate"]
  azure_vm: null              # optional: only to show the VM's power state
  azure_resource_group: null
```

**Push & start** copies the task folder to `<directory>/<site>` and starts it in a
`tmux` session; **Poll** reports pose files, whether the session runs, and any other
`tmux` sessions on the machine; **Pull poses** copies the results back. QPocket
never starts or stops a VM.

## 4. Analyse

```bash
qpocket screen analyze --workdir runs/7KG6 --site S1-H21
```

Every pose is scored by **in-place RMSD** to the crystal ligand: over the atoms of
their maximum common substructure, with no superposition, symmetric matches taken
into account, and the best over every crystal copy of that ligand. A docking
succeeds at a level when its RMSD is ≤ `analysis.rmsd_threshold` (2 Å):

| Measure | Pose it looks at | Tells you |
|---|---|---|
| top-1 | the best-scored pose | what you would get in practice |
| top-k | the best of the `analysis.top_k` (3) best-scored poses | a looser practical measure |
| best sampled | the best of every pose the engine returned | whether a right pose was generated at all |

The gap between top-1 and best-sampled success is what scoring missed; a low
best-sampled rate means the receptor cannot place those ligands correctly.

Outputs in the task folder: `crossdock_poses.csv` (every pose), `crossdock_pairs.csv`
(one row per docking with `top1_rmsd`, `best_topk_rmsd`, `best_rmsd` and their
successes) and `crossdock_success.csv` (rates per representative, per ligand and
overall).

The page shows, per representative, two bars — best-scored pose and any sampled pose
within the threshold — and a representative × ligand matrix of the best RMSD. The
**Poses** viewer draws one docking: receptor, crystal poses, and the top poses
coloured green (≤ threshold), amber (≤ twice) or red.

## Configuration

The page, `prepare` and `analyze` read `sites/<site>/screening/screen_config.yaml`,
else `<run>/screen_config.yaml`, else the defaults. Main keys:

```yaml
run_mode: cross_docking
docking: {engine: quickvina, exhaustiveness: 8, num_modes: 10, cpus: 4, seed: 42}
receptor: {repair: true, loop_radius: 10.0, use: pdb2pqr}
crossdock: {box_mode: representative_pockets, box_padding: 5.0}
analysis: {rmsd_threshold: 2.0, top_k: 3}
remote: {host: null, directory: ~/qpocket_crossdock, setup: []}
```

`run_mode: virtual_screening` (actives against generated decoys, with enrichment
metrics) exists on the command line; the explorer shows the active/decoy benchmark
as not implemented yet.
