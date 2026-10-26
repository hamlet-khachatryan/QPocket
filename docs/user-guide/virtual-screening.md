# Virtual screening

Docking a finished run's representative pockets of one binding site, either locally
or on a cluster. Screening is per site: every command takes `--site site_NN`, and the
site must be clustered first (`qpocket site cluster`, or the explorer). Virtual
screening and redocking use only that site's ligands.
Preparing inputs and running them are separate steps: `prepare` writes the task
list, and nothing runs until you say so.

## Prepare

```bash
qpocket screen prepare --workdir runs/7KG6 --site site_01
```

This builds the box, prepares receptors (from each pocket's `protein.pqr`, so the
run's protonation states and charges carry into docking) and ligands, and writes
`sites/site_01/screening/docking_jobs.csv`, one row per receptor × ligand. It prints
the command to run them and stops.

## Run

Locally, on this machine:

```bash
qpocket screen run-local runs/7KG6/sites/site_01/screening/docking_jobs.csv --workers 8
```

`--workers` defaults to half the cores; `--cpus` sets threads per docking job.
This works on Linux, macOS and Windows.

On a cluster, ask for submission scripts:

```bash
qpocket screen prepare --workdir runs/7KG6 --site site_01 --submit slurm
sbatch runs/7KG6/sites/site_01/screening/run_docking.sbatch
```

Each array task runs one job through `qpocket screen run-task`, so nothing parses
CSV in bash, and long job lists are split into chunks under the array-size cap.
`--submit local` writes `run_docking.sh` instead, wrapping the same local command.

## Analyse

```bash
qpocket screen analyze --workdir runs/7KG6 --site site_01
```

Writes scores, pose RMSD, enrichment metrics (virtual-screening mode) and
interaction statistics to `sites/site_01/screening/analysis/`.

## Pose RMSD

Pose RMSD is measured in place: the docked pose is compared with the crystal
ligand over the atoms of their maximum common substructure, with no
superposition and no best fit, so a pose that sits in the wrong place scores a
large RMSD. Symmetric ligands are scored against their best equivalent atom
matching. Values for misplaced poses are therefore larger than in earlier
versions, which fitted the pose onto the ligand first.

## Cross-docking

Cross-docking snapshots the representatives it used, under
`screening/crossdock/representatives_used/`, and records the clustering mode in
`box.json`. Re-clustering the site afterwards therefore does not change what an
earlier cross-docking run means.

## Known gaps

Screening is not finished. `qpocket screen prepare` is usable, but the box glob
never matches cavity files, the RMSD reference lookup is a placeholder, and PLIP is
run on the pose without its receptor. These are marked in the code with
`# Known issue (spec §6)`.
