# Virtual screening

Docking a finished run's representative pockets, either locally or on a cluster.
Preparing inputs and running them are separate steps: `prepare` writes the task
list, and nothing runs until you say so.

## Prepare

```bash
qpocket screen prepare --workdir runs/7KG6
```

This builds the box, prepares receptors (from each pocket's `protein.pqr`, so the
run's protonation states and charges carry into docking) and ligands, and writes
`screening/docking_jobs.csv` — one row per receptor × ligand. It prints the command
to run them and stops.

## Run

Locally, on this machine:

```bash
qpocket screen run-local runs/7KG6/screening/docking_jobs.csv --workers 8
```

`--workers` defaults to half the cores; `--cpus` sets threads per docking job.
This works on Linux, macOS and Windows.

On a cluster, ask for submission scripts:

```bash
qpocket screen prepare --workdir runs/7KG6 --submit slurm
sbatch runs/7KG6/screening/run_docking.sbatch
```

Each array task runs one job through `qpocket screen run-task`, so nothing parses
CSV in bash, and long job lists are split into chunks under the array-size cap.
`--submit local` writes `run_docking.sh` instead, wrapping the same local command.

## Analyse

```bash
qpocket screen analyze --workdir runs/7KG6
```

Writes scores, pose RMSD, enrichment metrics (virtual-screening mode) and
interaction statistics to `screening/analysis/`.

## Known gaps

Screening is not finished. `qpocket screen prepare` is usable, but the box glob
never matches cavity files, the RMSD reference lookup is a placeholder, and PLIP is
run on the pose without its receptor. These are marked in the code with
`# Known issue (spec §6)`.
