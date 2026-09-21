# Running on a cluster

The pipeline's cost is almost entirely APBS: on a 21-structure run it is 216 s of
366 s, spent one structure at a time. Every structure is independent, so on SLURM
each one becomes its own array task.

```
00_prepare     1 job       fetch, align, clean, locate the reference pocket
20_featurize   array N     cavity -> APBS -> features -> PLIP, one per structure
30_collect     1 job       merge per-structure results into the run-wide tables
40_geodesic    array N     geodesic graph, one per pocket
50_distance    1 job       fingerprints, distance matrices, clustering
```

## 1. Install on the cluster

Follow [Installation](../installation.md), and run both checks **on a compute
node** — login nodes often run a different OS image:

```bash
srun -p <partition> --pty python3 scripts/qpocket_doctor.py     # before install
srun -p <partition> --pty .venv/bin/qpocket doctor              # after install
```

## 2. Tell the jobs how to find the environment

Batch jobs start with a bare shell. Put the activation lines in the config; they
are copied verbatim into every generated script:

```yaml
slurm:
  partition: "compute"
  account: "my_project"          # if your site requires one
  environment:
    - "module load python/3.11"  # only if the venv was built on a module python
    - "source /home/me/QPocket/.venv/bin/activate"
```

Use an **absolute** path to the venv: the scripts run with `--chdir` set to a
scratch directory inside the run, not the directory you submitted from.

## 3. Prepare, then generate

The array stages are sized from `master.csv`, which `prepare` writes, so run it
first:

```bash
qpocket task prepare --config run.yaml
qpocket slurm generate --config run.yaml
```

!!! warning "Compute nodes without internet"
    In `pdb_query` mode, `prepare` downloads structures from RCSB and looks up
    UniProt accessions. If your compute nodes cannot reach the internet, run
    `qpocket task prepare` on the login node as above rather than submitting it.
    Every later stage is offline. Fragment-screen mode never touches the network.

`generate` writes the scripts and a `submit_all.sh` under `<run>/slurm/`. Read
them, then:

```bash
bash <workdir>/<run>/slurm/submit_all.sh
# or generate and submit in one step:
qpocket slurm generate --config run.yaml --submit
```

## 4. Watch it

```bash
qpocket slurm status --config run.yaml
```

```
stage               done   missing
prepare              1/1   -
featurize          19/21   4B1J, 6O9Y
collect              0/1   metadata/observed_interactions.h5
```

Progress is read from the files on disk, not from `sacct`, so it keeps working
after the scheduler has forgotten the jobs. Per-task logs are in
`<run>/slurm/logs/`.

## 5. Re-running

Every task skips work whose output already exists. Resubmitting the same chain
re-runs only what failed, and finishes in seconds when nothing did. Add `--force`
to a `qpocket task` command to recompute regardless.

By default a stage waits for the previous array with `afterany`: one structure
that fails to featurise does not block the rest of the run, and the later stages
skip it. Set `slurm.dependency: afterok` for all-or-nothing.

## Resources

Per-stage requests live in `slurm.stages`; see the
[configuration reference](../configuration.md#slurm). The defaults give
featurisation 8 GB and 4 hours per structure — APBS memory grows with the grid,
so raise `mem` for very large proteins.
