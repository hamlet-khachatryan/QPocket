# Running on a cluster

The pipeline's cost is almost entirely APBS: on a 21-structure run it is 216 s of
366 s, spent one structure at a time. Every structure is independent, so on SLURM
each one becomes its own array task.

```
00_prepare               1 job     fetch the structures, build the reference
10_structure             array N   align, clean, protonate, ligands, contacts, cavities
20_sites                 1 job     collect, discover the sites, cut every site's clouds

site_01_30_site_featurize  array M   features -> APBS, one per cloud of the site
site_01_40_site_cluster    1 job     fingerprints, distance matrices, clustering
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

The structure stage is sized from the structures `prepare` ingests, so run it
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
stage                   done   missing
prepare                  1/1   -
structure              19/21   4B1J, 6O9Y
sites                    0/1   sites/sites.csv
```

Progress is read from the files on disk, not from `sacct`, so it keeps working
after the scheduler has forgotten the jobs. Per-task logs are in
`<run>/slurm/logs/`.

## 5. Per-site analysis

Once `sites` has finished, featurise and cluster a site on the cluster — one task
per cloud, with APBS inside:

```bash
qpocket slurm generate --config run.yaml --site site_01
bash <workdir>/<run>/slurm/submit_site_01.sh
qpocket slurm status --config run.yaml --site site_01
```

A site's scripts, manifests and logs carry its id, so several sites can be
generated and run side by side.

`--config` locates the run and sets the SLURM resources; the site tasks
themselves read the settings saved with the run in `<run>/metadata/config.yaml`.
Edit that file to change features, charges, distance or clustering for later
site steps.

## 6. Re-running

Every task skips work whose output already exists. Resubmitting the same chain
re-runs only what failed, and finishes in seconds when nothing did. Add `--force`
to a `qpocket task` command to recompute regardless.

By default a stage waits for the previous array with `afterany`: one structure
that fails to featurise does not block the rest of the run, and the later stages
skip it. Set `slurm.dependency: afterok` for all-or-nothing.

## Resources

Per-stage requests live in `slurm.stages`; see the
[configuration reference](../configuration.md#slurm). The defaults give
`site_featurize` 8 GB and 4 hours per cloud — APBS memory grows with the grid,
so raise `mem` for very large proteins.
