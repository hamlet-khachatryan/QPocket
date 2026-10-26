# Runbook: Q-Pocket on a SLURM cluster

Starts from the point where the repository is already on the cluster's
filesystem, and ends with results in your browser. Every command is meant to be
copied; set the `<angle-bracket>` values once, in step 0.

```
login node ──sbatch──▶ compute nodes
    │
    └──ssh tunnel──▶ laptop browser (marimo notebook on the results)
```

---

## 0. Set these once

On the cluster, e.g. in `~/.bashrc`:

```bash
export QP_HOME=/home/<user>/QPocket             # the repository; compute nodes must see it
export QP_RUNS=/scratch/<user>/qpocket_runs     # results; a shared filesystem, never /tmp
export QP_PART=<partition>                      # e.g. compute
export QP_ACCT=<account>                        # leave empty if your site has none
```

`QP_RUNS` must be on a filesystem that **every compute node mounts** (usually
`/scratch`, `/work` or `/project`). Node-local `/tmp` looks fine and then loses
every result. Scratch is often purged after a set number of days, so copy
results out when a run finishes (step 8).

---

## 1. Check the checkout is complete

These files are required. At the time of writing they had never been committed,
so a transfer made from git would lack them unless they were committed first:

```bash
cd "$QP_HOME"
for f in src/qpocket/conf/config.yaml \
         src/qpocket/core/clustering.py \
         src/qpocket/pipelines/slurm.py src/qpocket/pipelines/tasks.py \
         src/qpocket/doctor.py src/qpocket/viz/runs.py \
         src/qpocket/core/structure.py \
         scripts/qpocket_doctor.py src/qpocket/viz/explorer.py; do
  [ -f "$f" ] && echo "ok       $f" || echo "MISSING  $f"
done
```

Any `MISSING`: commit it on the laptop (`git add` + `git commit` + `git push`)
and transfer again. The pipeline fails on every run without these.

---

## 2. Check a compute node can run it — before installing anything

```bash
cd "$QP_HOME"
srun -p "$QP_PART" ${QP_ACCT:+-A $QP_ACCT} -t 00:10:00 --pty \
  python3 scripts/qpocket_doctor.py
```

Run it through `srun`: login nodes often run a different OS image from compute
nodes, and the compute node is the one that matters. It prints either the
install command, or what is missing and why.

What decides it (APBS ships only as a platform wheel):

| compute-node OS | works with |
|---|---|
| Ubuntu 22.04+, Debian 12, Rocky / RHEL / Alma 8 and 9 | Python 3.9 – 3.12 |
| CentOS / RHEL 7, or any ARM | no pip install for APBS — needs a site module or a container |

If the default `python3` is the wrong version, look for a module first:
`module avail python`, then `module load python/3.11`, and re-run the check.

---

## 3. Install

Q-Pocket is built and installed with [uv](https://docs.astral.sh/uv/). Get it
once (no root needed; either line works):

```bash
python3 -m pip install --user uv                     # wherever pip reaches PyPI
curl -LsSf https://astral.sh/uv/install.sh | sh      # or the standalone installer
```

Both put `uv` in `~/.local/bin`. Then:

```bash
cd "$QP_HOME"
module load python/3.11                               # only if you needed one in step 2
uv sync --locked --extra viz --python "$(command -v python3)"
. .venv/bin/activate
```

`uv sync` creates `.venv` and installs exactly the versions in `uv.lock`, so the
cluster gets the same numbers as the laptop. `--python` makes it use the site's
interpreter instead of downloading one; drop it if downloads are fine. `--extra
viz` adds marimo for the notebook.

Then prove it works **on a compute node**. This is not optional: it runs
pdb2pqr and APBS for real, and a wheel can install cleanly and still fail to
load its libraries on a given OS.

```bash
srun -p "$QP_PART" ${QP_ACCT:+-A $QP_ACCT} -t 00:10:00 --pty \
  "$QP_HOME/.venv/bin/qpocket" doctor
```

It must end with `This node can run the pipeline.`

If it passes on the login node but **fails on a compute node**, the two run
different OS images and pip picked wheels for the login node's. Rebuild the venv
from a shell on a compute node (`srun ... --pty bash`, then `uv sync` as above),
provided compute nodes can reach PyPI; otherwise this cluster needs a container.

---

## 4. Write the run config

```bash
mkdir -p "$QP_RUNS"
cd "$QP_RUNS"
"$QP_HOME/.venv/bin/qpocket" init -o my_target.yaml     # every setting, with defaults
```

Edit `my_target.yaml`. The lines that matter for a cluster run:

```yaml
mode: "pdb_query"                  # or "fragment_screen"

reference:
  pdb_id: "7KG6"
  pocket_definition: "Interactions"

paths:
  workdir: "/scratch/<user>/qpocket_runs"      # absolute; results land in <workdir>/<pdb_id>/

slurm:
  partition: "<partition>"
  account: "<account>"                         # or null
  environment:                                 # copied into every job script
    - "module load python/3.11"                # only if you used one in step 3
    - "source /home/<user>/QPocket/.venv/bin/activate"
  stages:
    site_featurize: {cpus: 1, mem: "8G", time: "04:00:00"}   # APBS runs here
```

Use **absolute paths** in `environment` and `workdir`: every job starts in a
scratch directory under the run, not where you submitted from.

**Fragment screen instead?** Put your structure files somewhere on the cluster
filesystem and point the config at them; see
`docs/user-guide/fragment-screening.md`.

```bash
"$QP_HOME/.venv/bin/qpocket" validate my_target.yaml
```

---


## Running on one node

A whole run on a single machine, without SLURM:

```bash
"$QP_HOME/.venv/bin/qpocket" run --config my_target.yaml            # half the cores
"$QP_HOME/.venv/bin/qpocket" run --config my_target.yaml --workers 16
"$QP_HOME/.venv/bin/qpocket" run --config my_target.yaml --features hbond,aromatic
```

Each structure is aligned, cleaned, has its ligands found, is protonated and
has its contacts and cavities read on its own worker, then the sites are grouped and cut; see
[Performance](docs/user-guide/performance.md). Memory scales with `--workers`,
because each one can run APBS. Split the run across nodes with `qpocket slurm`
instead, below.

## 5. Prepare, generate, submit

```bash
cd "$QP_RUNS"
. "$QP_HOME/.venv/bin/activate"

qpocket task prepare --config my_target.yaml     # fetch, build the reference — on the login node
qpocket slurm generate --config my_target.yaml   # writes the job scripts, submits nothing
```

`prepare` runs on the login node on purpose. In `pdb_query` mode it downloads
from RCSB and queries UniProt, and compute nodes frequently have no internet.
It takes seconds to minutes and uses one core. (If your login node forbids even
that, `sbatch <run>/slurm/00_prepare.sbatch` instead — it only works if compute
nodes can reach the internet.) Fragment-screen mode never touches the network.

Look at what was generated before spending allocation on it:

```bash
ls   "$QP_RUNS/7KG6/slurm/"
less "$QP_RUNS/7KG6/slurm/10_structure.sbatch"
```

Then submit the whole chain. Each stage waits for the previous one:

```bash
bash "$QP_RUNS/7KG6/slurm/submit_all.sh"
```

---

## 6. Watch it

```bash
squeue -u "$USER"                                  # what SLURM is doing
qpocket slurm status --config my_target.yaml       # what is actually finished
tail -f "$QP_RUNS/7KG6/slurm/logs/structure_"*.err # one task's output
```

`slurm status` reads the files on disk, so it still answers after the jobs have
left `squeue`:

```
stage                   done   missing
prepare                  1/1   -
structure              19/21   4B1J, 6O9Y
sites                    0/1   sites/sites.csv
```

When `sites` is done, featurise and cluster a site the same way:
`qpocket slurm generate --config my_target.yaml --site S1-H21`, then
`bash "$QP_RUNS/7KG6/slurm/submit_S1-H21.sh"`.

---

## 7. When something fails

Resubmit the same chain. Finished tasks are skipped, so only failures rerun:

```bash
bash "$QP_RUNS/7KG6/slurm/submit_all.sh"
```

Rerun a single structure by hand, to see its error directly:

```bash
qpocket task structure --config my_target.yaml \
  --manifest "$QP_RUNS/7KG6/slurm/manifests/structure.txt" --index 3 --force
```

| symptom | cause | fix |
|---|---|---|
| `qpocket: command not found` in a job log | `environment` does not activate the venv | absolute path to `.venv/bin/activate` in `slurm.environment`, then regenerate |
| `apbs runs FAILED` from `qpocket doctor` | the wheel's binary cannot load its libraries on this OS | check step 2's table; this node needs a container or site modules |
| site_featurize tasks hit `TIME LIMIT` | a large protein makes APBS slow | raise `slurm.stages.site_featurize.time`, regenerate, resubmit |
| site_featurize tasks `OUT_OF_MEMORY` | APBS grid grows with protein size | raise `slurm.stages.site_featurize.mem` |
| `prepare` hangs or fails on network | compute nodes are offline | run `qpocket task prepare` on the login node (step 5) |
| `sites` finds no sidecars | every structure task failed | `slurm status`, then read one `structure_*.err` |
| `index N is outside manifest` | scripts older than the manifest | `qpocket slurm generate` again |

After changing anything under `slurm:` in the config, **regenerate** — the job
scripts are written once and do not re-read the config.

---

## 8. Look at the results

### In the browser, straight from the cluster

On the **login node**, serve the notebook on localhost only:

```bash
cd "$QP_HOME"
. .venv/bin/activate
qpocket explore "$QP_RUNS" --headless --port 2718
```

On the **laptop**, forward the port and open it:

```bash
ssh -N -L 2718:127.0.0.1:2718 <user>@<login-node>
# then browse to http://localhost:2718
```

`--runs` makes every run under that folder appear in the dropdown. You can also
type any other path into the **"…or open a path"** box in the notebook — a single
run directory, or a folder of runs.

Ports are shared on a login node: if 2718 is taken, pick another number and use
it on both sides. Stop the server with Ctrl-C when you are done.

If your site forbids long-running processes on login nodes, run it inside a job
and tunnel through the login node instead:

```bash
# cluster
srun -p "$QP_PART" -t 02:00:00 --pty bash
hostname                                          # note the node name
qpocket explore "$QP_RUNS" --headless --port 2718

# laptop
ssh -N -L 2718:<node-name>:2718 <user>@<login-node>
```

`--host 0.0.0.0` makes it reachable from the cluster's internal network, which
is why the token password is not optional there.

### Or copy the results home

The notebook needs `metadata/`, `pockets/` and `structures/` from each run, not
the SLURM logs. Copy those folders to the laptop with your usual transfer method,
then:

```bash
qpocket explore -- --runs <folder with the copied runs>
```

Do this before scratch is purged.

---

## 9. Updating the code later

Push on the laptop, transfer into the same `$QP_HOME`, then on the cluster:

```bash
cd "$QP_HOME"
uv sync --locked --extra viz --python "$(command -v python3)"   # picks up dependency changes
srun -p "$QP_PART" -t 00:10:00 --pty "$QP_HOME/.venv/bin/qpocket" doctor
```

The project is installed in editable mode, so code changes take effect without
reinstalling.
That also means a task picks up whatever code is on disk when it **starts** —
do not update while a run is in progress.
