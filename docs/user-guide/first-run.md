# Your first runs

Three runs, in the order worth doing them: a fragment screen over files you
already have, a PDB query that fetches its own structures, and the notebook that
shows you what came out. Every command here has been run as written.

For running on a cluster, see `RUNBOOK.md` in the repository root instead — this
page is about one machine.

## Before you start

```bash
uv sync --extra viz
uv run qpocket doctor
```

`doctor` ends with **"This node can run the pipeline."** It checks imports, then
actually runs pdb2pqr and APBS on a three-atom system, because importing a
library proves less than using it. A `warn` for `sbatch` is expected off a
cluster.

Chemical-component definitions download on first use and cache under
`~/.cache/qpocket/ccd`. The standard residues ship with the package, so an apo
protein needs no network; a novel ligand does, once.

---

## 1. Fragment screen: structures you already have

The fastest way to see the whole pipeline work. Point it at a few local files
and name the reference's ligand. The repository ships PARG structures, so this
runs with no network at all:

```bash
uv run qpocket run \
  --mode fragment_screen \
  --input-type paths \
  --reference-path experiments/PARG/7KG6/structures/original/7KG6.cif \
  --structure-paths experiments/PARG/7KG6/structures/original/7KG6.cif \
  --structure-paths experiments/PARG/7KG6/structures/original/4B1H.cif \
  --structure-paths experiments/PARG/7KG6/structures/original/6O9Y.cif \
  --ligand-id WDG \
  --chain-id A \
  --workdir ./runs/parg-minimal
```

**About 20 seconds for three structures**, on a laptop. The run is named after
the reference, so everything lands in `./runs/parg-minimal/7KG6/`.

Check it did what you think:

```bash
cut -d, -f1,6,7 runs/parg-minimal/7KG6/metadata/master.csv
```

```
pdb_id,ligand_id,protein_state
4B1H,AR6,BOUND_STATE
6O9Y,M0V,BOUND_STATE
7KG6,WDG,BOUND_STATE
```

`ligand_id` is the chemical-component ID of the ligand each structure turned out
to hold — the pipeline records it rather than guessing it back from a filename,
and interaction detection uses it. A structure with no ligand reports
`UNBOUND_STATE`, gets its protonated protein anyway, and the run continues.

With many structures, use a CSV instead of repeating `--structure-paths`:

```bash
uv run qpocket run --mode fragment_screen --csv-path structures.csv \
  --ligand-id WDG --chain-id A --workdir ./runs/my-screen
```

Residue renumbering against UniProt is **off** by default in this mode, since
your files may not correspond to a single UniProt entry. Turn it on with
`--renumber-residues` when they do.

---

## 2. PDB query: let it find the structures

Give it one reference and let it fetch every homologue that passes your filters:

```bash
uv run qpocket run \
  --pdb-id 7KG6 \
  --chain-id A \
  --ligand-id WDG \
  --sequence-identity 0.95 \
  --resolution-limit 2.0 \
  --workdir ./runs/parg-query
```

**Keep the first one small.** There is no cap on how many structures a query
returns, and every hit is downloaded, aligned, cleaned, protonated and
featurised. For PARG:

| filters | structures fetched |
|---|---|
| `--sequence-identity 0.95 --resolution-limit 2.0` | 19 |
| `--sequence-identity 0.9 --resolution-limit 2.5` | 32 |

Tighten `--sequence-identity` and `--resolution-limit` until the count looks
sane before widening. `--enforce-same-uniprot` restricts to one protein, which
is usually what you want for a conformational study.

Here renumbering to UniProt is **on** by default, so residue numbers agree
across structures. That needs UniProt to be reachable; add
`--no-renumber-residues` if it is not.

### Going faster

Per-structure work runs in parallel. `--workers 0` (the default) uses half the
cores, `-1` uses all, and `1` disables parallelism — useful when you want a log
you can read top to bottom:

```bash
uv run qpocket run ... --workers -1
```

The slowest step is APBS. For a geometry-only run, drop it:

```bash
uv run qpocket run ... --no-apbs
```

That skips pdb2pqr too, when nothing else needs it. Fingerprints from different
feature sets are **not comparable**, and each run records the set it used in
`metadata/features.json`.

---

## 3. Look at the results

```bash
uv run qpocket explore ./runs
```

It opens in your browser. The path may be one run or a folder of runs; every run
found appears in the dropdown, and you can type another path into the
**"…or open a path"** box.

The notebook ships inside the package, so this works from an installed wheel
with no checkout present.

The notebook shows the distance matrices, the embedding, the clustering and the
pocket point clouds.

### Three ways to cluster

Pick a method at the top of the **Clustering** section; every view below follows
the one you choose.

| method | what it does | knobs |
|---|---|---|
| Affinity propagation | clusters the combined score, as the pipeline does | damping, preference percentile |
| One hierarchical cut | cuts the combined score's tree at one height | cut height |
| Two stages | groups by shape, then splits each group by chemistry | shape cut, chemistry cut, minimum group to split |

The dendrogram draws the tree being cut, with a dashed line at the height, so
moving a slider shows you what it does. Each result reports its cluster count,
its silhouette, and how far it sits from the clustering stored with the run
(ARI, and how many pockets moved).

Only the two-stage result can be **saved** as the run's clustering, because that
is the form a run stores.

To change the notebook rather than use it:

```bash
uv run qpocket explore ./runs --edit
```

To reconsider the clustering without recomputing anything, adjust the two-stage
thresholds in the notebook and save, or from the command line:

```bash
uv run qpocket recluster ./runs/parg-minimal/7KG6 --shape-threshold 0.25
```

The previous clustering is archived under `metadata/clustering_history/` rather
than overwritten.

---

## What you get

```
runs/parg-minimal/7KG6/
├── qpocket_pipeline.log
├── logs/                       ← one log per structure, per stage
├── structures/
│   ├── original/               ← inputs, or downloads
│   ├── aligned/                ← superposed on the reference
│   ├── cleaned/                ← waters and unwanted residues removed
│   ├── protonated/             ← docking-ready: protein + ligand + hydrogens
│   └── extracted_ligands/
├── pockets/<pdb_id>/
│   ├── binding_site.csv        ← the pocket point cloud
│   └── featurized.csv          ← its features, one row per point
├── metadata/
│   ├── master.csv              ← one row per structure, with ligand_id
│   ├── spectral_distance_matrix.csv
│   ├── shape_distance_matrix.csv
│   ├── feature_distance_matrix.csv
│   ├── clustering_metadata.csv
│   ├── observed_interactions_freq.csv    ← what each residue actually does
│   ├── potential_interactions_freq.csv
│   └── binding_pocket_utilization.csv
└── screening/actives.smi
```

`structures/protonated/<ID>.pdb` is the docking-ready structure: the protein
with pdb2pqr hydrogens, the ligand protonated for pH 7.4, and hydroxyl hydrogens
oriented against the assembled complex. Interactions are read from this file, and
it is what you should hand to a docking engine. See
[Interactions](interactions.md).

---

## When something goes wrong

**Read the log for the structure, not the run.** Each structure writes its own
under `logs/`, so a failure in one does not bury the others.

| symptom | cause |
|---|---|
| `analysis_error` in the interaction tables | a chemical component could not be perceived; the log names which one |
| `no_ligand_found` | expected for an apo structure, and not an error |
| every structure `FAILED` | usually the reference: check `--chain-id` and `--ligand-id` exist in it |
| APBS or pdb2pqr errors | run `uv run qpocket doctor` first; it runs both for real |
| a component will not download | set `QPOCKET_CCD_CACHE` to a directory you populated on a connected machine |

### The notebook says "kernel not found"

The page is talking to a marimo server that is no longer running, or to a
different one than it was opened against. The headings still show, because they
rendered before the connection was lost, but nothing else does.

`marimo run` stops when the shell that started it stops, so closing the terminal
or pressing Ctrl-C leaves the tab stranded. Each server also issues its own
token, so a tab opened against one instance cannot attach to the next.

Start it fresh and open the URL it prints, rather than reusing an old tab:

```bash
pkill -f "marimo run"                    # stop anything still holding the port
uv run qpocket explore ./runs --headless
```

One cause is now gone: the notebook used to carry inline dependency metadata,
which made marimo offer to run it in a sandboxed venv — the prompt defaults to
yes, and that venv cannot see `qpocket`, so the kernel died on import and the
page reported exactly this.

Check the server is actually up before blaming the page:

```bash
curl -sI http://localhost:2718 | head -1     # expect: HTTP/1.1 200 OK
```

For a human-readable console log instead of JSON:

```bash
uv run qpocket run ... --log-format text
```

To see exactly what a config will do before running it:

```bash
uv run qpocket init          # writes qpocket_config.yaml, every setting documented
uv run qpocket validate --config qpocket_config.yaml
uv run qpocket run --config qpocket_config.yaml
```
