# Quick Start

For a step-by-step walkthrough with verified commands and timings, see
[Your First Runs](user-guide/first-run.md). This page is the short reference.

## PDB query mode

Run a full analysis starting from a reference PDB entry:

```bash
qpocket run \
  --pdb-id 6HP0 \
  --chain-id A \
  --workdir ./results \
  --sequence-identity 0.9 \
  --resolution-limit 2.5
```

This will:

1. Query the RCSB PDB for structures homologous to **6HP0:A** with ≥ 90 % sequence identity and resolution ≤ 2.5 Å.
2. Align all hits to the reference, clean them, and find every ligand in every structure.
3. Protonate each structure once, read its contacts, and detect its cavities once.
4. Group the ligands into **binding sites** and cut one point cloud per site per structure.

`qpocket run` stops there: no features, no distance matrices. Analyse a site next,
in the explorer or from the command line:

```bash
qpocket explore ./results                              # pick a site, press the step buttons
qpocket site list ./results/6HP0             # the sites and the steps each has done
qpocket site featurize ./results/6HP0 --site site_01
qpocket site cluster   ./results/6HP0 --site site_01
```

`qpocket site shape` clusters on shape alone, without featurising.

---

## Fragment screening mode

Analyse a pre-collected set of structures:

```bash
qpocket run \
  --mode fragment_screen \
  --csv-path my_structures.csv \
  --workdir ./screen_results \
  --reference-index 0
```

The CSV must contain a `pdb_path` column with absolute or relative paths to each PDB file.

---

## Using a configuration file

Generate a template:

```bash
qpocket init
```

Edit `qpocket_config.yaml`, then run:

```bash
qpocket run --config qpocket_config.yaml
```

---

## Python API

```python
from qpocket.config import load_qpocket_config
from qpocket.pipelines.run import run

cfg = load_qpocket_config(overrides=[
    "reference.pdb_id=6HP0", "reference.chain_id=A",
    "paths.workdir=./results", "pdb_query.sequence_identity=0.9",
    "pdb_query.resolution_limit=2.5",
    "io.log_format=text",            # human-readable console output
])
run(cfg)  # returns the run directory
```

Running it again into the same `workdir` reuses every structure whose input file and
settings are unchanged.

---

## Output directory layout

```
results/
└── 6HP0/
    ├── qpocket_pipeline.log
    ├── structures/
    │   ├── original/      ← downloaded CIF files
    │   ├── aligned/       ← structures superposed onto the reference
    │   ├── cleaned/       ← residue-cleaned PDB files
    │   ├── protonated/    ← docking-ready: protein + ligand + hydrogens
    │   ├── extracted_ligands/  ← per-structure ligand PDB files
    │   ├── cavities/      ← every structure's detected cavities
    │   └── ligands.csv    ← every site ligand found, per structure
    ├── sites/
    │   ├── sites.csv                      ← one row per site
    │   ├── grouping.json                  ← the cutoff the ligands were grouped at
    │   └── site_01/
    │       ├── ligands.pdb                ← the site's ligands, superposed
    │       ├── contact_atoms.csv
    │       ├── pockets/<pdb_id>/binding_site.csv   ← the site's cloud in that structure
    │       │                                       ← (+ featurized.csv after Featurise)
    │       ├── metadata/                  ← distances, clustering, representatives
    │       └── screening/                 ← docking inputs and results for this site
    └── metadata/
        ├── run.json                       ← layout version; older runs must be re-run
        ├── reference.json                 ← the reference every structure is aligned to
        ├── parts/<pdb_id>.json            ← one record per structure (resume, SLURM)
        ├── master.csv                     ← one row per structure
        ├── failures.csv                   ← structures that dropped out, and why
        └── config.yaml                    ← the settings the run used
```