# Quick Start

For a step-by-step walkthrough with verified commands and timings, see
[Your First Runs](user-guide/first-run.md). This page is the short reference.

## PDB query mode

Run a full analysis starting from a reference PDB entry:

```bash
qpocket run \
  --pdb-id 6HP0 \
  --chain-id A \
  --ligand-id LKQ \
  --workdir ./results \
  --sequence-identity 0.9 \
  --resolution-limit 2.5
```

This will:

1. Query the RCSB PDB for structures homologous to **6HP0:A** with ≥ 90 % sequence identity and resolution ≤ 2.5 Å.
2. Align all hits to the reference.
3. Extract the binding pocket using the **LKQ** centroid as the reference anchor.
4. Compute the spectral distance matrix and cluster with Affinity Propagation.

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
from qpocket.pipelines.stages import run_qpocket_pipeline

run_qpocket_pipeline(
    pdb_id="6HP0",
    chain_id="A",
    ligand_id="LKQ",
    workdir="./results",
    sequence_identity=0.9,
    resolution_limit=2.5,
    log_format="text",          # human-readable console output
    save_intermediate=True,      # keep pre-crop pocket CSVs
)
```

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
    │   └── extracted_ligands/  ← per-structure ligand PDB files
    ├── pockets/
    │   └── <pdb_id>/
    │       ├── binding_site.csv          ← final pocket point cloud
    │       ├── binding_site_initial.csv  ← pre-crop cloud (if save_intermediate)
    │       └── featurized.csv            ← APBS / CP / LP / geometric features
    ├── metadata/
    │   ├── master.csv                     ← one row per structure, with ligand_id
    │   ├── spectral_distance_matrix.csv
    │   ├── clustering_metadata.csv
    │   ├── selected_representatives.csv
    │   ├── observed_interactions.h5
    │   ├── observed_interactions_freq.csv
    │   ├── potential_interactions.h5
    │   ├── potential_interactions_freq.csv
    │   └── binding_pocket_utilization.csv
    └── screening/
        └── actives.smi
```