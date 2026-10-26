# Output Files

## Directory layout

```
<workdir>/<reference_pdb_id>/
├── qpocket_pipeline.log             ← Eliot structured log (JSON or text)
├── structures/
│   ├── original/                    ← downloaded .cif files
│   ├── aligned/                     ← structures superposed onto the reference
│   ├── cleaned/                     ← cleaned .pdb (protein-only + cofactors)
│   ├── protonated/                  ← one protonated complex per structure
│   ├── extracted_ligands/           ← per-structure ligand .pdb files
│   ├── cavities/                    ← cavities, detected once per structure
│   └── ligands.csv                  ← every site ligand found, per structure
├── sites/
│   ├── sites.csv                    ← one row per binding site
│   ├── grouping.json                ← the link cutoff the ligands were grouped at
│   └── site_NN/
│       ├── ligands.pdb              ← the site's ligands, superposed
│       ├── contact_atoms.csv        ← protein atoms the site's ligands contact
│       ├── pockets/<pdb_id>/
│       │   ├── binding_site.csv         ← the site's cloud in that structure (x, y, z)
│       │   └── featurized.csv           ← after Featurise: x, y, z + feature columns
│       ├── metadata/                ← distance matrices, clustering, representatives
│       └── screening/               ← docking inputs and results for this site
└── metadata/
    ├── run.json                     ← layout version; older runs must be re-run
    ├── config.yaml                  ← the settings the run used
    ├── master.csv                   ← per-structure metadata
    ├── failures.csv                 ← structures that dropped out, and why
    ├── observed_interactions.h5     ← ProLIF interactions (HDF5)
    ├── observed_interactions_freq.csv
    ├── potential_interactions.h5    ← geometric contacts (HDF5)
    └── potential_interactions_freq.csv
```

`qpocket run` writes everything above except `featurized.csv`, the site `metadata/`
and `screening/`; those appear when you run a site's steps.

## Key files

### `metadata/master.csv`

| Column | Description |
|---|---|
| `pdb_id` | Structure identifier |
| `rmsd_to_ref` | Cα RMSD after alignment |
| `aligned_file` | Path to aligned PDB |
| `cleaned_file` | Path to cleaned PDB |
| `ligand_file` | Path to extracted ligand PDB (or null) |
| `protein_state` | `BOUND_STATE` / `UNBOUND_STATE` / `CAVITY_CLOSED` / `FAILED` |

### `sites/site_NN/metadata/clustering_metadata.csv`

| Column | Description |
|---|---|
| `pdb_id` | Structure identifier |
| `cluster_id` | Cluster index (0-based) |
| `is_representative` | `True` if this is the Affinity Propagation exemplar |
| `dist_to_exemplar` | Distance to cluster exemplar |

### `sites/site_NN/pockets/<pdb_id>/featurized.csv`

Columns: `x`, `y`, `z`, `apbs`, `cp`, `lp`, `h_donor`, `h_acceptor`, `aromatic`.

### Interaction HDF5 files

Keys are PDB IDs; values are byte-encoded lists of:

- **potential**: `"RESNAME_RESNUM"` strings (all residues within 4 Å of cavity)
- **observed**: `"RESNAME_RESNUM|InteractionType"` strings from PLIP