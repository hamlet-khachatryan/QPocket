# Output Files

## Directory layout

```
<workdir>/<reference_pdb_id>/
├── qpocket_pipeline.log             ← Eliot structured log (JSON or text)
├── structures/
│   ├── original/                    ← downloaded .cif files
│   ├── aligned/                     ← structures superposed onto the reference
│   ├── cleaned/                     ← cleaned .pdb (protein-only + cofactors)
│   └── extracted_ligands/           ← per-structure ligand .pdb files
├── pockets/
│   └── <pdb_id>/
│       ├── binding_site.csv         ← final pocket point cloud (x, y, z)
│       ├── binding_site_initial.csv ← pre-crop cloud (io.save_intermediate)
│       └── featurized.csv           ← x, y, z + 6 pharmacophoric features
├── metadata/
│   ├── master.csv                   ← per-structure metadata
│   ├── spectral_distance_matrix.csv ← N×N pairwise distance matrix
│   ├── clustering_metadata.csv      ← cluster assignment per structure
│   ├── selected_representatives.csv ← Affinity Propagation exemplars
│   ├── observed_interactions.h5     ← PLIP interactions (HDF5)
│   ├── observed_interactions_freq.csv
│   ├── potential_interactions.h5    ← geometric contacts (HDF5)
│   ├── potential_interactions_freq.csv
│   └── binding_pocket_utilization.csv
└── screening/
    └── actives.smi                  ← SMILES of co-crystallised ligands
```

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

### `metadata/clustering_metadata.csv`

| Column | Description |
|---|---|
| `pdb_id` | Structure identifier |
| `cluster_id` | Cluster index (0-based) |
| `is_representative` | `True` if this is the Affinity Propagation exemplar |
| `dist_to_exemplar` | Distance to cluster exemplar |

### `pockets/<pdb_id>/featurized.csv`

Columns: `x`, `y`, `z`, `apbs`, `cp`, `lp`, `h_donor`, `h_acceptor`, `aromatic`.

### Interaction HDF5 files

Keys are PDB IDs; values are byte-encoded lists of:

- **potential**: `"RESNAME_RESNUM"` strings (all residues within 4 Å of cavity)
- **observed**: `"RESNAME_RESNUM|InteractionType"` strings from PLIP