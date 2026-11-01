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
│   ├── grouping.json                ← the site parameters the cavities were grouped with
│   ├── cavities.csv                 ← every CavFind cavity, its box and its sites
│   ├── cavfind_misses.csv           ← ligands CavFind found no cavity for
│   └── S<rank>-<residue>/       ← e.g. S1-H21
│       ├── ligands.pdb              ← the site's ligands, superposed
│       ├── pockets/<pdb_id>/
│       │   ├── binding_site.csv         ← the site's cloud in that structure (x, y, z)
│       │   └── featurized.csv           ← after Featurise: x, y, z + feature columns
│       ├── metadata/                ← distance matrices, clustering, representatives
│       ├── final/                   ← after Finalise: clusters.csv, settings.json,
│       │   │                          residue_profile.csv, ligands.csv
│       │   └── history/<finalised_at>/  ← earlier finals
│       └── screening/
│           ├── box.json             ← the saved docking box (Save box)
│           ├── screen_config.yaml   ← optional, this site's screening settings
│           └── crossdock/           ← task folder: docking_jobs.csv, receptors/, ligands/,
│                                      poses/, run_docking.sbatch, crossdock_*.csv
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
| `protein_state` | `BOUND_STATE` (cavity and ligand) / `UNBOUND_STATE` (cavity only) / `FAILED` |

### `sites/<site>/metadata/clustering_metadata.csv`

| Column | Description |
|---|---|
| `pdb_id` | Structure identifier |
| `cluster_id` | Cluster index (0-based; −1 before the site is clustered) |
| `is_representative` | `True` for the cluster's representative (its medoid, or the member picked instead) |
| `dist_to_exemplar` | Distance to the representative, on the matrix the clustering used |

### `sites/<site>/final/`

Written by **Finalise**. `clusters.csv` (pdb_id, cluster, cluster_id,
is_representative, dist_to_representative, holo), `residue_profile.csv` (per cluster
and residue: contacted_pct, lining_pct, n_members, n_with_contacts), `ligands.csv`
(the site's ligands per cluster, with SMILES) and `settings.json` (when, the
clustering parameters, features, representatives, and the hash of the clustering it
froze — a different clustering makes the final stale).

### `sites/<site>/pockets/<pdb_id>/featurized.csv`

Columns: `x`, `y`, `z`, `apbs`, `cp`, `lp`, `h_donor`, `h_acceptor`, `aromatic`.

### Interaction HDF5 files

Keys are PDB IDs; values are byte-encoded lists of:

- **potential**: `"RESNAME_RESNUM_CHAIN"` strings (residues with an atom within 4 Å of the cloud)
- **observed**: `"RESNAME_RESNUM_CHAIN|InteractionType"` strings from ProLIF