# QPocket

**QPocket** is a computational framework for the systematic analysis and benchmarking of the experimentally observed conformational space of protein binding pockets.  It is designed for drug design workflows requiring:

- Large-scale retrieval of structurally homologous proteins from the PDB
- Structure alignment and residue renumbering to a canonical UniProt reference
- Binding-pocket extraction with cavity detection and/or interaction-based cropping
- Spectral graph fingerprinting and pairwise distance matrix computation
- Automatic unsupervised clustering (Affinity Propagation) of pocket shapes
- Co-crystallised ligand extraction for active compound sets

---

## Architecture overview

```
PDB / fragment CSV
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 – Data Retrieval                                   │
│  fetch_structures(): PDB query  ─OR─  fragment CSV load     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 – Preprocessing                                    │
│  align_structures()     – alignment + UniProt renumbering   │
│  clean_structures()     – remove non-standard residues       │
│  extract_pockets_and_interactions()                         │
│    ├── Reference: centroid → cavity → bounding box (1 Å)   │
│    └── Others:   bbox overlap → best volumetric match       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 – Distance Analysis                                │
│  compute_spectral_distance_matrix()                         │
│    ├── Geodesic graph + shortest paths                      │
│    ├── Heat-kernel Laplacian → spectral fingerprints        │
│    ├── Pairwise L2 distance (eigenvalue + feature)          │
│    └── Affinity Propagation clustering                      │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
 metadata/  ─  spectral_distance_matrix.csv
              clustering_metadata.csv
              selected_representatives.csv
              observed / potential interaction H5
```

---

## Key concepts

### Pocket identification — two-phase bounding-box strategy

1. **Reference structure**: run cavity detection, find the cavity that overlaps the reference ligand centroid, compute its axis-aligned bounding box with **1 Å padding**.
2. **All other structures**: run cavity detection, score each cavity by the number of its grid points that fall inside the reference bounding box (**volumetric overlap**), select the best-matching cavity.

### Ligand selection

All heteroatom small molecules overlapping with the selected pocket are collected; the one with the **most heavy atoms** is returned as the representative co-crystallised ligand.

### Spectral fingerprinting

Each pocket is converted to a graph Laplacian fingerprint:

$$
\mathbf{W}_{ij} = \exp\!\left(-\frac{d_{\text{geo}}^2}{\sigma^2}\right), \quad
\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}
$$

Eigenvalues and GFT power-spectrum features are combined into a fixed-length fingerprint that is permutation- and sign-invariant.

---

## Links

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Configuration Reference](configuration.md)
- [API Reference](api/index.md)