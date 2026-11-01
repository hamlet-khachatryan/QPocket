# QPocket

**QPocket** is a computational framework for the systematic analysis and benchmarking of the experimentally observed conformational space of protein binding pockets.  It is designed for drug design workflows requiring:

- Large-scale retrieval of structurally homologous proteins from the PDB
- Structure alignment and residue renumbering to a canonical UniProt reference
- Binding sites built from CavFind cavities across all structures, with each structure's own cavity as its pocket cloud
- Spectral graph fingerprinting and pairwise distance matrix computation
- Clustering of pocket conformations (shape, then chemistry) with percentile cuts
- Co-crystallised ligand extraction, interaction fingerprints and per-cluster residue profiles
- Cross-docking of the cluster representatives, scored by in-place pose RMSD
- An interactive explorer (Site, Exploration and Screening tabs)

---

## Architecture overview

```
PDB / fragment CSV
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 – Data Retrieval                                   │
│  PDB query  ─OR─  local structure files (fragment CSV)      │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 – Preprocessing                                    │
│  align   – primary (+ partner) chains, UniProt renumbering  │
│  clean   – protein + allowed cofactors; modified → parent   │
│  find ligands · protonate · contacts · cavities (once)      │
│  group ligands into sites · cut one cloud per site          │
└─────────────────────────────────────────────────────────────┘
      │   qpocket run stops here
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 – Per-site Analysis (explorer or qpocket site …)   │
│  featurise → distance matrices                              │
│    ├── Geodesic graph + shortest paths                      │
│    ├── Heat-kernel Laplacian → spectral fingerprints        │
│    ├── Pairwise L2 distance (eigenvalue + feature)          │
│    └── clustering (hierarchical, percentile cuts)           │
│  representatives → Finalise → sites/<site>/final/           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 – Cross-docking (explorer Screening tab or         │
│  qpocket screen …): box → prepare → run → analyse           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
 sites/<site>/metadata/  ─  spectral_distance_matrix.csv
              clustering_metadata.csv
              selected_representatives.csv
              observed / potential interaction H5
```

---

## Key concepts

### Binding-site discovery

No reference ligand is needed. CavFind cavities that overlap across structures
(box IoU ≥ `sites.cavity_overlap`) form a **site**; ligands are assigned to the
sites whose boxes hold them; each structure's cloud at a site is its own cavity.
See [Pocket extraction](user-guide/pocket-extraction.md).

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