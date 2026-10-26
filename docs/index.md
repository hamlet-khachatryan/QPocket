# QPocket

**QPocket** is a computational framework for the systematic analysis and benchmarking of the experimentally observed conformational space of protein binding pockets.  It is designed for drug design workflows requiring:

- Large-scale retrieval of structurally homologous proteins from the PDB
- Structure alignment and residue renumbering to a canonical UniProt reference
- Discovery of every ligand-occupied binding site, with one pocket point cloud per site per structure
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
│    └── Affinity Propagation clustering                      │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
 sites/site_NN/metadata/  ─  spectral_distance_matrix.csv
              clustering_metadata.csv
              selected_representatives.csv
              observed / potential interaction H5
```

---

## Key concepts

### Binding-site discovery

No reference ligand is needed. Every ligand in every structure is found, ligands
whose atoms lie within `sites.link_cutoff` of each other across structures are
grouped into a **site**, and each site gets one pocket point cloud per structure.
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