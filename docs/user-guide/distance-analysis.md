# Distance Analysis

`qpocket run` computes no distances. Distance analysis is a per-site step, run after
the sites exist, either in the explorer or from the command line.

## Running it

In the explorer (`qpocket explore RUNS`), the **Binding sites** section has a site
picker, a status line and three buttons: **Shape clustering**, **Featurise** and
**Two-stage clustering**. **Regroup sites** changes the site grouping itself.

From the command line:

```bash
qpocket site list      runs/7KG6
qpocket site shape     runs/7KG6 --site S1-H21   # shape only, no features
qpocket site featurize runs/7KG6 --site S1-H21
qpocket site cluster   runs/7KG6 --site S1-H21   # shape, then chemistry
qpocket site regroup   runs/7KG6 --cavity-overlap 0.4
```

Shape clustering (clustering mode `shape_only`) needs no featurisation, so it is the
quick first look. Featurise writes `featurized.csv` beside each cloud; two-stage
clustering needs it. Results land in `sites/site_NN/metadata/`. On a cluster, see
[Running on a cluster](hpc.md).

These steps read the settings saved with the run in `<run>/metadata/config.yaml`.
Edit that file to change features, charges, distance or clustering settings for
later site steps; the YAML you passed to `qpocket run` is not read again.

## Pipeline

```
sites/site_NN/pockets/<pdb_id>/featurized.csv  (per pocket)
      │
      ▼
compute_geodesic()        ← shortest-path distances on neighbourhood graph
      │
      ▼
compute_spectral_fingerprint()  ← heat-kernel Laplacian → eigenvalues + GFT
      │
      ▼
pairwise L2 distance matrix
      │
      ▼
Affinity Propagation clustering
      │
      ▼
  sites/site_NN/metadata/spectral_distance_matrix.csv
  clustering_metadata.csv
  selected_representatives.csv
```

## Features per pocket point

| Feature | Description |
|---|---|
| `apbs` | Electrostatic potential (APBS / LPBE) |
| `cp` | Coulomb potential from partial charges |
| `lp` | Lipophilicity potential (Fermi-function weighted) |
| `h_donor` | Gaussian proximity to H-bond donors |
| `h_acceptor` | Gaussian proximity to H-bond acceptors |
| `aromatic` | Gaussian proximity to aromatic ring centroids |

## Geodesic graph

Points are connected if their Euclidean distance ≤ `geo_graph_neighbor_radius` (default 2.0 Å).  Shortest-path lengths (Dijkstra) form the geodesic distance matrix, which captures surface topology rather than straight-line distances.

## Spectral fingerprint

$$
\mathbf{W}_{ij} = e^{-d_\text{geo}(i,j)^2 / \sigma^2}, \quad
\mathbf{L} = \mathbf{I} - \mathbf{D}^{-\frac{1}{2}} \mathbf{W} \mathbf{D}^{-\frac{1}{2}}
$$

Eigenvalues $\boldsymbol{\lambda} \in \mathbb{R}^k$ and GFT power spectrum $|\mathbf{U}^\top \mathbf{F}|^2 \in \mathbb{R}^{k \times d}$ are concatenated into a fixed-length fingerprint.

## Combined distance metric

$$
d(i,j) = \alpha \|\boldsymbol{\lambda}_i - \boldsymbol{\lambda}_j\| + (1-\alpha)\,\beta\,\|\mathbf{f}_i - \mathbf{f}_j\|
$$

## Clustering

The default two-stage mode cuts a hierarchical (average-linkage) tree on shape,
then another within each shape cluster on chemistry. Each cut is by default
`relative` at 0.5: half the height of that tree's highest merge. `threshold` (an
absolute height) and `count` (a fixed `k >= 2`) are the alternatives; see
[Configuration](../configuration.md#distanceclustering). The cluster count is
never chosen by silhouette.

## Clustering — Affinity Propagation

Affinity Propagation is applied to the precomputed distance matrix (converted to a similarity matrix via negation).  The number of clusters is determined **automatically** — no `k` needs to be specified.

Key parameters:

| Config key | Default | Description |
|---|---|---|
| `affinity_damping` | `0.9` | Damping factor ∈ [0.5, 1.0) |
| `affinity_max_iter` | `300` | Maximum message-passing iterations |