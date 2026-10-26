# Spectral Distance — Mathematical Reference

This document formalises every mathematical step in the spectral distance pipeline
(`spectral_distance.py` + `features.py`). It is intended as a self-contained
reference for documentation, review, and future development.

---

## Overview

The pipeline converts a set of protein binding-pocket **point clouds** into a
symmetric pairwise **distance matrix** that jointly encodes:

1. **Pocket geometry** — via graph-Laplacian eigenvalues (spectral shape fingerprint).
2. **Physicochemical properties** — via a Graph Fourier Transform of per-point features.

```
PDB/PQR files
      │
      ▼
┌─────────────────────────┐
│  Stage 1 · Featurisation│  (features.py)
│  6 physico-chem fields  │
└──────────┬──────────────┘
           │  featurized.csv  (N × 9 = x,y,z + 6 features)
           ▼
┌─────────────────────────────┐
│  Stage 2 · Geodesic Graph   │  compute_geodesic()
│  ε-ball graph + Dijkstra    │
└──────────┬──────────────────┘
           │  D_geo  (N × N),  F  (N × 6)
           ▼
┌────────────────────────────────────┐
│  Stage 3 · Z-score Normalisation   │  data_processing_spectral()
│  Global scaler across all pockets  │
└──────────┬─────────────────────────┘
           │  F_norm  (N × 6)
           ▼
┌────────────────────────────────────────┐
│  Stage 4 · Spectral Fingerprint        │  compute_spectral_fingerprint()
│  Heat kernel → Laplacian → eigen → GFT │
└──────────┬─────────────────────────────┘
           │  λ  (K,)   spec_feat  (K·D,)
           ▼
┌──────────────────────────────────────┐
│  Stage 5 · Pairwise Distance Matrix  │  compute_spectral_distance_matrix()
│  Weighted L2 on fingerprints         │
└──────────┬───────────────────────────┘
           │  dist_matrix  (M × M)
           ▼
┌───────────────────────────────┐
│  Stage 6 · Clustering         │  core/clustering.py
│  Tree cut: relative height    │
└───────────────────────────────┘
```

**Notation**

| Symbol | Meaning |
|--------|---------|
| N | number of grid points in a single pocket |
| M | number of pockets in the dataset |
| D | number of physicochemical features per point (D = 6) |
| K | number of Laplacian eigenmodes retained (`global_k`, default 100) |
| r | geodesic-graph neighbour radius (default 2.0 Å) |
| σ | heat-kernel bandwidth (default 10.0, dimensionless after normalisation) |
| α | eigenvalue weight in final distance (default 0.85) |
| β | spectral-feature scale factor (default 12.5) |

---

## Stage 1 — Physicochemical Featurisation

> **Source**: `features.py · featurize()`

Each pocket grid point **p ∈ ℝ³** is assigned **D = 6** scalar features.
The resulting matrix is **F ∈ ℝ^{N × 6}**.

### 1.1 APBS Electrostatic Potential  (`apbs`)

The **linearised Poisson–Boltzmann equation** (LPBE) is solved numerically by APBS:

```
∇ · [ε(r) ∇φ(r)] − κ²(r) φ(r) = −ρ_f(r) / ε_0
```

| Variable | Value / meaning |
|----------|-----------------|
| ε(r) | dielectric: 2.0 inside protein, 78.54 (water) outside |
| κ | Debye–Hückel screening length (function of ionic strength) |
| ρ_f | fixed charge density (from PQR partial charges) |
| Ionic strength | 0.150 M NaCl + 0.002 M MgCl₂ (charge-balanced) |
| T | 298.15 K |

The solution φ(r) is computed on a regular grid and then **trilinearly interpolated**
to each pocket point p.

### 1.2 Coulomb Potential  (`cp`)

Classical point-charge approximation with unit conversion to kcal/mol·e:

```
CP(p) = 557 × Σᵢ [ qᵢ / |p − rᵢ|² ]
```

- qᵢ — partial charge of protein atom i (PARSE force field, from PQR)
- rᵢ — 3D position of atom i
- 557 — conversion constant (≈ 332 / 0.6, kcal·Å/(mol·e²))

### 1.3 Lipophilic Potential  (`lp`)

Fermi–Dirac-weighted hydrophobicity field:

```
w_i(p) = [e^{−c₁ c₂} + 1] / [e^{c₁(d(p,rᵢ) − c₂)} + 1]

LP(p) = Σᵢ w_i(p) · hpᵢ / max(Σᵢ w_i(p), 1e-6)
```

- c₁ = 1.0,  c₂ = σ_hydro = 1.5 Å
- hpᵢ — atom hydrophobicity (look-up table in `qpocket.constants.atomic_hydrophobicity`)
- The Fermi function provides a smooth, distance-decaying weight

### 1.4 H-Bond Donor Field  (`h_donor`)

Gaussian kernel density from **protein H-bond acceptor** atoms
(a ligand donor interacts with a protein acceptor):

```
HD(p) = Σⱼ exp(−|p − aⱼ|² / (2 σ_hbond²))
```

- aⱼ — coordinates of protein H-bond acceptor atoms
- σ_hbond = 2.2 Å

### 1.5 H-Bond Acceptor Field  (`h_acceptor`)

Gaussian kernel density from **protein H-bond donor** atoms:

```
HA(p) = Σₖ exp(−|p − dₖ|² / (2 σ_hbond²))
```

- dₖ — coordinates of protein H-bond donor atoms
- σ_hbond = 2.2 Å

### 1.6 Aromatic Field  (`aromatic`)

Gaussian kernel density from **aromatic ring centroids** cₗ:

```
AR(p) = Σₗ exp(−|p − cₗ|² / (2 σ_arom²))
```

- σ_arom = 2.7 Å
- Ring atoms used: PHE/TYR (CG,CD1,CD2,CE1,CE2,CZ), TRP (CG,CD1,CD2,NE1,CE2,CE3),
  HIS (CG,ND1,CD2,CE1,NE2)

**Summary — Feature Matrix**

```
F = [apbs | cp | lp | h_donor | h_acceptor | aromatic]   ∈ ℝ^{N × 6}
```

---

## Stage 2 — Geodesic Graph Construction

> **Source**: `spectral_distance.py · compute_geodesic()`

The goal is to capture **surface connectivity** of the pocket, not straight-line
(Euclidean) distances, by building a proximity graph and computing shortest paths.

### Algorithm

```
Input : coords  ∈ ℝ^{N × 3},  F  ∈ ℝ^{N × 6},  radius r

1.  D_euc[i,j] ← ||coordsᵢ − coordsⱼ||₂        (N × N Euclidean distances)

2.  A[i,j] ← D_euc[i,j]  if D_euc[i,j] ≤ r      (ε-ball weighted adjacency)
             0             otherwise

3.  D_geo ← ShortestPath(A, method="Dijkstra")    (all-pairs geodesic distances)

4.  For disconnected pairs (D_geo[i,j] = ∞):
        D_geo[i,j] ← 1.5 × max{ D_geo[i,j] : finite }

5.  D_geo ← D_geo / max(D_geo)                   (normalise to [0, 1])

6.  w ← 1/N · 𝟏_N                               (uniform node weights)

Output: D_geo  ∈ [0,1]^{N × N},  F,  w
```

**Why geodesic?**
Direct Euclidean distance between two points on opposite sides of a pocket
wall is small but topologically meaningless. Shortest-path distance on the
ε-ball graph respects the pocket's surface manifold structure.

---

## Stage 3 — Global Z-Score Normalisation

> **Source**: `spectral_distance.py · data_processing_spectral()`

Features are normalised *globally* across all M pockets before spectral analysis
so that no single feature dominates due to scale differences.

```
F_stacked ← vstack([F¹, F², …, Fᴹ])         shape: (Σ Nᵢ) × 6

μ_d  ← mean of column d  over all points and all pockets
σ_d  ← std  of column d  over all points and all pockets

F_norm[p, d] ← clip((F[p, d] − μ_d) / σ_d, −5, 5)
```

Clipping at ±5 standard deviations suppresses numerical outliers.

---

## Stage 4 — Spectral Fingerprint

> **Source**: `spectral_distance.py · compute_spectral_fingerprint()`

This is the core mathematical stage. Each pocket is converted into two
fixed-length vectors that form its **spectral fingerprint**.

### Step 4.1 — Gaussian Heat Kernel Affinity Matrix

```
W[i,j] = exp(−D_geo[i,j]² / σ²),   W[i,i] = 0
```

W converts geodesic distances (dissimilarities) into pairwise affinities
(similarities). σ controls the locality of the kernel:
- Small σ → only immediate neighbours have high affinity.
- Large σ → affinity decays slowly; more global connectivity.

### Step 4.2 — Symmetric Normalised Graph Laplacian

```
degree_i = Σⱼ W[i,j]                         (weighted degree)

D^{−1/2} = diag(degree_i^{−1/2})            (diagonal matrix)

L = I − D^{−1/2} W D^{−1/2}                 (L_sym)
```

Properties of L:
- Symmetric: L = Lᵀ
- Positive semi-definite: all eigenvalues λ ∈ [0, 2]
- Trivial eigenvalue: λ₀ = 0 with eigenvector ∝ (√degree_i)

The normalised Laplacian is preferred over the combinatorial Laplacian because
it is invariant to degree heterogeneity (pocket densities vary across regions).

### Step 4.3 — Partial Eigendecomposition

```
L = U Λ Uᵀ,      Λ = diag(λ₀ ≤ λ₁ ≤ … ≤ λ_{N−1})

Compute only the K+1 smallest eigenpairs:
    linalg.eigh(L, subset_by_index=[0, K])

Drop trivial mode:
    λ  ← [λ₁, …, λ_K]          ∈ ℝ^K       (eigenvalue fingerprint)
    U  ← [u₁, …, u_K]          ∈ ℝ^{N × K} (graph Fourier basis)
```

**Eigenvalue interpretation**:
λ_k is the *graph frequency* of mode k. Small eigenvalues correspond to
**low-frequency, global** pocket shape modes; large eigenvalues correspond to
**high-frequency, local** surface detail.

The K-dimensional vector λ is the **shape signature** of the pocket.

### Step 4.4 — Graph Fourier Transform (GFT)

Project the feature matrix onto the Laplacian eigenbasis:

```
F̂ = Uᵀ @ F_norm              ∈ ℝ^{K × D}     (graph Fourier coefficients)

spec_feat = |F̂|²              ∈ ℝ^{K × D}     (power spectrum, element-wise)
```

Taking the squared magnitude (power spectrum) ensures **sign invariance**:
eigenvectors are unique only up to a global sign flip, so |F̂|² is well-defined.

`spec_feat[k, d]` = energy of feature d at graph frequency k.
Flattened to a 1-D vector: `spec_vec ∈ ℝ^{K·D}`.

### Padding / Truncation

Pockets with fewer than K+1 points yield fewer eigenmodes. To keep fingerprints
of uniform length K across all pockets:

```
if len(λ) < K:  pad λ with zeros at the high-frequency end
if len(λ) > K:  truncate to first K entries

Same for spec_feat rows.
```

Zero-padding high-frequency components is physically meaningful: modes beyond
the graph's Nyquist limit carry no signal.

### Fingerprint Summary

| Vector | Symbol | Shape | Encodes |
|--------|--------|-------|---------|
| Eigenvalue fingerprint | λ | (K,) | Pocket geometry / topology |
| Spectral feature vector | spec_vec | (K·D,) | Chemistry at each graph frequency |

---

## Stage 5 — Pairwise Spectral Distance Matrix

> **Source**: `spectral_distance.py · compute_spectral_distance_matrix()`

For every pair of pockets (i, j):

```
d_shape(i,j)  = ||λᵢ − λⱼ||₂                     (shape distance)

d_chem(i,j)   = ||spec_vecᵢ − spec_vecⱼ||₂        (chemical-spectral distance)

d(i,j) = α · d_shape(i,j) + (1−α) · β · d_chem(i,j)
```

**Default hyperparameters**:

| Parameter | Default | Role |
|-----------|---------|------|
| α | 0.85 | Weight of shape vs chemistry (α → 1 = shape-only) |
| β | 12.5 | Re-scales d_chem to match the magnitude of d_shape |
| K | 100 | Number of eigenmodes retained |
| σ | 10.0 | Heat-kernel bandwidth |
| r | 2.0 Å | Geodesic-graph neighbour radius |

The matrix **D ∈ ℝ^{M × M}** is symmetric with D[i,i] = 0.

### Interpretation

- **d_shape = 0**: pockets have identical graph-Laplacian spectra → same topology/shape.
- **d_chem = 0**: pockets have identical physicochemical distribution at every frequency.
- The combined distance rewards pockets that are similar in *both* geometry and chemistry.

---

## Stage 6 — Clustering

> **Source**: `core/clustering.py · two_stage_labels(), cluster_labels()`

By default the pockets are grouped in two stages: first on the shape distances,
then on the chemistry distances inside each shape group. Each stage is a
hierarchical (average-linkage) tree on the precomputed distances, cut by one of
three criteria (`distance.clustering.{shape,chem}.criterion`):

```
Input: D  (M × M precomputed distance matrix)
Z ← linkage(D, average);  h_max ← max merge height in Z

criterion "count":     cut Z into exactly k clusters            (k set by the user)
criterion "threshold": cut Z at height t                        (t in D's units)
criterion "relative":  cut Z at height t · h_max,  0 < t ≤ 1    (default t = 0.5)

Output: cluster labels,  medoid_indices (one per cluster)
```

The relative cut does not depend on scale: multiplying D by a constant leaves the
labels unchanged, so one setting suits shape and chemistry distances alike, and
in stage two each shape group's chemistry tree is cut at its own `h_max`. A stage
with fewer than three pockets stays one cluster. `kmedoids` (with a set k) and
`affinity` (which chooses its own count) are offered as alternatives.

Medoids (actual pocket structures, not centroids) serve as **representative
structures** for each cluster — the most central real pocket in each group.

---

## Full Algorithm — End-to-End Pseudocode

```
Given: PDB files {P₁, …, P_M},  hyperparameters {r, σ, K, α, β, t}

═══ Featurisation ══════════════════════════════════════════
For each pocket Pₘ:
    1.  Solve LPBE (APBS) → φ(p) for each grid point p
    2.  Compute CP(p) = 557 · Σᵢ qᵢ / |p−rᵢ|²
    3.  Compute LP(p) via Fermi-weighted hydrophobicity
    4.  Compute HD(p), HA(p) via Gaussian HB kernels (σ=2.2Å)
    5.  Compute AR(p) via Gaussian aromatic kernels (σ=2.7Å)
    6.  F ← [φ, CP, LP, HD, HA, AR]  ∈ ℝ^{N×6}

═══ Geodesic Graph ═════════════════════════════════════════
For each pocket Pₘ:
    7.  Build ε-ball graph (r = 2.0 Å) on pocket coordinates
    8.  D_geo ← Dijkstra all-pairs shortest paths
    9.  Replace ∞ → 1.5 · max_finite;  normalise to [0,1]
   10.  Cache (D_geo, F) to disk

═══ Global Normalisation ═══════════════════════════════════
   11.  F_norm ← Z-score(F_stacked), clip(·, −5, 5)

═══ Spectral Fingerprints ══════════════════════════════════
For each pocket Pₘ:
   12.  W  ← exp(−D_geo² / σ²);  diag(W) = 0
   13.  L  ← I − D^{−½} W D^{−½}
   14.  (λ, U) ← eigh(L)[1:K+1]         # drop λ₀ = 0
   15.  F̂  ← Uᵀ F_norm
   16.  spec_vec ← flatten(|F̂|²)
   17.  Pad/truncate λ, spec_vec to length K / K·D

═══ Distance Matrix ════════════════════════════════════════
For each pair (i, j):
   18.  d_shape ← ||λᵢ − λⱼ||₂
   19.  d_chem  ← ||spec_vecᵢ − spec_vecⱼ||₂
   20.  D[i,j]  ← α · d_shape + (1−α) · β · d_chem

═══ Clustering ═════════════════════════════════════════════
   21.  Z ← linkage(D_shape);  shape labels ← cut Z at t · max height
   22.  Within each shape group: cut its chemistry tree the same way
   23.  Assign labels;  select medoid structures
```

---

## Hyperparameter Guide

| Parameter | Symbol | Default | Effect of increasing |
|-----------|--------|---------|---------------------|
| `global_k` | K | 100 | More eigenmode detail; higher cost; diminishing returns past ~50 |
| `sigma` | σ | 10.0 | Broader kernel → more global affinity structure captured |
| `geo_graph_neighbor_radius` | r | 2.0 Å | Denser graph → fewer disconnected components but more noise |
| `alpha` | α | 0.85 | More weight on shape; less on chemistry |
| `beta` | β | 12.5 | Amplifies chemical distance contribution |
| `clustering.*.threshold` (relative) | t | 0.5 | Higher cut → fewer, larger clusters |
| `max_points` | — | 5000 | Memory guard; pockets above this size are excluded |