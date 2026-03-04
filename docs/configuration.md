# Configuration Reference

QPocket uses [Hydra](https://hydra.cc/) for configuration management.  The built-in defaults live in `conf/config.yaml`; they can be overridden by an external YAML file or by CLI flags.

---

## `mode`

```yaml
mode: "pdb_query"   # or "fragment_screen"
```

Controls the data-retrieval stage.

---

## `reference`

```yaml
reference:
  pdb_id: "6HP0"                       # 4-character PDB ID
  chain_id: "A"                         # 1–2 uppercase letters
  ligand_id: "LKQ"                      # residue name of reference ligand
  pocket_definition: "Solvent"          # "Solvent" | "Interactions"
  pocket_mode: "cavity_detection"       # "cavity_detection" | "bounding_box"
  bounding_box:
    center: null                        # [x, y, z]   — bounding_box mode only
    dimensions: null                    # [dx, dy, dz] — bounding_box mode only
```

| Field | Description |
|---|---|
| `pocket_definition` | **Solvent** — full cavity point cloud.  **Interactions** — crop to PLIP-observed interacting residues. |
| `pocket_mode` | **cavity_detection** — run CavFind; use reference bounding box for other structures.  **bounding_box** — use an explicit box. |

---

## `pdb_query`

```yaml
pdb_query:
  sequence_identity: 1.0      # (0, 1]
  resolution_limit: 2.5       # Å
  min_length: 20
  max_length: null
  experimental_method: null   # e.g. "X-RAY DIFFRACTION"
  enforce_same_uniprot: false
```

---

## `fragment_screen`

```yaml
fragment_screen:
  input_type: "csv"           # "csv" | "paths"
  csv_path: null              # CSV with 'pdb_path' column
  structure_paths: null       # explicit list of PDB paths
  reference_index: 0          # index of the reference structure
  interaction_padding: 5.0    # Å — used in Interactions pocket mode
```

---

## `alignment`

```yaml
alignment:
  renumber_residues: true     # remap to UniProt residue numbers
  manual_uniprot_id: null     # override auto-detected UniProt ID
```

---

## `cleaning`

```yaml
cleaning:
  allowed_cofactors: null     # "|"-separated, e.g. "HEM|MG|CL"
```

---

## `io`

```yaml
io:
  log_format: "json"          # "json" (NDJSON) | "text" (pretty-printed)
  save_intermediate: false    # save pre-crop binding_site_initial.csv
  compression: "gzip"         # HDF5 compression: "gzip" | "none"
```

---

## `spectral_distance_matrix`

```yaml
spectral_distance_matrix:
  geo_graph_neighbor_radius: 2.0   # geodesic graph edge threshold (Å)
  paralell_jobs: -1                # joblib workers (-1 = all cores)
  global_k: 100                    # fingerprint vector length
  sigma: 10.0                      # heat-kernel σ
  max_points: 5000                 # skip clouds larger than this
  alpha: 0.85                      # geodesic weight ∈ [0, 1]
  beta: 12.5                       # feature cost scaling factor
  affinity_damping: 0.9            # AP damping ∈ [0.5, 1.0)
  affinity_max_iter: 300           # maximum AP iterations
```

### Distance metric

$$
d(i,j) = \alpha \cdot \|\boldsymbol{\lambda}_i - \boldsymbol{\lambda}_j\|_2
         + (1-\alpha) \cdot \beta \cdot \|\mathbf{f}_i - \mathbf{f}_j\|_2
$$

where $\boldsymbol{\lambda}$ are eigenvalues and $\mathbf{f}$ is the GFT feature vector.

---

## `paths`

```yaml
paths:
  workdir: null    # defaults to current working directory
```