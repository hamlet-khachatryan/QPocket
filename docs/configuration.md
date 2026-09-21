# Configuration Reference

QPocket is configured by a YAML file passed with `--config`. Every setting has a
default; you only write the ones you change.

```bash
qpocket init                              # writes a file containing every default
qpocket run --config qpocket_config.yaml
qpocket validate qpocket_config.yaml      # check it without running anything
```

The defaults the pipeline actually loads live in
`src/qpocket_pipeline/conf/config.yaml`, which `qpocket init` copies verbatim, so
the two cannot disagree. Individual settings can also be overridden on the
command line (`qpocket run --sigma 8.0`).

An unrecognised key is reported and ignored rather than failing the run, so a
typo tells you about itself instead of silently doing nothing.

---

## `mode`

```yaml
mode: "pdb_query"   # "pdb_query" or "fragment_screen"
```

| value | input |
|---|---|
| `pdb_query` | Search the PDB for structures similar to a reference entry. |
| `fragment_screen` | Use structure files you already have. See [Fragment screening](user-guide/fragment-screening.md). |

---

## `reference`

The pocket of interest, and how to find it.

```yaml
reference:
  pdb_id: null                # required in pdb_query mode; 4 characters, e.g. "6HP0"
  chain_id: "A"
  auxiliary_chain_ids: null   # e.g. ["B"] — chains forming an interface with chain_id
  ligand_id: null             # e.g. "LKQ" — marks the pocket
  ligand_chain_id: null       # null = search every chain (homodimers keep the ligand elsewhere)
  pocket_definition: "Solvent"       # "Interactions" or "Solvent"
  pocket_mode: "cavity_detection"    # "cavity_detection" or "bounding_box"
  bounding_box:               # only when pocket_mode is "bounding_box"
    center: null              # [x, y, z]
    dimensions: null          # [dx, dy, dz]
  interaction_padding: 5.0    # Å around contacting residues, when pocket_definition
                              # is "Interactions"
```

| setting | meaning |
|---|---|
| `pocket_definition: Solvent` | The whole solvent-accessible cavity. |
| `pocket_definition: Interactions` | The cavity cropped to residues the ligand actually contacts. Requires `ligand_id`. |
| `pocket_mode: cavity_detection` | Find the cavity geometrically, ranked against the reference pocket. |
| `pocket_mode: bounding_box` | Use the box given in `bounding_box`. |

In `fragment_screen` mode `pdb_id` is optional: the run is named after the
reference filename instead.

---

## `fragment_screen`

Only read when `mode: fragment_screen`.

```yaml
fragment_screen:
  input_type: "csv"       # "csv" or "paths"
  csv_path: null          # CSV with a 'pdb_path' column
  structure_paths: null   # explicit list of structure files
  reference_path: null    # the reference structure; names the run
  reference_index: 0      # fallback selector, used only when reference_path is null
```

`.pdb`, `.cif` and `.ent` are accepted. Structures are identified by filename,
and `metadata/inputs.csv` records where each came from.

---

## `pdb_query`

Only read when `mode: pdb_query`.

```yaml
pdb_query:
  sequence_identity: 1.0      # (0, 1]
  resolution_limit: 2.5       # Å
  min_length: 20              # residues
  max_length: null
  experimental_method: null   # e.g. "X-RAY DIFFRACTION"
  enforce_same_uniprot: false
```

---

## `alignment`

```yaml
alignment:
  renumber_residues: null     # null = auto
  manual_uniprot_id: null     # e.g. "P0DTC2"
  auxiliary_uniprot_ids: null # per-chain overrides, e.g. {B: "P00533"}
  min_auxiliary_identity: 0.70            # (0, 1]
  max_auxiliary_centroid_deviation: 15.0  # Å
  require_all_chains: true    # skip structures missing an auxiliary chain
```

`renumber_residues: null` means **on for `pdb_query`, off for `fragment_screen`**:
renumbering maps residues onto canonical UniProt positions, and local or modelled
structures usually have no UniProt entry. `true` or `false` overrides.

---

## `cleaning`

```yaml
cleaning:
  allowed_cofactors: null   # '|'-separated residues to keep, e.g. "HEM|MG|CL"
```

---

## `io`

```yaml
io:
  log_format: "json"        # "json" or "text"
  save_intermediate: false  # also write the pre-crop pocket cloud
  compression: "gzip"       # HDF5 compression: "gzip" or "none"
```

---

## `distance`

Turns each pocket into a fingerprint, compares them, and groups the result.

```yaml
distance:
  geo_graph_neighbor_radius: 2.0  # geodesic graph neighbourhood (Å)
  global_k: 100                   # spectral fingerprint length
  sigma: 10.0                     # heat-kernel width
  max_points: 5000                # larger pockets are skipped
  parallel_jobs: -1               # joblib workers (-1 = all cores)

  shape_weight: 0.85              # [0, 1]
  chem_scale: 12.5                # > 0
```

Two distances are computed per pair of pockets: a **shape** distance between
Laplacian eigenvalue vectors, and a **physicochemical** distance between
Graph-Fourier power spectra. They are blended as

$$d = \text{shape\_weight} \cdot d_{\text{shape}} + (1 - \text{shape\_weight}) \cdot \text{chem\_scale} \cdot d_{\text{chem}}$$

!!! note "The blend does not group anything in two-stage mode"
    With the default `clustering.mode: two_stage`, each stage clusters its own
    matrix. The blend is used only to decide which member of a cluster is named
    its representative. It is the grouping distance only in `single` mode.

### `distance.clustering`

```yaml
  clustering:
    mode: "two_stage"       # "two_stage" or "single"
    min_cluster_size: 4     # stage-one clusters smaller than this are not subdivided

    shape:                  # stage one: shape distances
      method: "hierarchical"  # "hierarchical", "kmedoids" or "affinity"
      linkage: "average"      # average | complete | single | weighted
      k: null                 # null = choose by silhouette over k_range
      k_range: [2, 10]        # inclusive, lower bound >= 2

    chem:                   # stage two: within each shape cluster
      method: "kmedoids"
      k: null
      k_range: [2, 6]

    single:                 # only when mode == "single"
      method: "affinity"
      damping: 0.9          # affinity propagation only, [0.5, 1.0)
      max_iter: 300         # affinity propagation only
```

| mode | behaviour |
|---|---|
| `two_stage` | Group on shape, then subdivide each group on physicochemistry. `clustering_metadata.csv` gains `shape_cluster_id` and `chem_cluster_id`. |
| `single` | Cluster the blended matrix in one pass. Reproduces pre-0.2 runs exactly. |

A stage ignores the settings its method does not use: `linkage` applies only to
`hierarchical`, `k`/`k_range` only to `hierarchical` and `kmedoids`, and
`damping`/`max_iter` only to `affinity`. They are still accepted, so switching
method needs no other edit.

---

## `slurm`

Read only by `qpocket slurm`. See [Running on a cluster](user-guide/hpc.md).

```yaml
slurm:
  account: null           # --account; null omits the directive
  partition: "compute"
  qos: null
  max_concurrent: 20      # simultaneously running array tasks (the %N)
  max_array_size: 1000    # cluster MaxArraySize; larger stages are split
  dependency: "afterany"  # "afterany" or "afterok"
  split_apbs_stage: false # not yet implemented

  environment: []         # shell lines injected verbatim into every script
  # environment:
  #   - "module load gcc/11.2"
  #   - "conda activate QPocket"

  defaults:               # used by any stage without an override
    cpus: 1
    mem: "4G"
    time: "01:00:00"

  stages:                 # keys: prepare, apbs, featurize, collect, geodesic, distance
    featurize:
      cpus: 1
      mem: "8G"
      time: "04:00:00"
    geodesic:
      cpus: 1
      mem: "8G"
      time: "02:00:00"
    distance:
      cpus: 16
      mem: "32G"
      time: "06:00:00"
```

`dependency: afterany` lets later stages start once an array finishes and skip
whatever is missing; `afterok` blocks the whole chain if any single task fails.

---

## `paths`

```yaml
paths:
  workdir: null   # defaults to the current directory
```

Results land in `<workdir>/<run name>/`, where the run name is
`reference.pdb_id`, or the reference filename in fragment-screen mode.

---

## Renamed in 0.2

Old names still work and print a notice naming their replacement.

| old | new |
|---|---|
| `spectral_distance_matrix` | `distance` |
| `spectral_distance_matrix.alpha` | `distance.shape_weight` |
| `spectral_distance_matrix.beta` | `distance.chem_scale` |
| `spectral_distance_matrix.affinity_damping` | `distance.clustering.single.damping` |
| `spectral_distance_matrix.affinity_max_iter` | `distance.clustering.single.max_iter` |
| `...clustering.<stage>.min_cluster_size` | `distance.clustering.min_cluster_size` |
| `fragment_screen.interaction_padding` | `reference.interaction_padding` |

Removed: `slurm.enabled` and `spectral_distance_matrix.max_clusters`, neither of
which was ever read.
