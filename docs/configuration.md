# Configuration Reference

QPocket is configured by a YAML file passed with `--config`. Every setting has a
default; you only write the ones you change.

```bash
qpocket init                              # writes a file containing every default
qpocket run --config qpocket_config.yaml
qpocket validate qpocket_config.yaml      # check it without running anything
```

The defaults the pipeline actually loads live in
`src/qpocket/conf/config.yaml`, which `qpocket init` copies verbatim, so
the two cannot disagree. Individual settings can also be overridden on the
command line (`qpocket run --sigma 8.0`).

An unrecognised key is reported and ignored rather than failing the run, so a
typo tells you about itself instead of silently doing nothing.

The per-site steps (`qpocket site …`, the explorer's site buttons and
`qpocket slurm generate --site`) read the settings saved with the run in
`<run>/metadata/config.yaml`, not the file you passed to `qpocket run`. Edit that
file to change `features`, `charges`, `distance` or `distance.clustering` for later
site steps.

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
  pocket_definition: "Solvent"       # "Interactions" or "Solvent"
  interaction_padding: 5.0    # Å around contacting residues, when pocket_definition
                              # is "Interactions"
```

| setting | meaning |
|---|---|
| `pocket_definition: Solvent` | The whole solvent-accessible cavity. |
| `pocket_definition: Interactions` | The cavity cropped to residues the ligand actually contacts. |

```yaml
sites:                        # every ligand-occupied binding site is discovered
  min_heavy_atoms: 6          # smaller HETATM groups are not ligands; carbon-free groups such as ions are never sites
  link_cutoff: 4.0            # ligands closer than this (Å, any heavy atoms) join one site
  box_padding: 4.0            # Å around the union of a site's ligands
```

`reference.ligand_id`, `ligand_chain_id`, `pocket_mode` and `bounding_box` are
obsolete: they are ignored with a warning.

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

Allowed cofactors stay in the cleaned protein and are parametrised with it
(pdb2pqr, APBS, the protonated complex). Buffers and cryoprotectants cannot be
listed here; see `sites.keep_ligands`.

---

## `sites`

```yaml
sites:
  min_heavy_atoms: 6
  link_cutoff: 4.0
  box_padding: 4.0
  keep_ligands: []          # buffers/cryoprotectants to treat as ligands in this run
```

Crystallisation additives are never ligands: glycerol, ethylene glycol, the
PEGs, MPD, the propanediols, small alcohols, DMSO and DMF, and the common
buffers (Tris, HEPES, MES, MOPS, Bis-Tris, CAPS, CHES, PIPES, bicine,
imidazole, citrate, tartrate, malonate, succinate, acetate, formate). The full
list is `qpocket.constants.BUFFER_RESNAMES`. They never form or join a site and
never enter the protonated complex. When one is a genuine binder in your
structures, name it in `keep_ligands` (e.g. `["CIT"]`): it is then an ordinary
ligand, still not part of the protein.

---

## `io`

```yaml
io:
  log_format: "json"        # "json" or "text"
  compression: "gzip"       # HDF5 compression: "gzip" or "none"
```

---

## `features`

Which providers produce the columns of `featurized.csv`.

```yaml
features:
  enabled: ["apbs", "cp_lp", "hbond", "aromatic"]
  options: {}
```

| provider | columns | needs |
|---|---|---|
| `apbs` | `apbs` | pdb2pqr + APBS |
| `cp_lp` | `cp`, `lp` | pdb2pqr |
| `hbond` | `h_donor`, `h_acceptor` | the cleaned structure |
| `aromatic` | `aromatic` | the cleaned structure |

Dropping a provider drops its columns, and a set that needs no PQR skips pdb2pqr
and APBS altogether — the fastest way to a geometry-only run. `qpocket run
--features hbond,aromatic` and `--no-apbs` set this from the command line.
Featurising a site records what it used in `sites/site_NN/metadata/features.json`; fingerprints built from
different sets are not comparable. See [Features](user-guide/features.md).

## `interactions`

Which protein-ligand interactions ProLIF reports, and with what geometry.

```yaml
interactions:
  enabled: ["Hydrophobic", "HBDonor", "HBAcceptor", "FaceToFace", "EdgeToFace",
            "Cationic", "Anionic", "CationPi", "PiCation",
            "XBDonor", "XBAcceptor", "MetalDonor", "MetalAcceptor"]
  parameters: {}
```

ProLIF names an interaction from the ligand's side; the stored strings name it
from the protein's, so `HBDonor` is recorded as `H-bond Acceptor` and `Cationic`
as `Saltbridge (-)`. `PiStacking` is rejected: it is the union of `FaceToFace`
and `EdgeToFace`, and enabling all three counts every stack twice.

`parameters` overrides the geometry of any enabled interaction, keyed by its
ProLIF name. Configuring one that is not enabled is an error, so a typo cannot
sit in a config doing nothing. To reproduce the cutoffs PLIP published:

```yaml
interactions:
  parameters:
    HBDonor: {distance: 4.1, DHA_angle: [100, 180]}
    HBAcceptor: {distance: 4.1, DHA_angle: [100, 180]}
    Hydrophobic: {distance: 4.0}
```

See [Interactions](user-guide/interactions.md).

## `compute`

How much of the machine a local run uses.

```yaml
compute:
  workers: 0              # 0 = half the cores, -1 = all, N = exactly N
  backend: "process"      # "process" (separate interpreters) or "thread"
```

| key | meaning |
|---|---|
| `workers` | Parallel workers for per-structure work. `0` (the default) uses half the cores, `-1` all of them, `N` exactly `N`. `1` runs everything serially. |
| `backend` | `process` runs workers in separate interpreters, which is what you want for featurisation; `thread` shares memory and suits work that waits on external tools. |

Peak memory scales with `workers`, because each one can run APBS. See
[Performance](user-guide/performance.md).

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
    mode: "two_stage"       # "two_stage", "single" or "shape_only"
    min_cluster_size: 4     # stage-one clusters smaller than this are not subdivided

    shape:                  # stage one: shape distances
      method: "hierarchical"  # "hierarchical", "kmedoids" or "affinity"
      linkage: "average"      # average | complete | single | weighted
      k: null                 # cluster count; needed only for criterion "count"
      criterion: "relative"   # "relative", "threshold" or "count"
      threshold: 0.5          # relative: fraction (0, 1]; threshold: absolute height

    chem:                   # stage two: within each shape cluster
      method: "hierarchical"
      linkage: "average"
      k: null
      criterion: "relative"
      threshold: 0.5

    single:                 # only when mode == "single"
      method: "affinity"
      damping: 0.9          # affinity propagation only, [0.5, 1.0)
      max_iter: 300         # affinity propagation only
```

| mode | behaviour |
|---|---|
| `two_stage` | Group on shape, then subdivide each group on physicochemistry. `clustering_metadata.csv` gains `shape_cluster_id` and `chem_cluster_id`. |
| `single` | Cluster the blended matrix in one pass. Reproduces pre-0.2 runs exactly. |
| `shape_only` | Run stage one only: group on shape, with no physicochemical subdivision. |

A stage ignores the settings its method does not use: `linkage` applies only to
`hierarchical`, `k` only to `hierarchical` and `kmedoids`, and
`damping`/`max_iter` only to `affinity`. They are still accepted, so switching
method needs no other edit.

`criterion` says how a hierarchical stage is cut:

| criterion | cut |
|---|---|
| `relative` (default) | at `threshold` times the height of the stage's highest merge, with `threshold` in (0, 1]. Scale-free, so `0.5` means about the same on any site. |
| `threshold` | at the absolute height `threshold` (> 0). |
| `count` | into exactly `k` clusters (`k >= 2`; required, there is no automatic choice). |

`threshold` and `relative` need method `hierarchical`; `affinity` ignores the
criterion. For the chemistry stage the cut applies within each shape cluster. Pick
the cut by eye with the sliders in the explorer notebook, or try it on a finished
run with `qpocket recluster RUN_DIR --shape-criterion relative --shape-threshold 0.5`.

!!! warning "Removed: `k_range` and the silhouette choice"
    The number of clusters is no longer chosen by silhouette. `k_range` is
    obsolete; a config that still sets it gets a warning and the key is ignored.
    A config, or a run's saved `metadata/config.yaml`, from an older version
    with `k: null` and `criterion: "count"` is converted automatically to a
    hierarchical relative cut at 0.5, with a warning naming the stage; set `k` or
    a threshold to choose otherwise. A setting that gives `k` but no `criterion`
    means `criterion: "count"`. Reclustering such a run's saved
    `clustering_params.json` still needs `--shape-criterion relative
    --chem-criterion relative` (threshold defaults to 0.5) or an explicit
    `--shape-k`/`--chem-k`.

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

  environment: []         # shell lines injected verbatim into every script
  # environment:
  #   - "module load gcc/11.2"
  #   - "conda activate QPocket"

  defaults:               # used by any stage without an override
    cpus: 1
    mem: "4G"
    time: "01:00:00"

  stages:                 # keys: prepare, structure, sites, site_featurize, site_cluster
    structure:
      cpus: 1
      mem: "4G"
      time: "02:00:00"
    site_featurize:       # APBS runs here
      cpus: 1
      mem: "8G"
      time: "04:00:00"
    site_cluster:
      cpus: 16
      mem: "32G"
      time: "06:00:00"
```

The run's stages are `prepare`, `structure` (one task per structure) and
`sites`; `qpocket slurm generate --site site_01` writes that site's
`site_featurize` (one task per cloud) and `site_cluster`. The old stage keys
`apbs`, `featurize`, `collect`, `geodesic` and `distance`, and
`split_apbs_stage`, are ignored with a warning; move their values to the new keys.

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
