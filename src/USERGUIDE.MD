# User Guide (Python API)

The command line (`qpocket run`, `qpocket screen …`) is documented in `docs/`.
This page shows the same pipeline from Python.

## A run

```python
from qpocket.config import load_qpocket_config
from qpocket.pipelines.run import run

cfg = load_qpocket_config("my_target.yaml", overrides=["compute.workers=8"])
run_dir = run(cfg)
```

`run` ingests the structures (`pdb_query` searches RCSB; `fragment_screen` copies
local files), builds the reference once, takes every structure through the same
per-structure chain (align, clean, find every site ligand, protonate, read contacts,
detect cavities), groups the ligands into binding sites and cuts one point cloud per
site per structure. It names no reference ligand and computes no features or
distances; those are per-site steps (`qpocket site …`, or the explorer).

## Pockets without a run

```python
import qpocket as qp

points, features = qp.featurize(qp.PocketSpec("7KG6", "A", "WDG"))
result = qp.compare([qp.PocketSpec("7KG6", "A", "WDG"), qp.PocketSpec("4B1H", "A", "AR6")])
```

## Extending it

Each choice below is a function registered by name in `qpocket.core.ports` and
picked in config. Adding one is one new module under `qpocket/plugins/` plus a
config value:

| Registry | Config | Built-ins |
|---|---|---|
| `SOURCES` | `mode` | `pdb_query`, `local` |
| `CHARGES` | `charges.use` | `pdb2pqr` |
| `FEATURES` | `features.enabled` | `apbs`, `cp_lp`, `hbond`, `aromatic`, `template` |
| `CONTACTS` | `contacts.use` | `prolif` |
| `CLUSTERING` | `distance.clustering.*.method` | `hierarchical`, `kmedoids`, `affinity` |
| `DECOYS` | screen `decoy.strategy` | `lude` |
| `DOCKING` | screen `docking.engine` | `vina`, `quickvina` |

A plug-in declares its parameters (a frozen dataclass, filled from
`<section>.options.<name>`), the executables it runs (`qpocket doctor` checks
them) and the resources it needs (SLURM job sizes follow them).
