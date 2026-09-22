# Performance

A run is mostly per-structure work, and Q-Pocket spreads it across cores.

## Measured

The 7-structure PARG run (`scripts/benchmark_parallel.py`) on a 14-core laptop,
default feature set:

| `compute.workers` | seconds | speedup |
|---|---|---|
| 1 (serial) | 76 | 1.0x |
| 2 | 47 | 1.6x |
| 4 | 37 | 2.1x |
| 7 (the default: half the cores) | 27 | 2.8x |

Outputs are byte-identical at every setting; `scripts/golden_diff.py` between the
1-worker and 4-worker runs reports a maximum delta of 0 and identical clusters.

Reproduce it with:

```bash
uv run python scripts/benchmark_parallel.py /tmp/bench --counts 1,2,4,0
```

## What runs in parallel

Each structure goes from input file to geodesic graph on one worker: align, clean,
identify the pocket, featurise (pdb2pqr, APBS, CavFind, PLIP), then the geodesic
graph. Three things are shared, so they stay serial:

| Step | Why it cannot fan out |
|---|---|
| the reference structure | its pocket defines the box every other structure is matched against, so it runs first, alone |
| `collect` | master.csv, the interaction archives, the frequency tables and `actives.smi` need every structure |
| the scaler, distances, clustering | feature standardisation is fitted across all pockets, and the matrices are global |

Fingerprints fan out again once the scaler exists.

## Choosing a worker count

```yaml
compute:
  workers: 0          # 0 = half the cores, -1 = all, N = exactly N
  backend: "process"  # "process" (separate interpreters) or "thread"
```

or `qpocket run --workers 8`.

- **`0` (default, half the cores)** leaves room for everything else on the machine.
  Most of the available speedup is already here.
- **`-1`** is worth it on an idle machine. Returns taper off, because `collect`,
  the distance matrices and clustering stay serial (Amdahl).
- **`1`** is the reference behaviour: a plain loop, no joblib, deterministic logs
  in one file. Use it when debugging.

**Memory is the real limit.** Every worker can run APBS on its own grid, so peak
memory scales with `workers`. A large protein at `-1` on a many-core node can
exhaust RAM; lower `workers` if that happens.

`backend: "thread"` shares memory and avoids process startup, but Python's GIL
limits it to work that waits on external tools. Prefer the default `process`.

## Logs

Workers do not inherit the parent's log destination, so each structure writes its
own file:

```
<run>/logs/structure/<PDB_ID>.log
```

The parent log keeps the run-level story, with one `parallel_map` entry per stage
recording items, workers and seconds.

## On a cluster

`compute.workers` is for one machine. To spread a run across nodes use
`qpocket slurm generate`, which keeps one structure per array task; see
[Running on a cluster](hpc.md).
