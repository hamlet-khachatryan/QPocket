# Using Q-Pocket as a library

Every step of the pipeline is also a plain function. Pure computations live in
`qpocket.core`, file and tool access in `qpocket.io`, and each pipeline is a short
orchestrator in `qpocket.pipelines`.

## Compare two pockets without running a pipeline

```python
import qpocket as qp

result = qp.compare([
    qp.PocketSpec("7KG6", ligand="WDG"),                          # PDB ID, downloaded
    qp.PocketSpec("my_model.pdb", chain="A", residues=("TYR795", "GLU756")),
])
print(result.to_frame())            # combined distance; to_frame("shape") / ("chem")
```

The same from the shell:

```bash
qpocket compare-structures 7KG6:WDG my_model.pdb@A:TYR795,GLU756
qpocket featurize 7KG6 --ligand WDG -o a.npz     # or featurize once, compare many times
qpocket compare a.npz b.npz c.npz
```

## Compare on a finished run's normalisation

Features are standardised before fingerprinting, and the scaling is fitted on the
pockets being compared. To put new pockets on the same footing as a run, reuse the
scaler every run saves in `metadata/scaler.npz`:

```python
scaler = qp.load_scaler("runs/7KG6/metadata/scaler.npz")
result = qp.compare(specs, scaler=scaler)
```

```bash
qpocket compare-structures 7KG6:WDG 1HSG:MK1 --scaler runs/7KG6
```

## The pieces

| Step | Function |
|---|---|
| pocket points and features | `qp.featurize(spec)` |
| geodesic graph | `qp.geodesic_matrix(points, radius)` |
| feature scaling | `qp.fit_scaler([features, ...])`, `qp.load_scaler(path)` |
| fingerprint | `qp.fingerprint(geodesic, features, scaler, global_k, sigma)` |
| distance | `qp.pocket_distance(fp_a, fp_b)` |
| many pockets at once | `qp.compare_features([(points, features), ...], ids)` |

## Pipelines

| Pipeline | Orchestrator | CLI |
|---|---|---|
| PDB query | `qpocket.pipelines.pdb_query.run(cfg)` | `qpocket run --config ...` |
| Fragment screen | `qpocket.pipelines.fragment_screen.run(cfg)` | `qpocket run --config ...` |
| Pocket comparison | `qpocket.pipelines.compare.run(specs)` | `qpocket compare-structures ...` |
| Virtual screening | `qpocket.pipelines.screening.prepare(run_dir, submit=...)` | `qpocket screen prepare --submit slurm\|local` |

::: qpocket.pipelines.compare

::: qpocket.core.spectral
