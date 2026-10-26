# QPocket

Q-Pocket: A Framework for Analysis and Benchmarking of the Experimentally Observed Conformational Space of Protein Binding Pockets for Drug Design Applications

## Install and run

```bash
uv sync --extra viz                                  # or: python3 -m pip install --user uv
uv run qpocket doctor                                # can this machine run the pipeline?
uv run qpocket run --config experiments/PARG.yaml    # a full run
```

`qpocket run` aligns and cleans the structures, finds every ligand-occupied binding
site, and cuts one pocket point cloud per site per structure. You do not name a
reference ligand. It stops there; analyse a site next:

```bash
uv run qpocket explore RUNS                          # pick a site, press the step buttons
uv run qpocket site list RUNS/7KG6
uv run qpocket site featurize RUNS/7KG6 --site site_01
uv run qpocket site cluster   RUNS/7KG6 --site site_01
```

A run uses half the cores by default; `--workers N` changes that, `--workers 1`
is serial. `--features hbond,aromatic` (or `--no-apbs`) narrows a site's featurisation,
which also skips the tools those providers would have needed.

## Comparing pockets without a pipeline

```bash
uv run qpocket compare-structures 7KG6:WDG 1HSG:MK1
```

```python
import qpocket as qp

result = qp.compare([qp.PocketSpec("7KG6", ligand="WDG"),
                     qp.PocketSpec("1HSG", ligand="MK1")])
print(result.to_frame())
```

## Documentation

- [Installation](docs/installation.md), [Quick start](docs/quickstart.md), [Configuration](docs/configuration.md)
- [Features](docs/user-guide/features.md) — providers, and how to add one
- [Performance](docs/user-guide/performance.md) — what runs in parallel
- [Library use](docs/api/library.md) — the public API
- [RUNBOOK.md](RUNBOOK.md) — deploying to a SLURM cluster, submitting runs, and viewing results in the notebook
