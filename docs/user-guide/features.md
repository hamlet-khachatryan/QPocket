# Features

Each column of `featurized.csv` comes from a named **provider**. Which providers
run is a config setting, so featurisation can be narrowed, or extended with a new
tool, without touching the pipeline.

## The built-in providers

| Provider | Columns | Needs |
|---|---|---|
| `apbs` | `apbs` | pdb2pqr and APBS |
| `cp_lp` | `cp`, `lp` | pdb2pqr |
| `hbond` | `h_donor`, `h_acceptor` | the cleaned structure |
| `aromatic` | `aromatic` | the cleaned structure |

All four are on by default, which reproduces the six historic columns in their
historic order.

## Choosing providers

```yaml
features:
  enabled: ["apbs", "cp_lp", "hbond", "aromatic"]
  options: {}
```

or from the command line:

```bash
qpocket run --config my.yaml --no-apbs                  # drop one
qpocket run --config my.yaml --features hbond,aromatic  # geometry only
qpocket featurize 7KG6 --ligand WDG --features hbond,aromatic -o pocket.npz
```

Two things follow automatically:

- **Columns.** `featurized.csv` carries exactly the enabled providers' columns.
- **Tools.** A provider declares what it needs, so a set with no `pqr` requirement
  skips pdb2pqr and APBS entirely. On the 7-structure PARG run, `hbond,aromatic`
  takes 13 s against 27 s for the full set.

## What a run records

`metadata/features.json` names the set a run used:

```json
{"enabled": ["hbond", "aromatic"], "columns": ["h_donor", "h_acceptor", "aromatic"]}
```

Fingerprints built from different feature sets are not comparable — the spectra
have different lengths and meanings. `qpocket compare` refuses pockets whose
`feature_names` disagree.

## Adding a provider

Copy `src/qpocket/core/providers/template.py`, which is registered but off by
default:

```python
class ConstantProvider:
    name = "template"
    columns = ("template",)
    requires = ()          # "pqr", "structure", or nothing

    def compute(self, points, ctx):
        value = float(ctx.options.get("value", 1.0))
        return {"template": np.full(len(points), value, dtype=float)}


register(ConstantProvider())
```

Then enable it:

```yaml
features:
  enabled: ["hbond", "aromatic", "template"]
  options:
    template: {value: 2.5}
```

`ctx` carries the cleaned PDB path, the PQR path, the parsed atoms
(`StructureData`) and your provider's options. Keep `compute` pure: no logging and
no files. Anything that runs an external tool belongs in `qpocket.io`, with the
result handed to the provider through `ctx.options` — that is how `apbs` works.

New columns land after the built-in ones, in provider order.
