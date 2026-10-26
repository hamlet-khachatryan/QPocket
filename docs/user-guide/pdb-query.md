# PDB Query Mode

In **pdb_query** mode QPocket queries the RCSB PDB REST API for structures that are homologous to a user-specified reference chain and downloads them automatically.

## Minimal example

```bash
qpocket run --pdb-id 6HP0 --chain-id A --workdir ./results
```

## Filtering options

| Flag | Config key | Default | Description |
|---|---|---|---|
| `--sequence-identity` | `pdb_query.sequence_identity` | `1.0` | Minimum sequence identity (0–1] |
| `--resolution-limit` | `pdb_query.resolution_limit` | `2.5` | Maximum resolution (Å) |
| `--min-length` | `pdb_query.min_length` | `20` | Minimum chain length |
| `--max-length` | `pdb_query.max_length` | `null` | Maximum chain length |
| `--experimental-method` | `pdb_query.experimental_method` | `null` | e.g. `X-RAY DIFFRACTION` |
| `--enforce-same-uniprot` | `pdb_query.enforce_same_uniprot` | `false` | Only same UniProt entry |

## How it works

1. The FASTA sequence of `reference.pdb_id:reference.chain_id` is fetched from RCSB.
2. A sequence-similarity search is executed with the supplied filters.
3. The reference PDB ID is added to the hit list if not already present.
4. All structures are downloaded as `.cif` files to `structures/original/`.

The reference is the structure the others are aligned to. It does not choose the
pocket: sites are built from the cavities of every hit, see
[Pocket extraction](pocket-extraction.md). Analyse a site with `qpocket explore` or
`qpocket site …`.