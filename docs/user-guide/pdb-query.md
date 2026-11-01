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
## Targets with two or more chains

By default every structure keeps **one chain**: the one closest in sequence to
`reference.chain_id`; the others are dropped. That is right when the pocket lies
within one chain, even if the crystal holds several copies (each copy's ligand
touches only its own chain).

When the pocket is formed by two or more chains (an interface, a dimer site), name
the partner chains of the reference:

```yaml
reference:
  pdb_id: "XXXX"
  chain_id: "A"                # primary: the PDB search and the superposition use it
  auxiliary_chain_ids: ["B"]   # partners lining the pocket, e.g. ["B", "C"]
alignment:
  min_auxiliary_identity: 0.70             # how similar a partner must be in other entries
  max_auxiliary_centroid_deviation: 15.0   # Å; a farther match is logged as suspicious
  require_all_chains: true                 # skip entries that lack a partner
  auxiliary_uniprot_ids: {B: "P00533"}     # only for a different protein, to renumber it
```

```bash
qpocket validate my_target.yaml
qpocket run --config my_target.yaml
```

In each entry the primary chain must match at ≥ 90 % identity; every chain is
moved with it, then each partner is matched by sequence, and among equally good
copies (a homodimer) the one whose Cα centre sits where the reference partner sits.
Only the primary and matched partners are kept; cavities, sites, contacts and
docking receptors use the complex. The run log names each match
(`auxiliary_chain_matched`) and each skipped entry (`auxiliary_chain_not_found`).

Chain IDs refer to the reference file; other entries may use other letters. The PDB
search uses the primary chain's sequence, so entries holding only that chain are
found and then skipped (unless `require_all_chains: false`, which keeps them as single
chains with a different pocket). Ligands bound only to a chain you did not name are
lost with that chain.
