# Fragment Screening

Fragment-screening mode analyses structures you already have on disk — a soaking
campaign, a set of in-house models — instead of querying the PDB. You supply the
structure files, the reference structure, and the ligand that marks the pocket in
that reference.

## Configuration

```yaml
mode: fragment_screen

fragment_screen:
  input_type: "paths"                       # or "csv"
  reference_path: "/data/frag/reference.pdb"
  structure_paths:
    - "/data/frag/soak_A07.pdb"
    - "/data/frag/soak_B11.pdb"
    - "/data/frag/soak_C03.pdb"

reference:
  ligand_id: "LIG"                          # ligand marking the pocket in the reference
  chain_id: "A"
  pocket_definition: "Interactions"

paths:
  workdir: "/data/frag/out"
```

```bash
qpocket run --config campaign.yaml
```

`.pdb`, `.cif` and `.ent` inputs are accepted.

!!! important "The reference must still contain its ligand"
    The pocket is located from the reference ligand's centroid, so point
    `reference_path` at a structure that still has the ligand in it. A *cleaned*
    structure has had its ligands stripped and will fail with
    `ligand 'LIG' not found`.

### The reference

`reference_path` names the structure every other one is measured against. It
defines the pocket bounding box that constrains cavity selection for the whole
campaign, and it names the run.

It does not have to appear in `structure_paths`; if it does, it is still ingested
only once. When `reference_path` is omitted, `reference_index` selects one from
the ingested structures instead.

### Run naming

A local file has no PDB accession, so `reference.pdb_id` is optional here. The run
directory is named after the reference filename:

```
/data/frag/out/reference/{metadata,pockets,structures}
```

Set `reference.pdb_id` to name it something else.

### Structure identifiers

Each structure is identified by its filename, so `soak_A07.pdb` appears as
`soak_A07` in `master.csv`, `clustering_metadata.csv` and `pockets/`. Two files
with the same basename in different directories get a numeric suffix.

`metadata/inputs.csv` records where every structure came from:

| pdb_id | source_path | is_reference |
|---|---|---|
| `reference` | `/data/frag/reference.pdb` | `True` |
| `soak_A07` | `/data/frag/soak_A07.pdb` | `False` |

### Residue renumbering

`alignment.renumber_residues` defaults to *auto*, which is **off** in
fragment-screening mode: renumbering maps residues onto canonical UniProt
positions, and local or modelled structures generally have no UniProt entry. Set
it to `true` with `alignment.manual_uniprot_id` if your structures are real PDB
depositions and you want canonical numbering.

## CSV input

For a large campaign, list the paths in a CSV with a `pdb_path` column:

```csv
pdb_path
/data/frag/soak_A07.pdb
/data/frag/soak_B11.pdb
```

```yaml
fragment_screen:
  input_type: "csv"
  csv_path: "/data/frag/campaign.csv"
  reference_path: "/data/frag/reference.pdb"
```

## Command line

Every setting above has a flag, so a campaign can be run without a config file:

```bash
qpocket run \
  --mode fragment_screen \
  --input-type paths \
  --reference-path /data/frag/reference.pdb \
  --structure-paths /data/frag/soak_A07.pdb \
  --structure-paths /data/frag/soak_B11.pdb \
  --ligand-id LIG \
  --workdir /data/frag/out
```

`--structure-paths` is repeated once per file; a space-separated list will not
work.

## On a cluster

Fragment mode uses the same SLURM path as everything else. Run `prepare` first so
the array stages know how many structures there are, then generate:

```bash
qpocket task prepare --config campaign.yaml
qpocket slurm generate --config campaign.yaml --submit
qpocket slurm status --config campaign.yaml
```

See [Running on a cluster](hpc.md).
