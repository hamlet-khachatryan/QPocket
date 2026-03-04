# Fragment Screening Mode

Use **fragment_screen** mode when you already have a local set of PDB structures to analyse.

## CSV input

Create a CSV file with a `pdb_path` column:

```csv
pdb_path
/data/structures/frag_001.pdb
/data/structures/frag_002.pdb
```

```bash
qpocket run \
  --mode fragment_screen \
  --csv-path structures.csv \
  --workdir ./screen_results \
  --reference-index 0
```

## Direct path list

```bash
qpocket run \
  --mode fragment_screen \
  --structure-paths /data/frag_001.pdb /data/frag_002.pdb \
  --workdir ./screen_results
```

## Reference index

`--reference-index` (default `0`) selects which structure in the list is used as the reference for bounding-box computation.  The reference structure's pocket is found via its ligand centroid (requires `--ligand-id` or `reference.ligand_id` in the config).