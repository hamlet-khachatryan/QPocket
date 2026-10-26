# Pocket extraction

QPocket does not ask which ligand defines the pocket. `qpocket run` finds every
ligand-occupied binding site in the structures and cuts one point cloud per site
per structure.

## Site ligands

A ligand is a non-polymer, non-water heteroatom group that

- is not in `cleaning.allowed_cofactors` (allowed cofactors stay in the protein and never define a site),
- has at least one carbon atom, so ions, sulfate and phosphate never define a site, and
- has at least `sites.min_heavy_atoms` heavy atoms.

There is no built-in list of names. To keep a cofactor out of the site list, add
it to `cleaning.allowed_cofactors`.

## What a run does

1. Align and clean every structure, and find its site ligands.
2. Protonate one complex per structure and read the ProLIF contacts of each ligand.
3. Detect cavities (CavFind / LigSite) once per structure.
4. **Group the ligands into sites**: ligands from different structures whose atoms
   come within `sites.link_cutoff` (4.0 Å) of one another belong to the same site.
5. For each site and structure, cut one cloud from the cavity points inside a box
   around the site's ligands (`sites.box_padding`, 4.0 Å).
6. Cache the geodesics, and save `metadata/config.yaml`.

It stops there: no features and no distance matrices. Those are per-site steps, see
[Distance analysis](distance-analysis.md). A structure with no ligand at a site is
still listed at that site, with the cloud cut from the same box.

## Pocket definition modes

| Mode | Description |
|---|---|
| `Solvent` | The full cavity point cloud is used as the pocket representation. |
| `Interactions` | The pocket is cropped to a sphere of radius `interaction_padding` around the contacting residues. |

## Regrouping

If the sites are too coarse or too fine, regroup at another cutoff without
recomputing the structures. Unchanged sites keep their analysis:

```bash
qpocket site regroup runs/7KG6 --cutoff 5.0
```

The explorer has the same action (**Regroup sites**).

## Settings

```yaml
sites:
  min_heavy_atoms: 6
  link_cutoff: 4.0
  box_padding: 4.0
```

`reference.ligand_id`, `ligand_chain_id`, `pocket_mode` and `bounding_box` are
obsolete: they are ignored with a warning. The old `--ligand-id`, `--pocket-mode`
and `--bbox-*` options are gone.
