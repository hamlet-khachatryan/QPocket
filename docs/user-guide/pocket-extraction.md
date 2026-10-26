# Pocket extraction

QPocket does not ask which ligand defines the pocket. `qpocket run` builds binding
sites from the cavities CavFind finds in every structure, assigns the ligands to
them, and gives each structure one point cloud per site: its own CavFind cavity,
never cut or cropped.

## Site ligands

A ligand is a non-polymer, non-water heteroatom group that

- is not in `cleaning.allowed_cofactors` (allowed cofactors stay in the protein),
- is not a buffer or cryoprotectant (`qpocket.constants.BUFFER_RESNAMES`: glycerol,
  PEGs, DMSO, HEPES, Tris, citrate, …) unless named in `sites.keep_ligands`,
- has at least one carbon atom, so ions, sulfate and phosphate are never ligands, and
- has at least `sites.min_heavy_atoms` heavy atoms.

## What a run does

1. Align and clean every structure, find its ligands, protonate one complex per
   structure and read each ligand's ProLIF contacts.
2. Detect cavities with CavFind once per structure.
3. **Candidates**: cavities with at least `sites.min_cavity_points` points (50).
   Each gets a box: its points' extent plus half the grid spacing per face.
4. **Group**: candidates are grouped so that every pair in a group has box IoU
   (intersection over union) ≥ `sites.cavity_overlap` (0.5); groups whose median
   boxes overlap by the same IoU merge into one site.
5. **Membership**: a cavity belongs to every site whose box it overlaps by ≥
   `sites.cavity_overlap`, so one cavity can sit in two neighbouring sites.
6. **Ligands**: a ligand belongs to every site whose box, padded by
   `sites.box_padding` (0 Å), holds ≥ `sites.ligand_fraction` (0.5) of its heavy atoms.
7. **Keep**: a site survives when its cavities come from ≥
   `sites.min_structure_fraction` (5 %) of the structures, or when it holds any ligand.
8. **Name**: sites are ranked by the number of structures with a cavity there and
   named after the residue nearest the centre: `S1-H21`, `S2-R115`, …
9. For each site and structure with a member cavity, write the cloud (the
   structure's member cavities' points, as CavFind gave them) and cache its
   geodesics; save `metadata/config.yaml`.

It stops there: no features and no distance matrices. Those are per-site steps, see
[Distance analysis](distance-analysis.md).

## Holo, apo and CavFind misses

| A structure at a site has | It is | It gets |
|---|---|---|
| a member cavity and a ligand | `BOUND_STATE` (holo) | a cloud and every per-site step |
| a member cavity, no ligand | `UNBOUND_STATE` (apo) | a cloud and every per-site step |
| a ligand, no member cavity | a CavFind miss | a row in `sites/cavfind_misses.csv`, nothing else |
| neither | not listed | — |

A ligand is a miss only when its structure has no cavity in any of the sites the
ligand lies in (`no_cavity_at_site`), or when it lies in no site at all (`no_site`,
typical for surface fragments). Misses never reach featurisation, clustering or
cross-docking.

`reference.pocket_definition` and `reference.interaction_padding` do not affect
site clouds.

## Regrouping

Re-site with other parameters without recomputing the structures. A site whose
member cavities are unchanged keeps its analysis:

```bash
qpocket site regroup runs/7KG6 --cavity-overlap 0.4 --ligand-fraction 0.5
```

`--box-padding`, `--min-cavity-points` and `--min-structure-fraction` are
accepted too. The explorer has the same action (**Regroup sites**).

## Settings

```yaml
sites:
  min_heavy_atoms: 6
  keep_ligands: []
  min_cavity_points: 50
  cavity_overlap: 0.5
  min_structure_fraction: 0.05
  ligand_fraction: 0.5
  box_padding: 0.0
```

`sites.link_cutoff` (ligand-linked sites) is obsolete and dropped with a note.
`reference.ligand_id`, `ligand_chain_id`, `pocket_mode` and `bounding_box` are
obsolete too: they are ignored with a warning.
