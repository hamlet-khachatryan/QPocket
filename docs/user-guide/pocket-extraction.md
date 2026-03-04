# Pocket Extraction

## Two-phase bounding-box strategy

### Phase 1 — Reference structure

1. **Cavity detection** (CavFind / LigSite) identifies all solvent-accessible cavities.
2. The cavity whose point cloud overlaps the **reference ligand centroid** (majority overlap: > 50 % of centroid neighbours inside) is selected.
3. An axis-aligned **bounding box** is computed from this point cloud with **1 Å padding** on every face.

### Phase 2 — All other structures

1. Cavity detection runs for each structure.
2. Each cavity is scored by **volumetric overlap** — the count of its grid points that fall inside the reference bounding box.
3. The cavity with the **highest volumetric overlap** is selected as the binding pocket.
4. If no cavity overlaps at all, the structure is marked `CAVITY_CLOSED`.

## Pocket definition modes

| Mode | Description |
|---|---|
| `Solvent` | The full cavity point cloud is used as the pocket representation. |
| `Interactions` | The pocket is cropped to a sphere of radius `interaction_padding` around PLIP-observed interacting residues. |

## Bounding box mode

When `pocket_mode: "bounding_box"`, an explicit box is used instead of CavFind:

```yaml
reference:
  pocket_mode: "bounding_box"
  bounding_box:
    center: [10.0, 20.0, 30.0]
    dimensions: [15.0, 15.0, 15.0]
```

## Ligand selection

After pocket extraction, all heteroatom small molecules whose atoms overlap with the pocket (majority threshold) are collected.  The **largest** ligand (highest heavy-atom count) is returned and saved to `structures/extracted_ligands/`.

Molecules in `cleaning.allowed_cofactors` and the built-in `HET_TO_REMOVE` blacklist are excluded.