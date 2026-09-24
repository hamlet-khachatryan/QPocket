# Interactions

Q-Pocket detects protein–ligand interactions with
[ProLIF](https://prolif.readthedocs.io), on structures whose chemistry comes from
the PDB's chemical component dictionary rather than from their coordinates.

## How a structure is perceived

A PDB file records where atoms are, not what they are bonded to. Inferring bond
orders from distances is unreliable on real structures: a peptide bond at 1.13 Å
reads as a triple bond, and proximity bonding invents a fifth bond on a carbonyl
carbon. Both happen in deposited entries.

Q-Pocket instead looks each residue and ligand up by its **chemical-component
ID** and takes the bonds from that definition:

| step | source | rule |
|---|---|---|
| heavy-atom bonds | CCD `_chem_comp_bond` | matched by atom name; `value_order` is the Kekulé order |
| hydrogens | geometry | each bonds to its nearest heavy atom with free valence, within 1.35 Å |
| links between residues | geometry | peptide `C–N` under 2.0 Å, disulfide `SG–SG` under 2.5 Å |

Aromaticity is perceived by RDKit from those Kekulé orders, not copied from the
CCD's aromatic flag — copying it produces rings that fail to kekulise.

Definitions download from RCSB on first use and cache under
`~/.cache/qpocket/ccd`, or wherever `QPOCKET_CCD_CACHE` points. The twenty
standard residues, water and the common ions ship with the package, so a run on
an apo protein needs no network.

## The docking-ready structure

Every structure gets one protonated artefact, written before its interactions
are read:

```
<run>/structures/protonated/<PDB_ID>.pdb
```

It holds the protein with pdb2pqr hydrogens, the ligand with RDKit hydrogens,
and `CONECT` records for the ligand. Three details matter:

- **Ligand protonation follows pKa, not the drawing.** CCD entries draw ligands
  neutral. Benzamidine's amidine is cationic at pH 7.4, and without that its
  salt bridge to Asp189 is reported as a plain hydrogen bond. `dimorphite-dl`
  assigns the dominant state, at non-aromatic sites only: charging an aromatic
  N–H can cost the ring its aromaticity and with it every π-stacking hit.
- **Hydroxyl hydrogens are re-aimed.** pdb2pqr protonates the protein before the
  ligand exists, so Ser, Thr and Tyr hydroxyls point wherever the protein alone
  suggested. In streptavidin that loses Ser27, Tyr43 and Ser88 at textbook
  distances. Their rotamers are re-chosen against the assembled complex.
- **An apo structure still gets its file**, holding the protonated protein, and
  the run continues with `no_ligand_found`.

## What the stored strings mean

Interactions are stored as `RESNAME_RESNUM_CHAIN|InteractionType`, for example
`TYR_795_A|Pi-stacking (P)`.

ProLIF names an interaction from the **ligand's** side; these strings name it
from the **protein's**, so the directional types are inverted:

| ProLIF | stored as |
|---|---|
| `HBDonor` | `H-bond Acceptor` |
| `HBAcceptor` | `H-bond Donor` |
| `Cationic` | `Saltbridge (-)` |
| `Anionic` | `Saltbridge (+)` |
| `CationPi` | `Pi-Cation (Pi)` |
| `PiCation` | `Pi-Cation (Cat)` |
| `FaceToFace` | `Pi-stacking (P)` |
| `EdgeToFace` | `Pi-stacking (T)` |
| `Hydrophobic` | `Hydrophobic` |
| `XBDonor`, `XBAcceptor` | `Halogen Bond` |
| `MetalDonor`, `MetalAcceptor` | `Metal Complex` |

A structure with no ligand records `no_ligand_found`; a missing file records
`file_not_found`; a structure whose chemistry could not be perceived records
`analysis_error` and names the component in the log.

## Choosing what is detected

See [`interactions`](../configuration.md) for the configuration block. The
defaults are ProLIF's own, which are stricter than the cutoffs PLIP published —
hydrogen bonds at 3.5 Å and 130–180° against 4.1 Å and 100°. To reproduce the
looser ones:

```yaml
interactions:
  parameters:
    HBDonor: {distance: 4.1, DHA_angle: [100, 180]}
    HBAcceptor: {distance: 4.1, DHA_angle: [100, 180]}
    Hydrophobic: {distance: 4.0}
```

This matters beyond the interaction tables. When
`reference.pocket_definition` is `Interactions`, each pocket is cropped to its
observed contacts and re-featurised, so the thresholds shape the point cloud,
the distance matrices and the clustering.

## Determinism

Given the same input, detection is reproducible: three consecutive runs produce
byte-identical interaction strings. The previous detector was not — it protonated
ligands with OpenBabel, whose hydrogen placement moved 290 of 4350 hydrogens
between two identical calls.
