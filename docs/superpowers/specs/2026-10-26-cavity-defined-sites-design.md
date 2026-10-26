# Cavity-defined binding sites

Date: 2026-10-26
Status: approved 2026-10-26

## Why

Sites are currently defined by ligands: every ligand of every structure is
pooled, ligands with heavy atoms within `sites.link_cutoff` (4 Å) are joined
transitively, an axis-aligned box padded by `sites.box_padding` is drawn
around the union of a site's ligands, and each structure's cloud at the site is
every CavFind point of **any** cavity inside that box (optionally cropped to
contact residues in `Interactions` mode).

On the OpenBind EV-A71 2A rerun (`~/qpocket_runs/EV71_2A_prep`, 884
structures) this produced:

- **site_01: 861 of 885 ligands, ~39 Å across.** Its box (45×25×27 Å) holds 49 %
  of the protein's atoms (protein ~40×34×38 Å).
- **Clouds merge cavities**: a median of 4 of a structure's ~6 CavFind cavities;
  only 65 % of cloud points lie within 4 Å of the structure's own fragment.
- **Surface fragments have no cloud** (site_02: 0 of 14 fragments within 2 Å of
  a CavFind point; clouds of 0–2 points; holo structures labelled
  `CAVITY_CLOSED`).

## Decisions (the user's)

1. **Sites are defined by CavFind cavities, not by ligands.** A cavity is used
   exactly as CavFind outputs it — its points are never cut, cropped or edited.
2. **Cavities below a configurable minimum size never form or join a site.**
3. **Cavity overlap = bounding-box volume overlap** above a configurable
   threshold (40–50 %).
4. **A cavity can belong to more than one site** (A overlaps B, B overlaps C,
   A and C do not → two sites {A, B} and {B, C}); sites must not chain A to C.
5. **Several cavities of one structure in the same site are pooled** as that
   structure's cloud at the site.
6. **Ligands are assigned to sites by box containment**: a ligand belongs to a
   site when at least a configurable fraction (40–50 %) of its heavy atoms lies
   inside the site box plus a configurable padding.
7. **A site is kept when it has at least a minimum share of structures or at
   least one ligand in any structure.**
8. **CavFind misses** are reported and excluded from all downstream analysis:
   a structure whose ligand lies in **no site where that structure has a
   cavity**, and a ligand that lies in no site at all.
9. **Site names `S<rank>-<residue>`** (e.g. `S1-H21`): rank by number of
   structures with a cavity in the site; residue nearest the site centre.
10. **The explorer shows a limited number of ligands (20–30) at once**, with a
    control for how many.

## The algorithm

Parameters (all under `sites.`, configurable, replacing `sites.link_cutoff`):

| key | default | meaning |
|---|---|---|
| `min_cavity_points` | 50 | smaller cavities are recorded but never site members |
| `cavity_overlap` | 0.5 | IoU threshold for cavity–cavity and site–site overlap |
| `min_structure_fraction` | 0.05 | share of structures a ligand-free site needs |
| `ligand_fraction` | 0.5 | share of a ligand's heavy atoms inside the site box |
| `box_padding` | 0.0 | Å added to every face of the site box for ligand assignment |
| `min_heavy_atoms`, `keep_ligands` | unchanged | which HET groups are ligands |

Steps:

1. **Cavities.** Read every structure's `structures/cavities/<pdb_id>.csv`
   (x, y, z, cavity). Each `(pdb_id, cavity)` with ≥ `min_cavity_points` points
   is a candidate. Its **box** is the points' extent widened by half the grid
   spacing on every face (*my call*: each point is a voxel, so flat cavities
   get a real volume; the points themselves are unchanged). Grid spacing is
   measured from the data (median nearest-neighbour distance), not assumed.
2. **Overlap.** IoU of two boxes = intersection volume / union volume
   (*my call*: not intersection over the smaller box, which turns one large
   cavity into a hub). Only pairs whose boxes intersect are computed.
3. **Cores.** Complete-linkage hierarchical clustering of candidates on
   `1 − IoU`, cut at `1 − cavity_overlap`: inside a core every pair overlaps by
   ≥ the threshold, so no chain forms.
4. **Site boxes.** A site's box is the element-wise median of its members'
   boxes (lower and upper corners separately).
5. **Site merge** (*my call*, confirmed by the user). Repeatedly merge the two
   sites whose site boxes have the highest IoU, recomputing the merged median
   box, while that IoU ≥ `cavity_overlap`. One pocket that drifts between
   structures splits into several cores at step 3; this reunites them. It
   compares whole-site boxes, so a cavity still cannot chain A to C.
   Spike on EV71_2A_prep (min 50 points): 38 cores → 12 sites; the main pocket
   becomes one site of 828 structures, box 13×8×9 Å.
6. **Membership.** A candidate belongs to every site whose box it overlaps by
   IoU ≥ `cavity_overlap` (its own site included). This is what lets a cavity be
   in two sites.
7. **Ligands.** A ligand belongs to every site whose box, padded by
   `box_padding`, holds ≥ `ligand_fraction` of its heavy atoms. Spike: padding
   4 Å assigns nearly every main-pocket fragment to several small neighbouring
   sites as well; 0 Å keeps the main site at 846 and the neighbours small, so
   0 Å is the default (*my call*, confirmed).
8. **Keep.** A site survives when its member cavities come from
   ≥ `min_structure_fraction` of the run's structures, or when any ligand is
   assigned to it.
9. **Per structure per site:**
   - `BOUND_STATE`: ≥ 1 member cavity and ≥ 1 assigned ligand (holo);
   - `UNBOUND_STATE`: member cavity, no ligand (apo);
   - neither: not listed for that site.
   The cloud (`binding_site.csv`) is the union of the structure's member
   cavities' points, unmodified. Only these two states get clouds, geodesics
   and downstream steps. `CAVITY_CLOSED` is no longer produced.
10. **CavFind misses** (`sites/cavfind_misses.csv`: `pdb_id, ligand_key,
    resname, sites, reason`): a ligand assigned to ≥ 1 site where its structure
    has no member cavity at any of them → reason `no_cavity_at_site`; a ligand
    assigned to no surviving site → reason `no_site`. A ligand that is holo in
    at least one site is not a miss. Misses never get clouds.
11. **Names and order.** Sites sorted by number of structures (desc), ties by
    box centre; `site_id = S<rank>-<X><num>` where `X<num>` is the one-letter
    code and number of the protein residue nearest the site centre in the
    reference structure (fallback: the first holo/apo structure). Rank makes
    names unique. `site_id` is the folder name.
12. **Site table** `sites/sites.csv`: `site_id, n_structures, n_holo, n_apo,
    n_cavfind_miss, n_ligands, n_cavities, centre_x/y/z, box_lo_x/y/z,
    box_hi_x/y/z, resnames, lining_residues, contact_residues` (lining residues:
    residues with an atom within 4 Å of the reference structure's member cavity
    points, or the first structure's when the reference has none).

## Around the algorithm (my calls)

- **Pocket definition modes**: site clouds are never cropped, so
  `reference.pocket_definition: Interactions` and `interaction_padding` no
  longer affect sites; keys stay valid, docs say so.
- **Regroup**: `qpocket site regroup RUN_DIR` takes the new parameters as
  options (each optional, saved to the run config), replacing `--cutoff`; the
  explorer's regroup control follows. "Unchanged sites keep their results" is
  keyed on a site's member-cavity set.
- **Config migration**: an old config's `sites.link_cutoff` is accepted and
  dropped with a logged note (existing migration machinery).
- **Explorer**: the site overview 3D view shows at most N ligands (default 25,
  number control 1–200), chosen deterministically (one per holo structure,
  largest first, in master order); the site list shows the new columns.

## Goals / non-goals / anti-goals

- Goals: no site spans half the protein; a structure's cloud at a site is its
  own CavFind cavity (or cavities) for that pocket; every ligand is holo in a
  site or listed as a miss; short names that say where the pocket is.
- Non-goals: changing CavFind or its parameters; analysing surface fragments no
  cavity covers; the zinc-cofactor protonation bug; migrating ligand-based
  sites of old runs (they are re-sited by regroup or rerun).
- Anti-goals: modifying cavity points; a miss reaching featurisation,
  clustering, cross-docking or statistics; transitive chaining into a giant
  site.

## Acceptance

- Unit tests (TDD) for every step above, including: voxel box of a flat
  cavity; IoU; cores never hold a pair below threshold; the A–B–C case gives
  two sites with B in both; a drifting pocket's cores merge into one site;
  pooling two cavities of one structure; ligand fraction at the boundary; keep
  rule (structure share or ligand); both miss reasons; names and ranks;
  regroup with new parameters; explorer ligand limit.
- Full suite green; `tests/golden/CHANGES.md` entry (PARG data absent).
- Regroup of `EV71_2A_prep` on the laptop copy: report sites (names, counts,
  box sizes) and miss counts for the user's review.
