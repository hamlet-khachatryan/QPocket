# Golden baseline changes

Every intentional change to the golden outputs is recorded here.

## Task 1 — environment locked with uv (no code change)

Baseline re-captured under `uv.lock` (numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2,
pandas 2.3.3, apbs-binary 3.4.1.1, pdb2pqr 3.6.2). The previous unlocked `.venv`
gave different APBS values on identical code:

- `apbs` column: max |delta| 0.039 (4B1J), 0.023 (7KG6), 0.009 (7KFP) kT/e; other pockets identical
- every other feature column: 0
- `feature_distance_matrix`: max |delta| 0.28; `shape_distance_matrix`: 0
- cluster assignments changed; runs are deterministic within the locked environment

## Task 13 — hydrophobicity from real elements and canonical residue names

- `spectral_distance_matrix`: max |delta| 0 (max value 768)
- `shape_distance_matrix`: max |delta| 0 (max value 0.035)
- `feature_distance_matrix`: max |delta| 0 (max value 409)
- 4A0D featurized: identical
- 4B1H featurized: identical
- 4B1J featurized: identical
- 6HMM featurized: identical
- 6O9Y featurized: identical
- 7KFP featurized: identical
- 7KG6 featurized: identical
- clusters: ARI 1.000, 0 structure(s) moved

No change on PARG: cleaning removes metal ions and the PQR behind cp/lp has none yet.
The fix matters once Task 14 feeds the ion-repaired PQR to cp/lp, and for kept
cofactors (heme iron, selenomethionine).

## Task 14 — one pdb2pqr run per structure, ions repaired for every feature

- `spectral_distance_matrix`: max |delta| 0 (max value 768)
- `shape_distance_matrix`: max |delta| 0 (max value 0.035)
- `feature_distance_matrix`: max |delta| 0 (max value 409)
- 4A0D featurized: identical
- 4B1H featurized: identical
- 4B1J featurized: identical
- 6HMM featurized: identical
- 6O9Y featurized: identical
- 7KFP featurized: identical
- 7KG6 featurized: identical
- clusters: ARI 1.000, 0 structure(s) moved

Identical on PARG: cleaning strips its ions, and AMBER names map back to the same
hydrophobicity. Each pocket now keeps protein.pqr (and pdb2pqr's protein.log); pdb2pqr
runs once per structure instead of twice. Structures that keep ions (via
cleaning.allowed_cofactors) now get them in cp/lp as well as APBS.

## Task 14b — ion repair adds real ions only (no calcium from a cofactor's CA)

- `spectral_distance_matrix`: max |delta| 1.37 (max value 768)
- `shape_distance_matrix`: max |delta| 0 (max value 0.035)
- `feature_distance_matrix`: max |delta| 0.733 (max value 409)
- 4A0D featurized: identical
- 4B1H featurized: identical
- 4B1J featurized: identical
- 6HMM featurized: identical
- 6O9Y featurized: identical
- 7KFP featurized: apbs 0.062
- 7KG6 featurized: apbs 0.0574
- clusters: ARI 1.000, 0 structure(s) moved

Only the two structures carrying the CME cofactor change (7KG6, 7KFP): ion repair had
re-added CME's alpha carbon (atom CA) as a +2 calcium ion, which shifted APBS. The
bug predates the refactor; it now matches ions by residue name.

## Task 16 — smooth 10–12 Å switching for cp/lp

- `spectral_distance_matrix`: max |delta| 26.8 (max value 767)
- `shape_distance_matrix`: max |delta| 0 (max value 0.035)
- `feature_distance_matrix`: max |delta| 14.3 (max value 409)
- 4A0D featurized: cp 15.7, lp 0.001
- 4B1H featurized: cp 22.6, lp 0.000553
- 4B1J featurized: cp 22.1, lp 0.00054
- 6HMM featurized: cp 12.6, lp 0.000839
- 6O9Y featurized: cp 11.1, lp 0.00111
- 7KFP featurized: cp 17.1, lp 0.000878
- 7KG6 featurized: cp 16.1, lp 0.00103
- clusters: ARI 1.000, 0 structure(s) moved

cp changes by design (median 10% of its range on 7KG6): atoms 10-12 A away now fade out
instead of counting fully. lp barely moves. Robustness, 7KG6 protein jittered by
+-0.0005 A (PDB rounding) over 20 trials: max |cp change| 2.71 with the hard cutoff,
0.04 with switching.

## Task 17 — geodesic cache keyed on featurized.csv content and radius

No change: a fresh run builds each cache once. Reruns and resubmitted SLURM tasks no
longer reuse a cache from an older featurized.csv or a different neighbour radius.

## Task 18 — two-stage threshold cuts, affinity settings honoured, recluster

- `spectral_distance_matrix`: max |delta| 0 (max value 767)
- `shape_distance_matrix`: max |delta| 0 (max value 0.035)
- `feature_distance_matrix`: max |delta| 0 (max value 409)
- 4A0D featurized: identical
- 4B1H featurized: identical
- 4B1J featurized: identical
- 6HMM featurized: identical
- 6O9Y featurized: identical
- 7KFP featurized: identical
- 7KG6 featurized: identical
- clusters: ARI 1.000, 0 structure(s) moved

Only clustering_params.json changes: each stage now records criterion and threshold.
Defaults (count criterion) reproduce the stored clustering exactly.

## 2026-09-24 — ProLIF replaces PLIP as the interaction detector

- `spectral_distance_matrix`: max |delta| 220 (max value 767)
- `shape_distance_matrix`: max |delta| 0.0136 (max value 0.035)
- `feature_distance_matrix`: max |delta| 117 (max value 409)
- 4A0D featurized: identical
- 4B1H featurized: identical
- 4B1J featurized: 215 -> 214 points
- 6HMM featurized: identical
- 6O9Y featurized: 180 -> 145 points
- 7KFP featurized: 427 -> 302 points
- 7KG6 featurized: 481 -> 463 points
- clusters: ARI 0.314, 2 structure(s) moved

**Why the geometry moved, not just the interaction tables.** PARG.yaml sets
`reference.pocket_definition: Interactions`, so the second pass crops each pocket
to its observed contacts and re-featurises. The interaction detector therefore
feeds pocket geometry, and changing it necessarily moves `featurized.csv`, all
three distance matrices and the clustering. The plan for this change expected
only the interaction tables to move; that was wrong for this configuration.

**Interaction counts, pooled over the seven structures:**

| type | PLIP | ProLIF |
|---|---|---|
| H-bond Donor | 28 | 17 |
| H-bond Acceptor | 11 | 2 |
| Pi-stacking (P) | 11 | 5 |
| Hydrophobic | 1 | 6 |

`observed_interactions_freq.csv` goes from 18 rows to 11;
`potential_interactions_freq.csv` and `binding_pocket_utilization.csv` keep their
32 and 7 rows, since those come from geometry rather than from detection.

ProLIF's defaults are stricter than PLIP's published cutoffs: hydrogen bonds at
3.5 A and 130-180 degrees against PLIP's 4.1 A and 100 degrees, and pi-stacking
with tighter angular criteria. Fewer contacts pass, so the interaction-cropped
pockets shrink and every downstream number follows. Setting
`interactions.parameters` to the PLIP-compatible preset documented in
`docs/configuration.md` moves the thresholds back; it is not the default.

Two caveats on the previous baseline, both measured:

- PLIP's `LigandFinder` strips every hydrogen from the ligand it extracts
  (`preparation.py:373`), so under `NOHYDRO` it reported **no** ligand-donor
  hydrogen bonds at all. The old counts came from PLIP protonating the ligand
  itself with OpenBabel, whose hydrogen placement is not reproducible: 290 of
  4350 hydrogens moved between two identical calls.
- The new path is deterministic. Three consecutive runs produce byte-identical
  interaction strings for every benchmark structure.

## 2026-09-24 — actives.smi lists every ligand, with real bond orders

Only `screening/actives.smi` changes. `featurized.csv`, the three distance
matrices and the clustering are untouched.

```
before   SMILES	PDB_ID	Ligand_Name
         CN1C(O)C2NC(SCCN3CCC(C(O)O)CC3)NC2N(C)C1O	7KG6	WDG

after    SMILES	PDB_ID	Ligand_Name
         Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OCC2OC(O)C(O)C2O)C(O)C1O	4B1H	AR6
         Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OCC2NCC(O)C2O)C(O)C1O	4B1J	A1R
         Cn1c(=O)c2c(ncn2CC(O)CN2CCOCC2)n(C)c1=O	6O9Y	M0V
         Cn1c(=S)c2c(ncn2CCNC(=O)NS(C)(=O)=O)n(C)c1=O	7KFP	WD7
         Cn1c(=O)c2[nH]c(SCCN3CCC(C(=O)O)CC3)nc2n(C)c1=O	7KG6	WDG
```

One row becomes five, one per bound structure. The previous file held only the
reference, because SMILES were derived inside the reference branch alone.

The chemistry was also wrong. `Chem.MolFromPDBBlock` bonds by proximity and
makes every bond single, so WDG's xanthine was recorded as an alcohol and the
adenines of AR6 and A1R lost their aromatic rings. The SMILES now come from each
ligand's chemical-component definition, which is where the bond orders live.

4A0D and 6HMM contribute no row: they are unbound.

## Binding-site discovery Task 8 — layout 3: per-site clouds, no features or matrices in `qpocket run`

`qpocket run` now stops after one cloud per site per structure. The run level
keeps `metadata/master.csv` and `metadata/failures.csv`; everything else that was
compared moved under `sites/<site_id>/` (`metadata/master.csv`,
`metadata/potential_interactions_freq.csv`, `screening/actives.smi`,
`pockets/<pdb_id>/binding_site.csv`), next to `sites/sites.csv` and
`structures/ligands.csv`. `featurized.csv` and the distance matrices are no
longer written by the run; they come back as per-site steps. The baseline
under `tests/golden/expected` must be re-captured with
`python scripts/make_golden.py tests/golden/expected` on a machine that has
`experiments/PARG`; it was not re-captured here because that data is absent.

## Buffers and cryoprotectants are never ligands

`find_ligands` skips `qpocket.constants.BUFFER_RESNAMES` (GOL, EDO, PEGs, DMS,
HEPES, Tris, citrate, …) unless a run lists them in `sites.keep_ligands`. PARG
holds GOL (e.g. 6HMM), so its sites and ligand tables change. The PARG data is
not available at the time of writing, so the golden test skips; re-capture the
baseline when it returns.

## Sites defined by CavFind cavities

Sites are no longer ligand-linked (`sites.link_cutoff`). CavFind cavities of at
least `sites.min_cavity_points` points are grouped by box IoU
(`sites.cavity_overlap`), ligands are assigned by box containment, each
structure's cloud is its own member cavities' points (no box cut, no
Interactions crop), site ids are `S<rank>-<residue>`, and CavFind misses go to
`sites/cavfind_misses.csv`. Every site table, cloud and downstream matrix of the
golden run changes. The PARG data is not available at the time of writing, so
the golden test skips; re-capture the baseline when it returns.

## Clustering cuts are site percentiles

Every hierarchical cut is now a percentile of the site's own pairwise distances
(`distance.clustering.<stage>.percentile`: shape 15, chemistry 20, measured on the
whole site's chemistry distances). The old relative 0.5 cuts no longer apply, so
the golden clusterings change. The PARG data is not available at the time of
writing; re-capture the baseline when it returns.
