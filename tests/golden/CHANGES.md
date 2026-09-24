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
