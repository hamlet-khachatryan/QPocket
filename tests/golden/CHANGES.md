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
