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
