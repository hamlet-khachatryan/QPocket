# Golden outputs

`scripts/make_golden.py OUT` runs the 7-structure PARG pipeline offline (fragment-screen mode, no UniProt renumbering).

```bash
python scripts/make_golden.py tests/golden/expected   # capture (git-ignored)
pytest tests/golden -m golden                        # compare a fresh run
```

Deliberate numerical changes are recorded in `CHANGES.md`.

PLIP (via OpenBabel) is not deterministic between processes: 4B1H gains or loses the CYS872/VAL486 H-bond donors on identical input. The test therefore compares PLIP's inputs byte for byte and leaves out the observed-interaction count tables.

Run it with the venv's tools on PATH (`export PATH=$PWD/.venv/bin:$PATH`); without `pdb2pqr` the
pipeline takes a different path and the comparison fails. CSVs are compared as numbers
(`atol 1e-8`), JSON as data, and the run is checked with both the default worker pool and `workers=1`.
