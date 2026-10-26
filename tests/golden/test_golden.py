"""Fresh pipeline outputs must equal the captured golden outputs."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = REPO / "tests/golden/expected"
EXPECTED = EXPECTED_ROOT / "7KG6"
COMPARED = [
    "metadata/master.csv",
    "metadata/failures.csv",
    "structures/ligands.csv",
    "sites/sites.csv",
]
# Per site. observed_interactions_freq.csv and binding_pocket_utilization.csv are
# left out: PLIP returns different H-bonds for the same file between processes.
SITE_COMPARED = [
    "metadata/master.csv",
    "metadata/potential_interactions_freq.csv",
    "screening/actives.smi",
]


def _normalise(text: str, root: Path) -> str:
    return text.replace(str(root), "<ROOT>")


def _same_table(got: Path, expected: Path, got_root: Path, rel: str) -> None:
    """CSV tables equal as numbers (atol 1e-8), whatever their formatting or column order."""
    a = pd.read_csv(got)
    b = pd.read_csv(expected)
    assert sorted(a.columns) == sorted(b.columns), rel
    assert len(a) == len(b), rel
    a = a[b.columns]
    for column in b.columns:
        if pd.api.types.is_numeric_dtype(b[column]):
            assert np.allclose(a[column].to_numpy(float), b[column].to_numpy(float),
                               atol=1e-8, rtol=0, equal_nan=True), f"{rel}:{column}"
        else:
            left = a[column].map(lambda s: _normalise(str(s), got_root))
            right = b[column].map(lambda s: _normalise(str(s), EXPECTED_ROOT))
            assert left.tolist() == right.tolist(), f"{rel}:{column}"


def _same_file(got_root: Path, rel: str) -> None:
    got, expected = got_root / "7KG6" / rel, EXPECTED / rel
    if not expected.exists():
        assert not got.exists(), f"{rel} appeared"
        return
    if rel.endswith(".csv"):
        _same_table(got, expected, got_root, rel)
    elif rel.endswith(".json"):
        assert json.loads(_normalise(got.read_text(encoding="utf-8"), got_root)) == json.loads(
            _normalise(expected.read_text(encoding="utf-8"), EXPECTED_ROOT)), rel
    else:
        assert _normalise(got.read_text(encoding="utf-8"), got_root) == _normalise(
            expected.read_text(encoding="utf-8"), EXPECTED_ROOT), rel


@pytest.mark.golden
@pytest.mark.skipif(not EXPECTED.exists(), reason="run scripts/make_golden.py tests/golden/expected first")
@pytest.mark.skipif(not (REPO / "experiments/PARG/7KG6/structures/original").exists(),
                    reason="needs the bundled PARG structures")
@pytest.mark.parametrize("workers", [0, 1])
def test_outputs_are_numerically_equal(tmp_path, workers):
    subprocess.run([sys.executable, str(REPO / "scripts/make_golden.py"), str(tmp_path),
                    "--workers", str(workers)], check=True)
    got = tmp_path / "7KG6"
    for rel in COMPARED:
        _same_file(tmp_path, rel)
    for folder in ("aligned", "cleaned", "extracted_ligands"):
        for f in sorted((EXPECTED / "structures" / folder).iterdir()):
            rel = f"structures/{folder}/{f.name}"
            assert (got / rel).read_bytes() == f.read_bytes(), rel
    for site in sorted(p for p in (EXPECTED / "sites").iterdir() if p.is_dir()):
        for rel in SITE_COMPARED:
            _same_file(tmp_path, f"sites/{site.name}/{rel}")
        for pocket in sorted(p for p in (site / "pockets").iterdir() if p.is_dir()):
            _same_file(tmp_path, f"sites/{site.name}/pockets/{pocket.name}/binding_site.csv")
    # layout 3: the run stops after the clouds; features and matrices are per-site steps.
    assert not list(got.rglob("featurized.csv")) and not list(got.rglob("*distance_matrix.csv"))
