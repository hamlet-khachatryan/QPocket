"""Fresh pipeline outputs must equal the captured golden outputs."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = REPO / "tests/golden/expected"
EXPECTED = EXPECTED_ROOT / "7KG6"
COMPARED = [
    "metadata/master.csv",
    # observed_interactions_freq.csv and binding_pocket_utilization.csv are left
    # out: PLIP returns different H-bonds for the same file between processes.
    "metadata/potential_interactions_freq.csv",
    "metadata/spectral_distance_matrix.csv",
    "metadata/shape_distance_matrix.csv",
    "metadata/feature_distance_matrix.csv",
    "metadata/clustering_metadata.csv",
    "metadata/clustering_params.json",
    "screening/actives.smi",
]


def _normalise(text: str, root: Path) -> str:
    return text.replace(str(root), "<ROOT>")


@pytest.mark.golden
@pytest.mark.skipif(not EXPECTED.exists(), reason="run scripts/make_golden.py tests/golden/expected first")
def test_outputs_are_identical(tmp_path):
    subprocess.run([sys.executable, str(REPO / "scripts/make_golden.py"), str(tmp_path)], check=True)
    got = tmp_path / "7KG6"
    for rel in COMPARED:
        expected = EXPECTED / rel
        if not expected.exists():
            assert not (got / rel).exists(), f"{rel} appeared"
            continue
        assert _normalise((got / rel).read_text(encoding="utf-8"), tmp_path) == _normalise(
            expected.read_text(encoding="utf-8"), EXPECTED_ROOT
        ), rel
    for folder in ("aligned", "cleaned", "extracted_ligands"):
        for f in sorted((EXPECTED / "structures" / folder).iterdir()):
            rel = f"structures/{folder}/{f.name}"
            assert (got / rel).read_bytes() == f.read_bytes(), rel
    for pocket in sorted(p for p in (EXPECTED / "pockets").iterdir() if p.is_dir()):
        rel = f"pockets/{pocket.name}/featurized.csv"
        if (EXPECTED / rel).exists():
            assert (got / rel).read_text(encoding="utf-8") == (EXPECTED / rel).read_text(
                encoding="utf-8"
            ), rel
