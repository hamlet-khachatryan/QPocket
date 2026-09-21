"""Summarise how two golden runs differ: per-file max |delta| and cluster agreement."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qpocket.core.clustering import compare_labelings

MATRICES = ["spectral_distance_matrix", "shape_distance_matrix", "feature_distance_matrix"]


def main(old: Path, new: Path, title: str) -> None:
    """Print a Markdown summary of what changed between two runs.

    Args:
        old: Root of the previous golden run.
        new: Root of the new run.
        title: Heading for the entry.
    """
    old, new = old / "7KG6", new / "7KG6"
    print(f"\n## {title}\n")
    for m in MATRICES:
        a = pd.read_csv(old / f"metadata/{m}.csv", index_col=0)
        b = pd.read_csv(new / f"metadata/{m}.csv", index_col=0).loc[a.index, a.columns]
        print(f"- `{m}`: max |delta| {np.abs(a.values - b.values).max():.3g} (max value {np.abs(a.values).max():.3g})")
    for pocket in sorted(p for p in (old / "pockets").iterdir() if p.is_dir()):
        a = pd.read_csv(pocket / "featurized.csv")
        b = pd.read_csv(new / "pockets" / pocket.name / "featurized.csv")
        if a.shape != b.shape:
            print(f"- {pocket.name} featurized: {len(a)} -> {len(b)} points")
            continue
        deltas = {c: float(np.abs(a[c] - b[c]).max()) for c in a.columns if c not in ("x", "y", "z")}
        changed = {k: v for k, v in deltas.items() if v > 0}
        print(f"- {pocket.name} featurized: " + (", ".join(f"{k} {v:.3g}" for k, v in changed.items()) or "identical"))
    la = pd.read_csv(old / "metadata/clustering_metadata.csv").set_index("pdb_id")["cluster_id"]
    lb = pd.read_csv(new / "metadata/clustering_metadata.csv").set_index("pdb_id")["cluster_id"].loc[la.index]
    cmp = compare_labelings(la.values, lb.values)
    print(f"- clusters: ARI {cmp.ari:.3f}, {cmp.n_moved} structure(s) moved")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "change")
