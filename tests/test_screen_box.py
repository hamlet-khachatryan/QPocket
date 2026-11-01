"""The two docking-box modes of the Screening page, and the box a site saves."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from qpocket.arms.screen import box
from qpocket.arms.sites import final
from qpocket.arms import sites as steps
from tests.test_site_steps import FAR, _finished_run


@pytest.fixture
def run(tmp_path, monkeypatch):
    root = _finished_run(tmp_path, monkeypatch)
    steps.shape_cluster_site(root, FAR)
    final.finalise_site(root, FAR)
    return root


def _contacted(root, residues):
    path = root / "sites" / FAR / "final" / "residue_profile.csv"
    pd.DataFrame({"cluster": "S1", "residue": residues, "contacted_pct": 50.0, "lining_pct": 50.0}).to_csv(
        path, index=False)


def test_cloud_box_covers_every_representative_cloud(run):
    preview = box.cloud_box(run, FAR, padding=5.0)
    reps = final.read_final(run, FAR)["clusters"].query("is_representative")["pdb_id"].tolist()
    assert sorted(preview.clouds) == sorted(reps)
    pts = np.vstack(list(preview.clouds.values()))
    assert preview.tight["min_x"] == pytest.approx(pts[:, 0].min())
    assert preview.padded["max_z"] == pytest.approx(pts[:, 2].max() + 5.0)
    assert len(preview.extremes) == 6 and set(preview.cluster_of) == set(reps)


def test_save_then_saved_box_is_current_until_refinalised(run):
    assert box.saved_box(run, FAR) == (None, "none")
    path = box.save_box(run, FAR, box.cloud_box(run, FAR, 5.0), now=datetime(2026, 10, 27, 19, 0))
    saved, status = box.saved_box(run, FAR)
    assert status == "current" and saved["mode"] == "representative_pockets" and saved["padding"] == 5.0
    assert saved["saved_at"] == "2026-10-27T19:00:00" and path.name == "box.json"
    table = pd.read_csv(run / "sites" / FAR / "metadata" / "clustering_metadata.csv")
    table["cluster_id"] = 0
    table.to_csv(run / "sites" / FAR / "metadata" / "clustering_metadata.csv", index=False)
    steps.set_representative(run, FAR, 0, "P2")
    final.finalise_site(run, FAR)
    assert box.saved_box(run, FAR)[1] == "stale"


