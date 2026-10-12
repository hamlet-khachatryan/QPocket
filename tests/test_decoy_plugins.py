"""Decoy generation is chosen by name; LUDe is the built-in."""
from pathlib import Path

import pytest

from qpocket.core.ports import DECOYS
from qpocket.screen_config import ScreenConfig, validate_screen_config


def test_lude_is_registered_with_its_settings():
    plugin = DECOYS.get("lude")
    params = plugin.params(chembl_url="u", chembl_dir="d", n_decoys_per_active=3, max_similarity=0.1,
                           mw_diff=1.0, logp_diff=1.0, rb_diff=1, hba_diff=1, hbd_diff=1, charge_diff=0)
    assert params.n_decoys_per_active == 3


def test_the_config_names_the_strategy():
    assert ScreenConfig().decoy.strategy == "lude"
    cfg = ScreenConfig()
    cfg.decoy.strategy = "nope"
    with pytest.raises(ValueError, match="decoy.strategy"):
        validate_screen_config(cfg)


def test_params_come_from_the_decoy_and_chembl_sections():
    from qpocket.arms.screen.campaign import decoy_params

    cfg = ScreenConfig()
    cfg.decoy.n_decoys_per_active = 7
    cfg.chembl.local_dir = "/pool"
    params = decoy_params(cfg, DECOYS.get("lude"))
    assert params.n_decoys_per_active == 7 and params.chembl_dir == "/pool"


def test_lude_downloads_the_pool_then_writes_decoys(tmp_path, monkeypatch):
    from qpocket.plugins.decoys import lude

    seen = {}
    monkeypatch.setattr(lude, "download_chembl_dataset", lambda url, d: seen.setdefault("pool", Path(d)))
    monkeypatch.setattr(lude, "generate_decoys_lude",
                        lambda actives, pool, out, cfg: (out / "decoys.smi").write_text("C\tD1\n"))
    params = DECOYS.get("lude").params(chembl_url="u", chembl_dir=str(tmp_path / "pool"),
                                       n_decoys_per_active=1, max_similarity=0.1, mw_diff=1.0,
                                       logp_diff=1.0, rb_diff=1, hba_diff=1, hbd_diff=1, charge_diff=0)
    out = DECOYS.get("lude").fn(tmp_path / "actives.smi", tmp_path, params)
    assert out == tmp_path / "decoys.smi" and seen["pool"] == tmp_path / "pool"
