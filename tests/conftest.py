"""
conftest.py — session-scoped mock injection + shared fixtures.

Heavy modules (pymol, apbs_binary, cavitomix) are patched into sys.modules
*before* any test imports them, so module-level side-effects
(cmd.feedback in alignment.py, popen_apbs in features.py) never execute.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Inject mock modules before any QPocket source is imported
# ---------------------------------------------------------------------------

def _make_pymol_mock() -> types.ModuleType:
    pymol_mod = types.ModuleType("pymol")
    cmd_mock = MagicMock()
    # Make cmd.get_chains return a list so iteration works
    cmd_mock.get_chains.return_value = ["A"]
    cmd_mock.get_fastastr.return_value = ">header\nAACCGG"
    cmd_mock.align.return_value = (0.5, 10, 10, 10, 1, 0.5, 10)
    pymol_mod.cmd = cmd_mock
    return pymol_mod


def _make_apbs_mock() -> types.ModuleType:
    apbs_mod = types.ModuleType("apbs_binary")
    proc_mock = MagicMock()
    proc_mock.communicate.return_value = (b"", b"")
    apbs_mod.popen_apbs = MagicMock(return_value=proc_mock)
    return apbs_mod


def _make_cavitomix_mock() -> types.ModuleType:
    cav_mod = types.ModuleType("cavitomix")
    cav_mod.cavfind = MagicMock()
    return cav_mod


# Patch before any src imports happen
for _name, _factory in [
    ("pymol", _make_pymol_mock),
    ("apbs_binary", _make_apbs_mock),
    ("cavitomix", _make_cavitomix_mock),
    ("cavitomix.cavfind", lambda: MagicMock()),
]:
    if _name not in sys.modules:
        sys.modules[_name] = _factory()


# ---------------------------------------------------------------------------
# Test-data helpers
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"

MINIMAL_PDB_CONTENT = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.538   1.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.077   2.400   1.000  1.00  0.00           C
ATOM      4  N   ALA A   2       3.000   3.000   1.000  1.00  0.00           N
ATOM      5  CA  ALA A   2       3.538   3.000   1.000  1.00  0.00           C
ATOM      6  C   ALA A   2       4.077   4.400   1.000  1.00  0.00           C
ATOM      7  N   ALA A   3       5.000   5.000   1.000  1.00  0.00           N
ATOM      8  CA  ALA A   3       5.538   5.000   1.000  1.00  0.00           C
ATOM      9  C   ALA A   3       6.077   6.400   1.000  1.00  0.00           C
HETATM   10  C1  LIG A 100       7.000   7.000   7.000  1.00  0.00           C
END
"""

SAMPLE_DX_CONTENT = """\
object 1 class gridpositions counts 3 3 3
origin 0.0 0.0 0.0
delta 1.0 0.0 0.0
delta 0.0 1.0 0.0
delta 0.0 0.0 1.0
object 2 class gridconnections counts 3 3 3
object 3 class array type double rank 0 items 27 data follows
0.1 0.2 0.3
0.4 0.5 0.6
0.7 0.8 0.9
1.0 1.1 1.2
1.3 1.4 1.5
1.6 1.7 1.8
1.9 2.0 2.1
2.2 2.3 2.4
2.5 2.6 2.7
object "density" class field
"""


# ---------------------------------------------------------------------------
# Session-scoped: write static data files once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def create_data_dir():
    """Ensure tests/data/ exists and is populated with static test files."""
    DATA_DIR.mkdir(exist_ok=True)

    (DATA_DIR / "minimal.pdb").write_text(MINIMAL_PDB_CONTENT)
    (DATA_DIR / "sample.dx").write_text(SAMPLE_DX_CONTENT)

    # pocket.csv — 20-point cloud
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (20, 3))
    pd.DataFrame(coords, columns=["x", "y", "z"]).to_csv(DATA_DIR / "pocket.csv", index=False)

    # featurized.csv — 20-point cloud + 6 feature columns
    feat_df = pd.DataFrame(
        np.hstack([coords, rng.standard_normal((20, 6))]),
        columns=["x", "y", "z", "apbs", "cp", "lp", "h_donor", "h_acceptor", "aromatic"],
    )
    feat_df.to_csv(DATA_DIR / "featurized.csv", index=False)

    # master.csv — 3-row metadata
    master = pd.DataFrame(
        {
            "pdb_id": ["1ABC", "2DEF", "3GHI"],
            "aligned_file": ["1ABC_aligned.pdb", "2DEF_aligned.pdb", "3GHI_aligned.pdb"],
            "cleaned_file": ["1ABC_cleaned.pdb", "2DEF_cleaned.pdb", "3GHI_cleaned.pdb"],
        }
    )
    master.to_csv(DATA_DIR / "master.csv", index=False)

    # sample_interactions.h5
    h5_path = DATA_DIR / "sample_interactions.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset(
            "1ABC",
            data=[b"ALA_10|HBondDonor", b"GLU_20|SaltBridge"],
        )
        hf.create_dataset(
            "2DEF",
            data=[b"PHE_5|PiStack", b"LYS_30|HBondAcceptor"],
        )


# ---------------------------------------------------------------------------
# Function-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pocket_csv(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, (20, 3))
    p = tmp_path / "pocket.csv"
    pd.DataFrame(coords, columns=["x", "y", "z"]).to_csv(p, index=False)
    return p


@pytest.fixture
def featurized_csv(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, (20, 3))
    feat = rng.standard_normal((20, 6))
    df = pd.DataFrame(
        np.hstack([coords, feat]),
        columns=["x", "y", "z", "apbs", "cp", "lp", "h_donor", "h_acceptor", "aromatic"],
    )
    p = tmp_path / "featurized.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def minimal_pdb(tmp_path) -> Path:
    p = tmp_path / "minimal.pdb"
    p.write_text(MINIMAL_PDB_CONTENT)
    return p


@pytest.fixture
def sample_dx(tmp_path) -> Path:
    p = tmp_path / "sample.dx"
    p.write_text(SAMPLE_DX_CONTENT)
    return p


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_config(**overrides):
    """Returns a valid QPocketConfig, optionally overriding nested fields."""
    from qpocket_pipeline.config import (
        AlignParams,
        BoundingBoxParams,
        CleanParams,
        DistanceConfig,
        FragmentScreenParams,
        IOConfig,
        PathConfig,
        QueryParams,
        QPocketConfig,
        ReferenceParams,
    )

    cfg = QPocketConfig(
        mode=overrides.pop("mode", "pdb_query"),
        reference=overrides.pop(
            "reference",
            ReferenceParams(
                pdb_id="1ABC",
                chain_id="A",
                ligand_id="LIG",
                pocket_definition="Interactions",
                pocket_mode="cavity_detection",
            ),
        ),
        paths=overrides.pop("paths", PathConfig()),
        fragment_screen=overrides.pop(
            "fragment_screen",
            FragmentScreenParams(
                input_type="csv",
                csv_path="/some/path.csv",
                reference_index=0,
            ),
        ),
        pdb_query=overrides.pop("pdb_query", QueryParams()),
        alignment=overrides.pop("alignment", AlignParams()),
        cleaning=overrides.pop("cleaning", CleanParams()),
        io=overrides.pop("io", IOConfig()),
        distance=overrides.pop("distance", DistanceConfig()),
    )
    return cfg


@pytest.fixture
def valid_config():
    return make_config()


@pytest.fixture
def valid_config_fragment_screen():
    return make_config(
        mode="fragment_screen",
    )
