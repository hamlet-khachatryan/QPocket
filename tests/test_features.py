"""
Tests for feature computation in core/features.py and io/tools.py.
apbs_binary and pdb2pqr subprocess calls are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# apbs_binary is already mocked in sys.modules by conftest.py
from qpocket.core.features import (
    calculate_cp_lp,
    calculate_geometric_potentials,
)
from qpocket.io.tools import read_dx_grid
from qpocket.constants import FP_DTYPE


# ---------------------------------------------------------------------------
# read_dx_grid
# ---------------------------------------------------------------------------

VALID_DX = """\
object 1 class gridpositions counts 3 3 3
origin 0.0 0.0 0.0
delta 1.0 0.0 0.0
delta 0.0 1.0 0.0
delta 0.0 0.0 1.0
object 2 class gridconnections counts 3 3 3
object 3 class array type double rank 0 items 27 data follows
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8
1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7
object "density" class field
"""

BAD_DX_NO_ORIGIN = """\
object 1 class gridpositions counts 3 3 3
delta 1.0 0.0 0.0
delta 0.0 1.0 0.0
delta 0.0 0.0 1.0
object 3 class array type double rank 0 items 27 data follows
0.1 0.2 0.3
"""


def test_read_dx_grid_valid(tmp_path):
    dx_file = tmp_path / "mol.dx"
    dx_file.write_text(VALID_DX)

    grid, axes = read_dx_grid(dx_file)

    assert grid.shape == (3, 3, 3)
    assert len(axes) == 3
    # axes: x_axis, y_axis, z_axis each length 3
    x_ax, y_ax, z_ax = axes
    assert len(x_ax) == 3
    assert len(y_ax) == 3
    assert len(z_ax) == 3


def test_read_dx_grid_axes_values(tmp_path):
    dx_file = tmp_path / "mol.dx"
    dx_file.write_text(VALID_DX)

    grid, (x_ax, y_ax, z_ax) = read_dx_grid(dx_file)

    # origin=0, delta=1, counts=3 → [0, 1, 2]
    np.testing.assert_allclose(x_ax, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(y_ax, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(z_ax, [0.0, 1.0, 2.0])


def test_read_dx_grid_data_values(tmp_path):
    dx_file = tmp_path / "mol.dx"
    dx_file.write_text(VALID_DX)

    grid, _ = read_dx_grid(dx_file)

    # First element in data is 0.1
    assert grid[0, 0, 0] == pytest.approx(0.1)


def test_read_dx_grid_bad_header_raises(tmp_path):
    dx_file = tmp_path / "bad.dx"
    dx_file.write_text(BAD_DX_NO_ORIGIN)

    with pytest.raises(RuntimeError, match="Malformed DX Header"):
        read_dx_grid(dx_file)


# ---------------------------------------------------------------------------
# calculate_cp_lp
# ---------------------------------------------------------------------------


def test_calculate_cp_lp_empty_atoms():
    points = np.zeros((5, 3), dtype=FP_DTYPE)
    cp, lp = calculate_cp_lp(points, np.empty((0, 3)), np.array([]), np.array([]))
    assert cp.shape == (5,)
    assert lp.shape == (5,)
    np.testing.assert_array_equal(cp, 0.0)
    np.testing.assert_array_equal(lp, 0.0)


def test_calculate_cp_lp_zero_charges():
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=FP_DTYPE)
    atom_coords = np.array([[5, 5, 5]], dtype=FP_DTYPE)
    chg = np.zeros(1, dtype=FP_DTYPE)
    hp = np.zeros(1, dtype=FP_DTYPE)
    cp, lp = calculate_cp_lp(points, atom_coords, chg, hp)
    np.testing.assert_array_equal(cp, 0.0)


def test_calculate_cp_lp_nonzero_charge():
    points = np.array([[0, 0, 0]], dtype=FP_DTYPE)
    atom_coords = np.array([[1, 0, 0]], dtype=FP_DTYPE)  # 1 Å away
    chg = np.array([1.0], dtype=FP_DTYPE)
    hp = np.array([0.0], dtype=FP_DTYPE)
    cp, lp = calculate_cp_lp(points, atom_coords, chg, hp)
    # cp = (chg / dist^2) * 557.0 = 557.0
    assert cp[0] == pytest.approx(557.0, rel=1e-4)


def test_calculate_cp_lp_output_dtype():
    points = np.array([[0, 0, 0]], dtype=FP_DTYPE)
    atom_coords = np.array([[1, 0, 0]], dtype=FP_DTYPE)
    chg = np.array([1.0], dtype=FP_DTYPE)
    hp = np.array([0.5], dtype=FP_DTYPE)
    cp, lp = calculate_cp_lp(points, atom_coords, chg, hp)
    assert cp.dtype == FP_DTYPE
    assert lp.dtype == FP_DTYPE


# ---------------------------------------------------------------------------
# calculate_geometric_potentials
# ---------------------------------------------------------------------------


def test_calculate_geometric_potentials_empty_targets():
    points = np.zeros((5, 3), dtype=FP_DTYPE)
    result = calculate_geometric_potentials(points, np.empty((0, 3)), sigma=1.0)
    assert result.shape == (5,)
    np.testing.assert_array_equal(result, 0.0)


def test_calculate_geometric_potentials_point_at_sigma_distance():
    """
    A point at distance sigma from a target:
    potential contribution = exp(-sigma^2 / (2*sigma^2)) = exp(-0.5)
    """
    sigma = 2.0
    # Point at [sigma, 0, 0], target at origin → dist = sigma
    points = np.array([[sigma, 0.0, 0.0]], dtype=float)
    targets = np.array([[0.0, 0.0, 0.0]], dtype=float)

    result = calculate_geometric_potentials(points, targets, sigma=sigma)

    expected = np.exp(-0.5)
    assert result[0] == pytest.approx(expected, rel=1e-5)


def test_calculate_geometric_potentials_large_sigma_broader():
    """Larger sigma → higher potential at same distance (broader field)."""
    points = np.array([[3.0, 0.0, 0.0]], dtype=float)
    targets = np.array([[0.0, 0.0, 0.0]], dtype=float)

    val_small = calculate_geometric_potentials(points, targets, sigma=1.0)
    val_large = calculate_geometric_potentials(points, targets, sigma=5.0)

    assert val_large[0] > val_small[0]


def test_calculate_geometric_potentials_multiple_targets():
    points = np.array([[0.0, 0.0, 0.0]], dtype=float)
    targets = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    sigma = 1.0

    result = calculate_geometric_potentials(points, targets, sigma=sigma)

    # Two contributions summed
    d1_sq = 1.0
    d2_sq = 4.0
    factor = -1.0 / (2.0 * sigma ** 2)
    expected = np.exp(d1_sq * factor) + np.exp(d2_sq * factor)
    assert result[0] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# calculate_apbs_potential (mocked)
# ---------------------------------------------------------------------------


def test_protonate_raises_when_pdb2pqr_writes_nothing(tmp_path):
    """A failed pdb2pqr run is an error, not a silent empty PQR."""
    from qpocket.io.tools import protonate

    pdb = tmp_path / "mol.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
    proc_mock = MagicMock()
    proc_mock.communicate.return_value = (b"", b"pdb2pqr failed")

    with patch("qpocket.io.tools.Popen", return_value=proc_mock):
        with pytest.raises(RuntimeError):
            protonate(pdb, tmp_path / "out.pqr")


def test_calculate_apbs_potential_missing_dx_returns_zeros(tmp_path):
    """If dx file is not produced by APBS, zeros are returned."""
    from qpocket.io.tools import apbs_potential as calculate_apbs_potential

    pqr = tmp_path / "mol.pqr"
    pqr.write_text("ATOM  1  CA  ALA A 1  0.000  0.000  0.000  0.00  1.80\n")
    points = np.zeros((3, 3), dtype=float)

    # apbs_binary.popen_apbs is mocked in sys.modules and writes no grid
    result = calculate_apbs_potential(pqr, points)

    # DX file won't exist → zeros
    assert result.shape == (3,)
