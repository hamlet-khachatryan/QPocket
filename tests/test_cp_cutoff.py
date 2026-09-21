"""cp/lp switch atoms off smoothly between 10 and 12 A instead of a hard cutoff."""
import numpy as np

from qpocket.core.features import calculate_cp_lp, switching


def test_switching_is_one_then_zero_and_smooth_between():
    s = switching(np.array([0.0, 9.99, 10.0, 11.0, 12.0, 15.0]))
    assert s[0] == s[1] == s[2] == 1.0 and s[4] == s[5] == 0.0 and 0 < s[3] < 1
    r = np.linspace(9.9, 12.1, 2001)
    assert np.abs(np.diff(switching(r))).max() < 2e-3        # no jumps


def _cp_lp_with_atom_at(x):
    points = np.zeros((1, 3))
    coords = np.array([[3.0, 0, 0], [x, 0, 0]])
    return calculate_cp_lp(points, coords, np.array([1.0, 1.0]), np.array([1.0, -2.0]))


def test_cp_and_lp_are_continuous_across_the_cutoff():
    inside, outside = _cp_lp_with_atom_at(11.9995), _cp_lp_with_atom_at(12.0005)
    assert abs(inside[0][0] - outside[0][0]) < 1e-6
    assert abs(inside[1][0] - outside[1][0]) < 1e-6


def test_prefilter_equals_brute_force_with_switching():
    rng = np.random.default_rng(0)
    pts, atoms = rng.random((30, 3)) * 5, rng.random((400, 3)) * 40 - 15
    q, h = rng.normal(size=400), rng.normal(size=400)
    cp, lp = calculate_cp_lp(pts, atoms, q, h)
    r = np.linalg.norm(pts[:, None] - atoms[None], axis=2)
    assert np.allclose(cp, (switching(r) * q / r ** 2).sum(1) * 557.0, rtol=1e-4)


def test_atoms_within_ten_angstrom_are_unchanged():
    points = np.zeros((1, 3))
    coords = np.array([[3.0, 0, 0], [5.0, 1.0, 0]])
    cp, _ = calculate_cp_lp(points, coords, np.array([1.0, -0.5]), np.zeros(2))
    assert np.isclose(cp[0], (1.0 / 9.0 - 0.5 / 26.0) * 557.0, rtol=1e-5)
