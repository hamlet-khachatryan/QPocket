"""The distance heatmap's ordering, outside the notebook."""
import numpy as np
import pandas as pd

from qpocket.viz import matrix


def _d():
    ids = ["A", "B", "C", "D"]
    raw = np.array([[0, 1, 9, 9], [1.2, 0, 9, 9], [9, 9, 0, 1], [9, 9, 1, 0]], dtype=float)
    return pd.DataFrame(raw, index=ids, columns=ids)


def test_the_matrix_is_symmetrised_with_a_zero_diagonal():
    m = matrix.symmetrised(_d())
    assert np.allclose(m.to_numpy(), m.to_numpy().T) and m.loc["A", "B"] == 1.1


def test_close_pockets_end_up_adjacent():
    order = matrix.leaf_order(matrix.symmetrised(_d()), ["A", "C", "B", "D"])
    assert abs(order.index("A") - order.index("B")) == 1


def test_tiny_subsets_keep_their_order():
    assert matrix.leaf_order(matrix.symmetrised(_d()), ["B", "A"]) == ["B", "A"]
