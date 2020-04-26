from traffic_assignment.sparse import SparseMatrix
import numpy as np


def test_resize_on_assign():
    data = SparseMatrix((3, 4))
    data[0, 1] = 5
    data[3, 1] = 2
    matrix = data.get_data()
    expected = np.zeros((4, 4))
    expected[0, 1] = 5
    expected[3, 1] = 2
    actual = matrix.toarray()
    print(actual)
    assert np.array_equal(actual, expected)
    assert data._data.shape == (6, 4)
    assert matrix.shape == (4, 4)
