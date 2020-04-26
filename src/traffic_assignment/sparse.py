from scipy.sparse import dok_matrix



class SparseMatrix:
    def __init__(self, shape, dtype='d'):
        self._data = dok_matrix(shape, dtype=dtype)

    def __setitem__(self, key, value):
        n, m = self._size
        i, j = key
        resize = False
        if i >= n:
            resize = True
            n = 2 * n
            self._shape[0] = i+1
        if j >= m:
            resize = True
            m = 2 * n
            self._shape[1] = j+1
        if resize:
            self._resize(n, m)
        self._data[i, j] = value

    def get_data(self):
        n, m = self._shape
        return self._data[:n, :m].copy()
