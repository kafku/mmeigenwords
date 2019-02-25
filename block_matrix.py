# coding: utf-8

import itertools
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot as ssdot
from methoddispatch import methdispatch
import math_utils as mu

class block_mat_base(object):
    def __init__(self, shape_detail, block_shape):
        self._shape = (sum(shape_detail[0]),
                       sum(shape_detail[1]))
        self._shape_detail = shape_detail
        self._block_shape = block_shape

    @property
    def shape(self):
        return self._shape

    @property
    def shape_detail(self):
        return self._shape_detail

    @property
    def block_shape(self):
        return self._block_shape

    def __getitem__(self, key):
        return NotImplemented

    def __setitem__(self, key, values):
        return NotImplemented

    def __matmul__(self, other):
        return NotImplemented

    def __rmatmul__(self, other):
        return NotImplemented

    def dot(self, other):
        return self.__matmul__(other)

    def sum(self, axis=0):
        return NotImplemented

    def inv(self):
        return NotImplemented

    @property
    def T(self):
        return NotImplemented

    def tomatrix(self, **kwargs):
        return NotImplemented


class block_mat(block_mat_base):
    def __init__(self, matrices):
        self._matrices = matrices
        row_shapes = [*itertools.repeat(None, len(matrices))]
        col_shapes = [*itertools.repeat(None, len(matrices[0]))]
        for i, j in itertools.product(range(len(row_shapes)), range(len(col_shapes))):
            if matrices[i][j] is None:
                continue
            # TODO: check dims
            if row_shapes[i] is None:
                row_shapes[i] = matrices[i][j].shape[0]
            if col_shapes[j] is None:
                col_shapes[j] = matrices[i][j].shape[1]

        super().__init__((tuple(row_shapes), tuple(col_shapes)),
                         (len(row_shapes), len(col_shapes)))

    def __getitem__(self, key):
        # TODO: check key
        # TODO: slice?
        i, j = key
        return self._matrices[i][j]

    @methdispatch
    def __matmul__(self, other):
        return NotImplemented

    @__matmul__.register(np.ndarray)
    def _(self, other):
        # TODO: check dims
        if other.ndim == 1:
            res = np.zeros(self.shape[0])
        elif other.ndim == 2:
            res = np.zeros((self.shape[0], other.shape[1]))
        else:
            return NotImplemented

        start_row = 0
        for i in range(self.block_shape[0]):
            end_row = start_row + self.shape_detail[0][i]
            start_other_row = 0
            for k in range(self.block_shape[1]):
                end_other_row = start_other_row + self.shape_detail[1][k]
                if self[i, k] is None:
                    start_other_row = end_other_row
                    continue
                res[start_row:end_row, ] += ssdot(self[i, k],
                                                  other[start_other_row:end_other_row, ])
                start_other_row = end_other_row
            start_row = end_row

        return res

    @methdispatch
    def __rmatmul__(self, other):
        return NotImplemented #TODO: ndarray @ block_mat

    def tomatrix(self, **kwargs):
        return sparse.bmat(self._matrices, **kwargs)

@block_mat.__matmul__.register(block_mat)
def _(self, other):
    # TODO: check dims
    res = [[val for val in itertools.repeat(None, other.block_shape[1])]
           for _ in range(self.block_shape[0])]
    for i, j in itertools.product(range(self.block_shape[0]), range(other.block_shape[1])):
        for k in range(self.block_shape[1]):
            if self[i, k] is None or other[k, j] is None:
                continue
            res[i][j] = ssdot(self[i, k], other[k, j])

    return block_mat(res)



class block_sym_mat(block_mat):
    def __init__(self, matrices, upper_tri=True):
        self._matrices = matrices
        self._upper_tri = upper_tri
        row_shapes = [*itertools.repeat(None, len(matrices))]
        col_shapes = [*itertools.repeat(None, len(matrices[0]))]
        # TODO: check if diagonal block is symmetric
        assert(row_shapes == col_shapes)
        need_swap = lambda i, j: (upper_tri and i > j) or (not upper_tri and i < j)
        for i, j in itertools.product(range(len(row_shapes)), range(len(col_shapes))):
            k, l = i, j
            if need_swap(i, j):
                k, l = j, i
            if matrices[k][l] is None:
                continue
            # TODO: check dims
            if row_shapes[i] is None:
                row_shapes[i] = matrices[k][l].shape[1 if need_swap(i, j) else 0]
            if col_shapes[j] is None:
                col_shapes[j] = matrices[k][l].shape[0 if need_swap(i, j) else 1]

        block_mat_base.__init__(self, (tuple(row_shapes), tuple(col_shapes)),
                                (len(row_shapes), len(col_shapes)))

    @property
    def upper_tri(self):
        return self._upper_tri

    def __getitem__(self, key):
        # TODO: check key
        # TODO: slice?
        i, j = key
        if (self.upper_tri and i > j) or (not self.upper_tri and i < j):
            i, j = j, i
            return self._matrices[i][j].T if self._matrices[i][j] is not None else None
        else:
            return self._matrices[i][j]

    @property
    def T(self):
        return self

    def tomatrix(self, **kwargs):
        return NotImplemented # TODO

class block_diag_mat(block_mat_base):
    def __init__(self, matrices):
        # TODO: check dims
        self._matrices = matrices
        super().__init__((tuple([mat.shape[0] for mat in matrices]),
                          tuple([mat.shape[1] for mat in matrices])),
                         (len(matrices), len(matrices)))

    def __getitem__(self, key):
        # TODO: check key
        # TODO: slice?
        i, j = key
        return self._matrices[i] if i == j else None

    @methdispatch
    def __matmul__(self, other):
        return NotImplemented

    @__matmul__.register(np.ndarray)
    def _(self, other):
        # TODO: check dims
        if other.ndim == 1:
            res = np.zeros(self.shape[0])
        elif other.ndim == 2:
            res = np.zeros((self.shape[0], other.shape[1]))
        else:
            return NotImplemented

        start_row = 0
        for i in range(self.block_shape[0]):
            end_row = start_row + self.shape_detail[1][i]
            if self[i, i] is None:
                start_row = end_row
                continue
            res[start_row:end_row, ] = ssdot(self[i, i],
                                              other[start_row:end_row, ])
            start_row = end_row
        return res

    @__matmul__.register(block_mat)
    def _(self, other):
        # TODO: check dims
        res = [[val for val in itertools.repeat(None, other.block_shape[1])]
               for _ in range(self.block_shape[0])]
        for i, j in itertools.product(range(self.block_shape[0]), range(other.block_shape[1])):
            if self[i, i] is None or other[i, j] is None:
                continue
            res[i][j] = ssdot(self[i, i], other[i, j])

        return block_mat(res)

    @methdispatch
    def __rmatmul__(self, other): #TODO: ndarray @ block_diag_mat
        return NotImplemented

    @__rmatmul__.register(block_mat)
    def _(self, other):
        # TODO: check dims
        res = [[val for val in itertools.repeat(None, self.block_shape[1])]
               for _ in range(other.block_shape[0])]
        for i, j in itertools.product(range(other.block_shape[0]), range(self.block_shape[1])):
            if other[i, j] is None or self[i, j] is None:
                continue
            res[i][j] = ssdot(other[i, j], self[j, j])

        return block_mat(res)

    def inv(self):
        return block_diag_mat([mu.inv(mat) for mat in self._matrices])

    def tomatrix(self, **kwargs):
        return sparse.block_diag(self._matrices, **kwargs)

@block_diag_mat.__matmul__.register(block_diag_mat)
def _(self, other):
    # TODO: check dims
    res = [*itertools.repeat(None, self.block_shape[0])]
    for i in range(self.block_shape[0]):
        if self[i, i] is None or other[i, i] is None:
            continue
        res[i] = ssdot(self[i, i], other[i, i])

    return block_diag_mat(res)
