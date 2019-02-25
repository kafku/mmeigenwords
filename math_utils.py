# coding: utf-8

from functools import singledispatch
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from named_array import NamedArrBase

## sum
@singledispatch
def sum(x, *args, **kwargs):
    sum_op = getattr(x, 'sum', None)
    if callable(sum_op):
        return x.sum(*args, **kwargs)

    return NotImplemented

sum.register(np.ndarray, np.sum)

@sum.register(sparse.base.spmatrix)
def _(x, *args, **kwargs):
    return np.squeeze(np.asarray(x.sum(*args, **kwargs)))

@sum.register(NamedArrBase)
def _(x, *args, **kwargs):
    return sum(x._obj, *args, **kwargs)

## sqrt
@singledispatch
def sqrt(x):
    sqrt_op = getattr(x, 'sqrt', None)
    if callable(sqrt_op):
        return x.sqrt()

    return NotImplemented

sqrt.register(np.ndarray, np.sqrt)

## log1p
@singledispatch
def log1p(x, *args, **kwargs):
    log1p_op = getattr(x, 'log1p', None)
    if callable(log1p_op):
        return x.log1p(*args, **kwargs)

    return NotImplemented

log1p.register(np.ndarray, np.log1p)

## inv
@singledispatch
def inv(x):
    inv_op = getattr(x, 'inv', None)
    if callable(inv_op):
        return x.inv()

    return NotImplemented

inv.register(np.ndarray, np.linalg.inv)
inv.register(sparse.csr_matrix, sparse.linalg.inv)
inv.register(sparse.csc_matrix, sparse.linalg.inv)

@inv.register(sparse.dia.dia_matrix)
def _(x):
    return sparse.diags(1.0 / np.squeeze(x.data))
