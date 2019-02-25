# coding: utf-8
import os
import contextlib
import functools
from more_itertools import windowed
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix
from joblib import Parallel, delayed
import dask
from dask_utils import tree_agg
from dask.diagnostics import ProgressBar
from corpus_io import corpus_segment

def construct_matrix(seq, context_window, vocab_size):
    window_size = 2 * context_window + 1
    offsets = np.array([i * vocab_size for i in range(2 * context_window)])
    offsets = np.insert(offsets, context_window, 0)

    row_idx = []
    col_idx = []
    for tokens in windowed(seq, window_size):
        tokens = np.atleast_1d(tokens) + offsets
        row_idx.append(tokens[context_window])
        col_idx.append(np.delete(tokens, context_window))

    tVV_diag = np.bincount(row_idx, minlength=vocab_size)
    row_idx = np.repeat(row_idx, 2 * context_window)
    col_idx = np.concatenate(col_idx)

    tVC = csr_matrix((np.repeat(1, len(row_idx)), (row_idx, col_idx)),
                     shape=(vocab_size, vocab_size * context_window * 2))
    tCC_diag = np.bincount(col_idx, minlength=vocab_size * context_window * 2)

    return tVC, tVV_diag, tCC_diag


def _add_elem(x, y):
    res = []
    for x_elem, y_elem in zip(x, y):
        res.append(x_elem + y_elem)

    return tuple(res)

def construct_matrix_dask(path, context_window, vocab_size, tokens2idx,
                          n_workers=-1, n_partition=200, verbose=False):
    if n_workers == -1:
        n_workers = os.cpu_count()

    corpus_info = os.stat(path)
    read_size = corpus_info.st_size // n_partition
    n_chunk = corpus_info.st_size // (1024 ** 3) + 1 # NOTE: size in GB + 1
    partitions = np.arange(n_partition) * read_size
    context_mgr = ProgressBar() if verbose else contextlib.suppress()
    load = functools.partial(corpus_segment, path, max_len=read_size, convert=tokens2idx, join=False)

    tVC = csr_matrix((vocab_size, vocab_size * context_window * 2),
                     dtype=np.dtype('int64'))
    tVV_diag = np.zeros(vocab_size)
    tCC_diag = np.zeros(vocab_size * context_window * 2)
    # NOTE: multiprocessing backend fails to handle too large data
    # see https://bugs.python.org/issue17560
    for phase, chunk in enumerate(np.array_split(partitions, n_chunk)):
        if verbose:
            chunk_size_kb = (len(chunk) * read_size) // 1024
            print('%d / %d chunk, %d KB:'%(phase + 1, n_chunk, chunk_size_kb))
        load_delayed = [dask.delayed(load)(seek_pos=pos) for pos in chunk]
        construct_delayed = [dask.delayed(construct_matrix)(seq, context_window, vocab_size) for seq in load_delayed]
        construct_dask = tree_agg(construct_delayed, _add_elem)

        with context_mgr:
            _tVC, _tVV_diag, _tCC_diag = construct_dask.compute(
                num_workers=n_workers, scheduler='processes')
        if verbose:
            tVC_size = _tVC.data.nbytes + _tVC.indptr.nbytes + _tVC.indices.nbytes
            tVV_size = _tVV_diag.nbytes
            tCC_size = _tCC_diag.nbytes
            total_size_kb = (tVC_size + tVV_size + tCC_size) // 1024
            print('\treturn %d KB'%total_size_kb)
        tVC += _tVC
        tVV_diag += _tVV_diag
        tCC_diag += _tCC_diag

    return tVC, tVV_diag, tCC_diag

def _caption2idx(caption, context_window, offsets):
    window_size = 2 * context_window + 1
    row_idx_doc = []
    col_idx_doc = []
    for tokens in windowed(caption, window_size):
        tokens = np.atleast_1d(tokens) + offsets
        row_idx_doc.append(tokens[context_window])
        col_idx_doc.append(np.delete(tokens, context_window))

    return row_idx_doc, np.concatenate(col_idx_doc)

def _idx2csc(arr, minlength, pad_idx=None):
    count = np.bincount(arr, minlength=minlength)
    if pad_idx is not None:
        count = np.delete(count, pad_idx)
    return csc_matrix(np.atleast_2d(count).T)

def construct_doc_matrix(caps_seqs, context_window, vocab_size, pad_idx=None, n_jobs=1, verbose=False):
    offsets = np.array([i * vocab_size for i in range(2 * context_window)])
    offsets = np.insert(offsets, context_window, 0)

    verbose_level = 10 if verbose else 0
    idx_list = Parallel(n_jobs=n_jobs, verbose=verbose_level)(
        delayed(_caption2idx)(caption, context_window, offsets) for caption in caps_seqs)
    tVJD = sparse.hstack(
        Parallel(n_jobs=n_jobs, verbose=verbose_level)(
            delayed(_idx2csc)(idx[0], vocab_size, pad_idx) for idx in idx_list),
        format='csr')
    tCJD = sparse.hstack(
        Parallel(n_jobs=n_jobs, verbose=verbose_level)(
            delayed(_idx2csc)(idx[1], vocab_size * context_window * 2) for idx in idx_list),
        format='csr')
    row_idx = np.concatenate([idx[0] for idx in idx_list])
    col_idx = np.concatenate([idx[1] for idx in idx_list])
    tVV_diag = np.bincount(row_idx, minlength=vocab_size)
    tCC_diag = np.bincount(col_idx, minlength=vocab_size * context_window * 2)
    row_idx = np.repeat(row_idx, 2 * context_window)
    tVC = csr_matrix((np.repeat(1, len(row_idx)), (row_idx, col_idx)),
                     shape=(vocab_size, vocab_size * context_window * 2))

    if pad_idx is not None:
        tVV_diag = np.delete(tVV_diag, pad_idx)
        tVC = sparse.vstack([tVC[:pad_idx, ], tVC[(pad_idx + 1):, ]], format='csr')

    return tVC, tVV_diag, tCC_diag, tVJD, tCJD
