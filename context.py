# coding: utf-8

from more_itertools import windowed
import numpy as np
import scipy.linalg
from sklearn.preprocessing import normalize

class Context(object):
    def __init__(self, vectors, vocab_size, window_size):
        self._window_vec = vectors
        self._vocab_size = vocab_size
        self._window_size = window_size
        self._offsets = np.array([i * vocab_size for i in range(2 * window_size)])

    def window2vec(self, token_idx, return_sum=True):
        idx_with_offset = np.atleast_1d(token_idx) + self._offsets
        if return_sum:
            return self._window_vec[idx_with_offset, ].sum(axis=0)
        else:
            return self._window_vec[idx_with_offset, ]

    def mapping(self, token_idx, norm=False):
        # TODO: check length
        arr = []
        for window in windowed(token_idx, self._window_size * 2 + 1):
            arr.append(
                self.window2vec(window[:self._window_size] + window[self._window_size + 1:]))
        arr = np.vstack(arr)
        if norm:
            return normalize(arr)
        else:
            return arr

    def left_context(self, token_idx, norm=False):
        assert(len(token_idx) <= self._window_size)
        idx_with_offset = np.atleast_1d(token_idx) + \
            self._offsets[(self._window_size - len(token_idx)):self._window_size]
        arr = self._window_vec[idx_with_offset, ].sum(axis=0)
        if norm:
            return arr / scipy.linalg.norm(arr)
        else:
            return arr

    def right_context(self, token_idx, norm=False):
        assert(len(token_idx) <= self._window_size)
        idx_with_offset = np.atleast_1d(token_idx) + \
            self._offsets[self._window_size:(self._window_size + len(token_idx))]
        arr = self._window_vec[idx_with_offset, ].sum(axis=0)
        if norm:
            return arr / scipy.linalg.norm(arr)
        else:
            return arr
