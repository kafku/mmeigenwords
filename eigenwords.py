# coding: utf-8

import os
from collections import Counter
from tqdm import tqdm
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot
from corpus_io import read_corpus, count_word_dask
from construct_matrix import construct_matrix, construct_matrix_dask
import math_utils as mu
from sammlung import DefaultOrderedDict
from embed_base import WordEmbedBase
from context import Context

class EigenwordsOSCCA(WordEmbedBase):
    def __init__(self, vocab_size=10000, window_size=4, dim=300,
                 oversampling=20, n_iter=3, squash='sqrt',
                 word_dict=None):
        self.corpus_path = []
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.dim = dim
        self.oversampling = oversampling
        self.n_iter = n_iter
        self.squash = squash
        self.wv = None
        self.sv = None
        self.context = None

        if squash == 'log1p':
            self._squash = mu.log1p
        elif squash == 'sqrt':
            self._squash = mu.sqrt
        else:
            raise Exception('invaldi squash option')

        if word_dict is not None:
            self.word_dict = word_dict
        else:
            self.word_dict = None

    def load_corpus(self, path, context=False, use_dask=False,
                    n_worker=-1, n_chunk=200, verbose=False):
        if self.word_dict is None:
            if use_dask:
                word_count = count_word_dask(
                    path, self.vocab_size, n_workers=n_worker,
                    n_partition=n_chunk, verbose=verbose)
            else:
                word_count = Counter()
                word_count.update(read_corpus(path, max_len=50000000))

            vocab2idx = DefaultOrderedDict(int, append_missing=False)
            vocab2idx['<OOV>'] = 0
            vocab2idx.update(
                (word, i + 1) for i, word in enumerate(x[0] for x in word_count.most_common(self.vocab_size - 1)))
            self.word_dict = vocab2idx


        self.corpus_path.append(path)
        self.train(path if use_dask else read_corpus(path, max_len=50000000, convert=self._tokens2idx),
                   context, use_dask, n_worker, n_chunk, verbose)

    def train(self, tokens, context=False, use_dask=False,
              n_worker=-1, n_chunk=200, verbose=False):
        verboseprint = lambda x: print(x) if verbose else None

        verboseprint('Constructing matrices...')
        if verbose and use_dask == False:
            tokens = tqdm(tokens)

        if use_dask:
            tVC, tVV_diag, tCC_diag = construct_matrix_dask(
                tokens, self.window_size, self.vocab_size,
                self._tokens2idx, n_worker, n_chunk, verbose)
        else:
            tVC, tVV_diag, tCC_diag = construct_matrix(
                tokens, self.window_size, self.vocab_size)

        verboseprint('Squashing...')
        tVC, tVV_diag, tCC_diag = self._squash_arrays(tVC, tVV_diag, tCC_diag)

        verboseprint('Calculating word vectors...')
        S = sparse.diags(1.0 / np.sqrt(tVV_diag)) @ tVC @ sparse.diags(1.0 / np.sqrt(tCC_diag))
        U, singular_values, R = randomized_svd(S, self.dim,
                                               n_oversamples=self.oversampling,
                                               n_iter=self.n_iter)
        self.sv = singular_values[::-1]
        self._set_keyedvector('wv', self.word_dict.keys(), self.dim,
                              vec=safe_sparse_dot(sparse.diags(1.0 / np.sqrt(tVV_diag)), U)[:, ::-1])
        if context:
            self.context = Context(safe_sparse_dot(sparse.diags(1.0 / np.sqrt(tCC_diag)), R.T)[:, ::-1],
                                   len(self.word_dict), self.window_size)


    def _save_meta_hook(self, model_meta):
        model_meta['init_param'].update({
            'oversampling': self.oversampling,
            'n_iter': self.n_iter,
            'squash': self.squash
        })
        model_meta['non_init_param'].update({
            'corpus_path': self.corpus_path
        })
        return model_meta

    def save_model(self, dir_path, save_context=False, **kwargs):
        super().save_model(dir_path, **kwargs)
        self._save_np_params(dir_path, param_list=['sv'])
        if save_context and self.context is not None:
            np.savez(os.path.join(dir_path, 'context_param.npz'),
                     context=self.context._window_vec)

    @classmethod
    def load_model(cls, dir_path, load_context=False):
        model = super().load_model(dir_path)
        model.word_dict = DefaultOrderedDict(int, append_missing=False)
        model.word_dict.update((word, i) for i, word in enumerate(model.wv.index2word))
        if load_context:
            try:
                with np.load(os.path.join(dir_path, 'context_param.npz')) as data:
                    model.context = Context(data['context'], len(model.word_dict),
                                            model.window_size)
            except IOError:
                print('Failed to load context_param.npz')
        return model
