# coding: utf-8

import os
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.extmath import safe_sparse_dot as ssdot
from corpus_io import read_corpus, count_word_dask
from construct_matrix import construct_matrix, construct_matrix_dask
import math_utils as mu
import block_matrix as bm
import named_array as na
from algorithm import randomized_ghep
from sammlung import DefaultOrderedDict
from embed_base import WordEmbedBase
from context import Context

class MMEigenwords(WordEmbedBase):
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
        self.context = None
        self.ev = None
        self.image_mapping = None
        self.mean_image = None
        self.mapped_image = None
        self.train_image_id = np.array([], dtype=str)

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


    def load_corpus(self, path, images, Wvi, context=False, use_dask=False,
                    n_worker=-1, n_chunk=200, verbose=False): #TODO: implement dask option
        verboseprint = lambda x: print(x) if verbose else None
        verboseprint('Fixing Wvi')
        mandatory_words = set()
        if isinstance(Wvi, na.NamedArrBase):
            mandatory_words = set(Wvi.names[0])
        elif isinstance(Wvi, (pd.DataFrame, pd.SparseDataFrame)):
            mandatory_words = set(Wvi.index)
        else:
            raise ValueError("Wvi must be one of named_array.NamedArray, pandas.DataFrame, or pandas.SparseDataFrame.")

        if self.word_dict is None:
            if use_dask:
                word_count = count_word_dask(
                    path, self.vocab_size, n_workers=n_worker,
                    n_partition=n_chunk, verbose=verbose)
            else:
                word_count = Counter()
                word_count.update(read_corpus(path, max_len=50000000))

            mandatory_words.intersection_update(word_count.keys())
            vocab2idx = DefaultOrderedDict(int, append_missing=False)
            vocab2idx['<OOV>'] = 0
            vocab2idx.update(
                (word, i + 1) for i, word in enumerate(x[0] for x in word_count.most_common(self.vocab_size - 1)))
            vocab2idx.update(
                (word, i + self.vocab_size) for i, word in enumerate(mandatory_words.difference(vocab2idx.keys())))
            self.word_dict = vocab2idx

            # update Wvi
            missing_words = set(vocab2idx.keys()).difference(mandatory_words)
            if isinstance(Wvi, na.NamedArrBase):
                missing_Wvi = sparse.csr_matrix((len(missing_words), Wvi.shape[1]), dtype=Wvi.dtype)
                new_Wvi = Wvi[list(mandatory_words), :] # error without ":"
                missing_Wvi = na.NamedArray(missing_Wvi, axis_names=[missing_words, None])
                new_Wvi = na.vstack([new_Wvi, missing_Wvi], format='csr')
                new_Wvi = new_Wvi[list(vocab2idx.keys()), ]
                self.train_image_id = np.array(Wvi.names[1], dtype=str) # FIXME: what if it's missing?
            elif isinstance(Wvi, pd.DataFrame):
                missing_Wvi = np.zeros((len(missing_words), Wvi.shape[1]), dtype=Wvi.dtype)
                new_Wvi = Wvi.loc[list(mandatory_words)]
                missing_Wvi = pd.DataFrame(missing_Wvi, index=missing_words)
                new_Wvi = pd.concat([new_Wvi, missing_Wvi]).loc[vocab2idx.keys()]
                new_Wvi = new_Wvi.values
                self.train_image_id = np.array(Wvi.columns, dtype=str)
            elif isinstance(Wvi, pd.SparseDataFrame):
                missing_Wvi = sparse.csr_matrix((len(missing_words), Wvi.shape[1]), dtype=Wvi.dtype)
                new_Wvi = Wvi.loc[list(mandatory_words)]
                missing_Wvi = pd.SparseDataFrame(missing_Wvi, index=missing_words)
                new_Wvi = pd.concat([new_Wvi, missing_Wvi]).loc[vocab2idx.keys()]
                new_Wvi = new_Wvi.to_coo().tocsr()
                self.train_image_id = np.array(Wvi.columns, dtype=str)
            self.vocab_size = len(vocab2idx)


            # show info
            verboseprint('  Vocab size: %d'%self.vocab_size)

        self.corpus_path.append(path)
        self.train(path if use_dask else read_corpus(path, max_len=50000000, convert=self._tokens2idx),
                   images, new_Wvi, context,
                   use_dask, n_worker, n_chunk, verbose)

    def train(self, tokens, images, Wvi, context=False,
              use_dask=False, n_worker=-1, n_chunk=200, verbose=False):
        verboseprint = lambda x: print(x) if verbose else None

        verboseprint('Constructing matrices...')
        if verbose and use_dask == False:
            tokens = tqdm(tokens)

        if use_dask:
            tVC, tVV_diag, tCC_diag = construct_matrix_dask(
                tokens, self.window_size, self.vocab_size,
                self._tokens2idx, n_worker, n_chunk, verbose)
        else:
            tVC, tVV_diag, tCC_diag = construct_matrix(tokens, self.window_size, self.vocab_size)

        self.mean_image = np.mean(images, axis=0, keepdims=True)
        Xvis = images - self.mean_image

        verboseprint('Squashing...')
        tVC, tVV_diag, tCC_diag = self._squash_arrays(tVC, tVV_diag, tCC_diag)

        verboseprint('Preparing arrays...')
        n_tags_per_vocab = mu.sum(Wvi, axis=1)
        tVWviXvis = ssdot(ssdot(sparse.diags(tVV_diag), Wvi), Xvis)
        Gvv_diag = tVV_diag + tVV_diag * n_tags_per_vocab
        Gvis = Xvis.T @ ssdot(sparse.diags(ssdot(Wvi.T, tVV_diag)), Xvis)

        verboseprint('Calculating word vectors...')
        H = bm.block_sym_mat([[None, tVC, tVWviXvis],
                              [None, None, None],
                              [None, None, None]])
        G = bm.block_diag_mat(
            [sparse.diags(Gvv_diag), sparse.diags(tCC_diag), Gvis])
        eigenvalues, A = randomized_ghep(H, G,
                                         n_components=self.dim,
                                         n_oversamples=self.dim + self.oversampling,
                                         n_iter=self.n_iter)

        self.ev = eigenvalues[::-1]
        self._set_keyedvector('wv', self.word_dict.keys(), self.dim,
                              vec=A[:self.vocab_size, ::-1])
        self.image_mapping = A[-Xvis.shape[1]:, ::-1]
        if context:
            self.context = Context(A[self.vocab_size:-Xvis.shape[1], ::-1],
                                   len(self.word_dict), self.window_size)

    def map_image(self, images):
        if isinstance(images, pd.DataFrame):
            image_ids = images.index.tolist()
            images = images.values
        elif isinstance(images, na.NamedArrBase):
            image_ids = images.names[0]

        self._set_keyedvector('mapped_image', image_ids, self.dim,
                              vec=normalize((images - self.mean_image) @ self.image_mapping))

    def most_similar(self, pos_word=[], neg_word=[], pos_img=[], neg_img=[],
                         target="word", topn=10):
        positive = []
        negative = []
        positive.extend(self.wv.word_vec(x, use_norm=True) for x in pos_word)
        positive.extend(self.mapped_image.word_vec(x, use_norm=True) for x in pos_img)
        negative.extend(self.wv.word_vec(x, use_norm=True) for x in neg_word)
        negative.extend(self.mapped_image.word_vec(x, use_norm=True) for x in neg_img)
        if target == "word":
            return self.wv.most_similar(positive=positive, negative=negative, topn=topn)
        elif target == "image":
            return self.mapped_image.most_similar(positive=positive, negative=negative, topn=topn)
        else:
            raise ValueError("invalid target. target must be one of word or image")

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
        self._save_np_params(dir_path, param_list=[
            'ev', 'image_mapping', 'mean_image', 'train_image_id'])
        if save_context and self.context is not None:
            np.savez(os.path.join(dir_path, 'context_param.npz'),
                     context=self.context._window_vec)

    @classmethod
    def load_model(self, dir_path, load_context=False):
        model = super().load_model(dir_path)
        if load_context:
            try:
                with np.load(os.path.join(dir_path, 'context_param.npz')) as data:
                    model.context = Context(data['context'], len(model.word_dict),
                                            model.window_size)
            except IOError:
                print('Failed to load context_param.npz')

        model.word_dict = DefaultOrderedDict(int, append_missing=False)
        model.word_dict.update((word, i) for i, word in enumerate(model.wv.index2word))
        return model
