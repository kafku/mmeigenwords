# coding: utf-8

import numpy as np
from scipy import linalg
from sklearn.utils.extmath import randomized_svd
import math_utils as mu
from src import wgs

__EPS = 1e-6

# (Modified) Gram-Schmidt with W-inner product
# see Algorithm 3 in "Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application to computing Karhunen-Loeve expansion`"
# Saibaba+ 2011
def winner_gs(X, W, inplace=False):
    if inplace:
        Y = X
    else:
        Y = X.copy()
    WY = W @ Y
    wgs.wgs_core_inplace(Y, WY)

    if inplace == False:
        return Y

# randomized method for solving a generalized Hermitian eigenvalue problem
# see also,
# 1. Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application to computing Karhunen-Loeve expansion (Algorithm 6)
# 2. SVD and it's Application to Generalized Eigenvalue Problems
def randomized_ghep(H, G, n_components, n_oversamples=20, n_iter=3, method=1, return_evec=False):
    k = n_components + n_oversamples
    if method == 1:
        Q = np.random.randn(H.shape[0], k)
        G_inv = mu.inv(G)
        for i in range(n_iter):
            # calculating HQ every time is inefficient but memory saving
            # when H is a large block_sym_mat
            HQ = H @ Q
            Q = G_inv @ HQ
            winner_gs(Q, G, inplace=True)
    elif method == 2:
        Q, sv, _ = randomized_svd(G, n_components + n_oversamples,
                                  n_oversamples=0, n_iter=n_iter)
        Q = Q / np.sqrt(sv)
    else:
        raise ValueError()

    eig_values, eig_vectors = linalg.eigh(Q.T @ (H @ Q))
    if return_evec:
        return eig_values[-n_components:], Q @ eig_vectors[:, -n_components:], eig_vectors[:, -n_components:]
    else:
        return eig_values[-n_components:], Q @ eig_vectors[:, -n_components:]
