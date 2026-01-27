import numpy as np
from scipy.sparse import csr_matrix, issparse


def Clustering_global(Y):
    if issparse(Y):
        Y2_sparse = Y @ Y   # using sparse matrix for light computation
        Y2_sparse.setdiag(0)

        numerator = Y2_sparse.multiply(Y).sum()
        denominator = Y2_sparse.sum()

        Y2_return = Y2_sparse

    else:
        raise ValueError("Y is not a sparse matrix")

    globalcoef = float(numerator) / float(denominator) if denominator != 0 else 0.0
    if np.isnan(globalcoef):
        globalcoef = 0.0

    return globalcoef, Y2_return