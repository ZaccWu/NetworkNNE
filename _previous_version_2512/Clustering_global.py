import numpy as np
from scipy.sparse import csr_matrix, issparse


def Clustering_global(Y):
    """
    Compute a global clustering coefficient as in MATLAB code:
        globalcoef = sum(Y^2 where Y==1) / sum(Y^2)
    Returns:
        globalcoef : float
        Y2 : Y squared with diagonal zeroed
    """
    if issparse(Y):
        Y = Y.toarray()

    n = Y.shape[0]
    Y2 = Y @ Y  # matrix square (n*n)
    np.fill_diagonal(Y2, 0)  # set diagonal to zero

    numerator = np.sum(Y2[Y == 1])
    denominator = np.sum(Y2)

    globalcoef = numerator / denominator if denominator != 0 else 0.0
    if np.isnan(globalcoef):
        globalcoef = 0.0

    return globalcoef, Y2
