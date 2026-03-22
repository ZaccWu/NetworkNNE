import numpy as np
import pandas as pd
from scipy.sparse import issparse

def PeerDataDescriptive(network):
    """
    Compute descriptive statistics for network and guild over multiple periods.
    Returns a pandas DataFrame similar to MATLAB table.
    """
    n = network[0].shape[0]
    period = len(network)

    # Initialize DataFrame
    stat = pd.DataFrame({
        'period': np.arange(1, period+1),
        'net_density': np.nan,
        'net_max_deg': np.nan,
    })
    # Compute statistics for each period
    for p in range(period):
        Y = network[p].toarray() if issparse(network[p]) else network[p]

        stat.at[p, 'net_density'] = np.count_nonzero(Y) / (n * (n - 1))
        stat.at[p, 'net_max_deg'] = np.max(np.sum(Y, axis=1))
    # Round to 3 significant figures
    stat.iloc[:, 1:] = stat.iloc[:, 1:].apply(lambda x: np.round(x, 3))
    print(stat)
    return stat
