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

def MixDataDescriptive(network, guild):
    """
    Compute descriptive statistics for network and guild over multiple periods.
    Returns a pandas DataFrame similar to MATLAB table.
    """
    n = guild[0].shape[0]
    m = guild[0].shape[1]
    period = len(guild)

    # Initialize DataFrame
    stat = pd.DataFrame({
        'period': np.arange(1, period+1),
        'net_density': np.nan,
        'net_max_deg': np.nan,
        'guild_max': np.nan,
        'guild_min': np.nan,
        'guild_change': np.nan
    })

    # Compute guild indices
    guild_idx = []
    for p in range(period):
        G = guild[p].toarray() if issparse(guild[p]) else guild[p]
        guild_idx.append(np.argmax(G, axis=1))

    # Compute statistics for each period
    for p in range(period):
        Y = network[p].toarray() if issparse(network[p]) else network[p]
        G = guild[p].toarray() if issparse(guild[p]) else guild[p]

        stat.at[p, 'net_density'] = np.count_nonzero(Y) / (n * (n - 1))
        stat.at[p, 'net_max_deg'] = np.max(np.sum(Y, axis=1))
        stat.at[p, 'guild_max'] = np.max(np.sum(G, axis=0))
        stat.at[p, 'guild_min'] = np.min(np.sum(G, axis=0))

        if p > 0:
            stat.at[p, 'guild_change'] = np.count_nonzero(guild_idx[p-1] != guild_idx[p]) / n

    # Round to 3 significant figures
    stat.iloc[:, 1:] = stat.iloc[:, 1:].apply(lambda x: np.round(x, 3))

    print(stat)
    return stat
