import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, tril, isspmatrix
from scipy.sparse import triu as sparse_triu
from Clustering_global import Clustering_global

def Moments(network, guild):
    n = network[0].shape[0]
    m = guild[0].shape[1]
    period = len(guild)

    # 生成随机稀疏矩阵
    draw1 = (csr_matrix(np.random.rand(n, n) < 0.03)).astype(bool)
    draw1 = draw1 + draw1.T  # 对称
    draw2 = (csr_matrix(np.random.rand(n, m) < 0.10)).astype(bool)

    moment1_list = []
    moment2_list = []

    # --- moment1: period-wise statistics ---
    for p in range(period):
        Y = network[p]
        deg = np.array(Y.sum(axis=1)).flatten()  # 度数

        moment1 = np.array([
            np.mean(deg),
            np.var(deg),
            Clustering_global(Y)
        ])
        moment1_list.append(moment1)

    # --- moment2: cross-period statistics ---
    for p in range(1, period):
        Y0 = network[p - 1].toarray() if isspmatrix(network[p - 1]) else network[p - 1]
        L0 = guild[p - 1].toarray() if isspmatrix(guild[p - 1]) else guild[p - 1]
        Y = network[p].toarray() if isspmatrix(network[p]) else network[p]
        L = guild[p].toarray() if isspmatrix(guild[p]) else guild[p]

        deg = np.sum(Y0, axis=1)
        log_deg = np.log1p(deg)
        log_deg_sum = log_deg[:, None] + log_deg[None, :]

        same_guild = L0 @ L0.T
        num_friend = Y0 @ L0
        fac_friend = num_friend / (deg[:, None] + 1e-12)
        siz_guild = np.ones((n, 1)) * np.sum(L0, axis=0)

        # distances using networkx
        G = nx.from_numpy_array(Y0)
        dist = np.zeros((n, n))
        for i in range(n):
            sp = nx.single_source_shortest_path_length(G, i)
            for j, d in sp.items():
                dist[i, j] = d
        dist = 1 - 1 / (1 + dist)

        linkage = dist @ L0 / (siz_guild + 1e-12)
        storage = log_deg[:, None].T @ L0 / (siz_guild + 1e-12)

        # lower triangular indices
        i1 = np.tril_indices(n, k=-1)
        D1 = np.column_stack([Y0[i1], dist[i1], log_deg_sum[i1], same_guild[i1]])

        i2 = np.tril((Y - draw1.toarray() if isspmatrix(draw1) else Y - draw1).astype(bool), k=-1)
        D2 = np.column_stack([Y0[i2], dist[i2], log_deg_sum[i2], same_guild[i2]])

        i3 = L.astype(bool)
        D3 = np.column_stack([L0[i3], fac_friend[i3], storage[i3], linkage[i3]])

        i4 = (L - draw2.toarray() if isspmatrix(draw2) else L - draw2) < 0
        D4 = np.column_stack([L0[i4], fac_friend[i4], storage[i4], linkage[i4]])

        moment2 = Stat(D1, D2, D3, D4)
        moment2_list.append(moment2)

    # 合并 moment1, moment2
    moment1_array = np.vstack(moment1_list)
    moment2_array = np.mean(np.vstack(moment2_list), axis=0)

    moment = np.hstack([moment1_array.flatten(), moment2_array])

    return moment


# ----------------------------
# Stat function
# ----------------------------
def Stat(*args):
    output_list = []
    for data in args:
        data = np.array(data)
        u = np.mean(data, axis=0)
        v = np.cov(data, rowvar=False)
        v_flat = v[np.triu_indices(v.shape[0])]
        output_list.append(np.hstack([u, v_flat]))
    output = np.hstack(output_list)
    return output

