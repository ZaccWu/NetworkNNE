import numpy as np
from scipy.sparse import csr_matrix, tril, isspmatrix
from scipy.sparse import triu as sparse_triu
from Clustering_global import Clustering_global
from scipy.sparse.csgraph import shortest_path

# calculate data moments of Peer Model DGP
def PeerFeatureMoments(network, feature):
    n = network[0].shape[0]
    period = len(network)
    draw1 = (csr_matrix(np.random.rand(n, n) < 0.03)).astype(bool) # 生成随机稀疏矩阵 (元素有3%概率为正)
    draw1 = draw1 + draw1.T  # 对称
    moment1_list, moment2_list = [], []

    # --- moment1: period-wise statistics ---
    for p in range(period):
        Y = network[p]
        deg = np.array(Y.sum(axis=1).A1).flatten()  # 求和+转成 长度 n 的一维 numpy array

        moment1 = np.array([
            np.mean(deg),
            np.var(deg),
            np.mean(feature.squeeze(-1)), # add peer features
            np.var(feature.squeeze(-1)),  # add peer features
            Clustering_global(Y)[0],
        ])
        moment1_list.append(moment1)

    # --- moment2: cross-period statistics ---
    for p in range(1, period):
        Y0 = network[p - 1].toarray() if isspmatrix(network[p - 1]) else network[p - 1]
        Y = network[p].toarray() if isspmatrix(network[p]) else network[p]
        deg = np.sum(Y0, axis=1)
        log_deg = np.log1p(deg)
        log_deg_sum = log_deg[:, None] + log_deg[None, :] # 每个节点对的联合强度：log(1+di)+log(1+dj)

        adj = csr_matrix(Y0)
        dist = shortest_path(adj, method='D', directed=False)  # shape = (n, n)
        dist = 1 - 1 / (1 + dist)
        
        fea_cov = np.dot(feature, feature.T)

        # lower triangular indices
        i1 = np.tril_indices(n, k=-1) # 获得n*n的矩阵的左下三角的索引(bool)
        D1 = np.column_stack([Y0[i1], dist[i1], log_deg_sum[i1], fea_cov[i1]]) # 获得对应索引的元素值

        A2 = Y.astype(int)
        B2 = draw1.toarray().astype(int) if isspmatrix(draw1) else draw1.astype(int) # B2: 随机对称矩阵
        i2 = np.tril((A2-B2)<0, k=-1) # 下三角的索引(bool)：相当于随机选取3%的yijt=0的边
        D2 = np.column_stack([Y0[i2], dist[i2], log_deg_sum[i2], fea_cov[i2]])

        # TODO: check i1是整个网络的下三角，i2是实际网络和3%正边随机网络不一致的节点对

        moment2 = Stat(D1, D2)
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

