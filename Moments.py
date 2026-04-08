import numpy as np
from scipy.sparse import csr_matrix, tril, isspmatrix
from scipy.sparse import triu as sparse_triu
from Clustering_global import Clustering_global
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye
from scipy.sparse.linalg import expm



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
    if period == 1:
        Y = network[p].toarray() if isspmatrix(network[p]) else network[p]
        deg = np.sum(Y, axis=1)
        log_deg = np.log1p(deg)
        log_deg_sum = log_deg[:, None] + log_deg[None, :] # 每个节点对的联合强度：log(1+di)+log(1+dj)
        adj = csr_matrix(Y)
        dist = shortest_path(adj, method='D', directed=False)  # shape = (n, n)
        dist = 1 - 1 / (1 + dist)
        fea_cos = cosine_similarity(feature)

        # lower triangular indices
        A = Y.astype(int)
        i1 = np.tril(A>0, k=-1)
        i2 = np.tril(A<=0, k=-1)
        D1 = np.column_stack([Y[i1], dist[i1], log_deg_sum[i1], fea_cos[i1]]) # 获得对应索引的元素值
        D2 = np.column_stack([Y[i2], dist[i2], log_deg_sum[i2], fea_cos[i2]]) # 获得对应索引的元素值
        moment2 = Stat(D1, D2)
        moment2_list.append(moment2)

    else:
        for p in range(1, period):
            Y0 = network[p - 1].toarray() if isspmatrix(network[p - 1]) else network[p - 1]
            Y = network[p].toarray() if isspmatrix(network[p]) else network[p]
            deg = np.sum(Y0, axis=1)
            log_deg = np.log1p(deg)
            log_deg_sum = log_deg[:, None] + log_deg[None, :] # 每个节点对的联合强度：log(1+di)+log(1+dj)

            adj = csr_matrix(Y0)
            dist = shortest_path(adj, method='D', directed=False)  # shape = (n, n)
            dist = 1 - 1 / (1 + dist)
            fea_cos = cosine_similarity(feature)

            # lower triangular indices
            A = Y.astype(int)
            i1 = np.tril(A>0, k=-1)
            D1 = np.column_stack([Y0[i1], dist[i1], log_deg_sum[i1], fea_cos[i1]]) # 获得对应索引的元素值
            B2 = draw1.toarray().astype(int) if isspmatrix(draw1) else draw1.astype(int) # B2: 随机对称矩阵
            i2 = np.tril((A-B2)<0, k=-1) # 下三角的索引(bool)：相当于随机选取3%的yijt=0的边
            D2 = np.column_stack([Y0[i2], dist[i2], log_deg_sum[i2], fea_cos[i2]])

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
        u = np.mean(data, axis=0)   # 返回列维度均值矩阵，这里长为4
        v = np.cov(data, rowvar=False)  # 返回列维度的协方差矩阵（D*D），这里是4*4
        v_flat = v[np.triu_indices(v.shape[0])] # 返回协方差矩阵上三角展平（包括对角线，即特征方差）
        output_list.append(np.hstack([u, v_flat])) # feature维度：4+10
    output = np.hstack(output_list)
    return output

