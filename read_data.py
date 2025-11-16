import numpy as np
from scipy import sparse
import scipy.io as sio

# ----------------------------
# LOAD
# ----------------------------
data = sio.loadmat('data_raw.mat')  # 假设 MATLAB 的结构体变量名保持不变
data1 = data['data1']
data2 = data['data2']
data3 = data['data3']
data5 = data['data5']

n, m = data1['outcome_guild'].shape

# ----------------------------
# BUILD NETWORK AND GUILD
# ----------------------------
network = [
    data1['state_dep_friend'],
    data2['outcome_friend'],
    data3['outcome_friend'],
    data5['outcome_friend']
]

guild = [
    data1['state_dep_guild'],
    data2['outcome_guild'],
    data3['outcome_guild'],
    data5['outcome_guild']
]

period = len(network)

# ----------------------------
# PROCESS NETWORK
# ----------------------------
for p in range(period):
    Y = network[p].copy()
    # 对角线置零
    np.fill_diagonal(Y, 0)

    # 检查是否对称且只包含0/1
    unique_vals = np.unique(Y)
    if not np.allclose(Y, Y.T) or not np.array_equal(unique_vals, [0, 1]):
        raise ValueError("wrong data type.")

    # 转为逻辑稀疏矩阵
    network[p] = sparse.csr_matrix(Y.astype(bool))

# ----------------------------
# PROCESS GUILD
# ----------------------------
for p in range(period):
    L = guild[p].copy()

    # 检查每行是否只有一个1
    if not np.all(np.sum(L, axis=1) == 1):
        raise ValueError("wrong data type.")

    # 转为逻辑稀疏矩阵
    guild[p] = sparse.csr_matrix(L.astype(bool))

# ----------------------------
# SAVE
# ----------------------------
# 保存为 Python 可读格式，也可以保存为 MATLAB .mat 文件
sio.savemat('data_real.mat', {'network': network, 'guild': guild})
