import numpy as np
import torch
import torch.nn as nn

def Positive_transform(x):
    # softplus as a smooth positive transform (log(1+exp(x)))
    return np.log1p(np.exp(x))


class NormalRegressionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Matlab 中没有额外属性，这里保持空 constructor

    def forward_loss(self, Y, T):
        k = Y.shape[0] // 2
        n = Y.shape[1]
        U = Y[:k, :]
        S = Positive_transform(Y[k:2*k, :])
        X = T[:k, :]
        squared_err = 2 * torch.log(S) + ((U - X) / S) ** 2
        # sum all elements, then divide by n
        loss = squared_err.sum() / n
        return loss

def upper(input_matrix):
    input_matrix = np.asarray(input_matrix)
    k = input_matrix.shape[0]
    # 获取上三角布尔索引
    mask = np.triu(np.ones((k, k), dtype=bool))
    # 提取上三角元素
    output = input_matrix[mask]
    return output