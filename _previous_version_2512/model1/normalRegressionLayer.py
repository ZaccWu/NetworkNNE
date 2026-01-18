# normalRegressionLayer.py
"""
Converted from normalRegressionLayer.m
In MATLAB this might implement a custom regression layer; in Python we provide a PyTorch module.
"""
import torch
import torch.nn as nn

from Positive_transform import Positive_transform

class NormalRegressionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Matlab 中没有额外属性，这里保持空 constructor

    def forward_loss(self, Y, T):
        """
        Python equivalent of:
            function loss = forwardLoss(layer, Y, T)
        """

        k = Y.shape[0] // 2
        n = Y.shape[1]

        U = Y[:k, :]
        S = Positive_transform(Y[k:2*k, :])
        X = T[:k, :]

        squared_err = 2 * torch.log(S) + ((U - X) / S) ** 2

        # sum all elements, then divide by n
        loss = squared_err.sum() / n

        return loss