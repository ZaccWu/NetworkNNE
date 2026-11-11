import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

from Model import PeerIModel

def get_args():
    parser = argparse.ArgumentParser('nneGen')
    parser.add_argument('--epoch', type=int, help='total epoch', default=50) # 直接优化，可增加 epoch
    parser.add_argument('--z_dim', type=int, help='embedding dim', default=64) 
    return parser.parse_args()


def Test_error_summary_emb(y_hat, basket_theta, label_name, figure=True, table=True):
    y_hat = y_hat.detach().cpu().numpy()
    err = y_hat - basket_theta[:, 3]

    if figure:
        fig, axes = plt.subplots(2, 1, figsize=(3*2, 8))  # approximate figure size
        ax = axes[1]
        ax.hist(err, bins=30, color=[0, 0.4, 0.7], edgecolor='none')
        ax.set_xlabel(f"$\\hat{{{label_name[3]}}}-{label_name[3]}$")
        ax.set_ylim([0, ax.get_ylim()[1]*1.1])

        ax = axes[0]
        ax.scatter(basket_theta[:, 3], err + basket_theta[:, 3], s=3, color=[0, 0.4, 0.7], marker='.')
        ax.set_xlabel(f"${label_name[3]}$")
        ax.set_ylabel(f"$\\hat{{{label_name[3]}}}$")

        ax.plot([basket_theta[:, 3].min(), basket_theta[:, 3].max()],
                [basket_theta[:, 3].min(), basket_theta[:, 3].max()], 'r')  # reference line y=x
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.show()

# 核心优化 1：抛弃 Node2Vec，直接用自由 Embedding 替代潜在特征 \omega
class End2EndModel(nn.Module):
    def __init__(self, num_nodes, z_dim):
        super(End2EndModel, self).__init__()
        # 直接定义可学习的潜在特征 \omega
        self.omega = nn.Parameter(torch.randn(num_nodes, z_dim))
        
        # 网络形成参数 (式1)
        self.beta_0 = nn.Parameter(torch.tensor(0.0))
        self.beta_w = nn.Parameter(torch.tensor(1.0))  
        
        # 结果模型参数 (式2)
        self.alpha_0 = nn.Parameter(torch.tensor(0.0))
        self.alpha_w = nn.Parameter(torch.randn(z_dim)) 
        self.alpha_p = nn.Parameter(torch.tensor(0.0))  

def simulate_moment(econmodel, theta, z_dim, epoch):
    torch.set_num_threads(1)
    device = torch.device('cpu')
    
    network_simul, feature_simul = econmodel.get_data(theta)
    
    dense_A = network_simul.toarray() if sp.issparse(network_simul) else network_simul
    A_dense = torch.tensor(dense_A, dtype=torch.float32, device=device)
    A_dense = torch.maximum(A_dense, A_dense.t()) 
    
    N = A_dense.shape[0]
    
    y0 = torch.tensor(feature_simul[0].squeeze(-1), dtype=torch.float32, device=device)
    y1 = torch.tensor(feature_simul[1].squeeze(-1), dtype=torch.float32, device=device)
    
    model = End2EndModel(N, z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 计算图的度，用于同伴平均
    degree = A_dense.sum(1)
    
    for epoch in range(epoch):
        model.train()
        optimizer.zero_grad()
        
        z_norm = (model.omega - model.omega.mean(0)) / (model.omega.std(0) + 1e-8)
        
        D = torch.cdist(z_norm, z_norm, p=1) 
        V = model.beta_0 - model.beta_w * D
        A_pred = torch.sigmoid(V) 
        loss_net = F.binary_cross_entropy(A_pred, A_dense)
        
        peer_avg = torch.mv(A_dense, y0) / (degree + 1e-8)
        y1_pred = model.alpha_0 + (z_norm @ model.alpha_w) + model.alpha_p * peer_avg
        loss_out = F.mse_loss(y1_pred, y1)
        
        loss = loss_out + 1.0 * loss_net 
        
        loss.backward()
        optimizer.step()

    return model.alpha_p.item()

def emb_gen(data):
    basket_theta = data['basket_theta']
    network = data['network']
    label_name = data['label_name']

    n = network.shape[0]
    econmodel = PeerIModel(n)
    R = basket_theta.shape[0]

    n_jobs = 20
    print(f"Using {n_jobs} CPU cores for parallel computation.")

    # 并行执行
    # results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
    #     delayed(simulate_moment)(econmodel, basket_theta[t, :], args.z_dim, args.epoch) 
    #     for t in range(R)
    # )

    results = Parallel(n_jobs=16, backend="multiprocessing", verbose=10)(
        delayed(simulate_moment)(econmodel, basket_theta[t, :], args.z_dim, args.epoch) 
        for t in range(R)
    )
        
    test_pred = torch.tensor(results, dtype=torch.float32)
    Test_error_summary_emb(test_pred, basket_theta, label_name)

if __name__ == "__main__":
    args = get_args()
    with open('training_set_small.pkl', 'rb') as f:
        data = pickle.load(f)
    emb_gen(data)