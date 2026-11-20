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
    parser.add_argument('--epoch', type=int, help='total epoch', default=200) # default: 200
    parser.add_argument('--z_dim', type=int, help='embedding dim', default=64) 
    parser.add_argument('--history', type=bool, help='plot history (z=1)', default=False) 
    return parser.parse_args()



def Test_error_summary_emb(y_hat, basket_theta, label_name, figure=True, table=True):
    if args.z_dim > 1:
        label_test = basket_theta[:,:-1]
    else:
        label_test = basket_theta
    n, k = label_test.shape # num of samples, num of output_dim
    m = len(label_name)     # num of parameters
    if args.z_dim>1:
        m = m-1
    err = y_hat[:, :m].detach().cpu().numpy() - label_test[:, :m]

    if figure:
        p = min(10, m)
        fig, axes = plt.subplots(2, p, figsize=(3*p*2, 8))  # approximate figure size

        for j in range(p):  # Histograms
            ax = axes[1, j]
            ax.hist(err[:, j], bins=30, color=[0, 0.4, 0.7], edgecolor='none')
            ax.set_xlabel(f"$\\hat{{{label_name[j]}}}-{label_name[j]}$")
            ax.set_ylim([0, ax.get_ylim()[1]*1.1])

        for j in range(p):  # Scatter plots
            ax = axes[0, j]
            ax.scatter(label_test[:, j], err[:, j] + label_test[:, j], s=3, color=[0, 0.4, 0.7], marker='.')
            ax.set_xlabel(f"${label_name[j]}$")
            ax.set_ylabel(f"$\\hat{{{label_name[j]}}}$")

            ax.plot([label_test[:, j].min(), label_test[:, j].max()],
                    [label_test[:, j].min(), label_test[:, j].max()], 'r')  # reference line y=x
            ax.set_box_aspect(1)
        plt.tight_layout()
        plt.show()

    # Print result table
    if table:
        bias = [f"{np.mean(err[:, j]):.3f} ({np.std(err[:, j])/np.sqrt(n):.1f})" for j in range(m)]
        rmse = [f"{np.sqrt(np.mean(err[:, j]**2)):.3f} ({0.5/np.sqrt(np.mean(err[:, j]**2))*np.std(err[:, j]**2)/np.sqrt(n):.1f})"
                for j in range(m)]
        mean_SD = ["nan"]*m
        result = pd.DataFrame({
            'bias': bias,
            'rmse': rmse,
            'mean_SD': mean_SD
        }, index=label_name)

        print("Test results:", result)

# def Test_error_summary_emb(y_hat, basket_theta, label_name, figure=True, table=True):
#     y_hat = y_hat.detach().cpu().numpy()
#     err = y_hat - basket_theta[:, 3]

#     if figure:
#         fig, axes = plt.subplots(2, 1, figsize=(3*2, 8))  # approximate figure size
#         ax = axes[1]
#         ax.hist(err, bins=30, color=[0, 0.4, 0.7], edgecolor='none')
#         ax.set_xlabel(f"$\\hat{{{label_name[3]}}}-{label_name[3]}$")
#         ax.set_ylim([0, ax.get_ylim()[1]*1.1])

#         ax = axes[0]
#         ax.scatter(basket_theta[:, 3], err + basket_theta[:, 3], s=3, color=[0, 0.4, 0.7], marker='.')
#         ax.set_xlabel(f"${label_name[3]}$")
#         ax.set_ylabel(f"$\\hat{{{label_name[3]}}}$")

#         ax.plot([basket_theta[:, 3].min(), basket_theta[:, 3].max()],
#                 [basket_theta[:, 3].min(), basket_theta[:, 3].max()], 'r')  # reference line y=x
#         ax.set_box_aspect(1)
#         plt.tight_layout()
#         plt.show()

def plot_history(history, label_name):
    """
    history: simulate_moment 返回的 dict
    label_name: set_up 里的参数名列表，顺序固定：
        [beta_0, delta_f(=beta_w), alpha_0, alpha_p, alpha_w]
    """
    true = history['true_theta']
    ep = np.arange(len(history['beta_0']))

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    ax = axes.flatten()

    # 与 theta 索引的对应关系（PeerI 模型）
    spec = [
        (0, 'beta_0',  history['beta_0'],  0),
        (1, 'beta_w',  history['beta_w'],  1),
        (2, 'alpha_0', history['alpha_0'], 2),
        (3, 'alpha_p', history['alpha_p'], 3),
    ]
    for i, name, vals, ti in spec:
        ax[i].plot(ep, vals, lw=1.2, label='estimate')
        ax[i].axhline(true[ti], color='r', ls='--', lw=1.2, label='true')
        ax[i].set_title(f'{name}   (true={true[ti]:.3f})', fontsize=10)
        ax[i].set_xlabel('epoch')
        ax[i].legend(fontsize=8)

    # alpha_w：z_dim=1 时是标量；否则画每条分量
    aw = np.array(history['alpha_w'])
    if aw.ndim == 2 and aw.shape[1] > 1:
        for k in range(aw.shape[1]):
            ax[4].plot(ep, aw[:, k], lw=1.0, label=f'alpha_w[{k}]')
    else:
        ax[4].plot(ep, aw.reshape(-1), lw=1.2, label='estimate')
    ax[4].axhline(true[4], color='r', ls='--', lw=1.2, label='true')
    ax[4].set_title(f'alpha_w   (true={true[4]:.3f})', fontsize=10)
    ax[4].set_xlabel('epoch')
    ax[4].legend(fontsize=8)

    # 三个 loss 一起看
    ax[5].plot(ep, history['loss'],     label='total')
    ax[5].plot(ep, history['loss_net'], label='net (BCE)')
    ax[5].plot(ep, history['loss_out'], label='out (MSE)')
    ax[5].set_yscale('log')
    ax[5].set_title('losses (log)', fontsize=10)
    ax[5].set_xlabel('epoch')
    ax[5].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # 额外：omega 的尺度变化（识别性诊断用）
    fig2, ax2 = plt.subplots(1, 2, figsize=(7, 3))
    om_std  = np.array(history['omega_std']).reshape(len(ep), -1)
    om_mean = np.array(history['omega_mean']).reshape(len(ep), -1)
    ax2[0].plot(ep, om_std,  label='omega std')
    ax2[0].set_title('omega std over time'); ax2[0].set_xlabel('epoch'); ax2[0].legend(fontsize=8)
    ax2[1].plot(ep, om_mean, label='omega mean')
    ax2[1].set_title('omega mean over time'); ax2[1].set_xlabel('epoch'); ax2[1].legend(fontsize=8)
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

    history = {
        'beta_0': [], 'beta_w': [],
        'alpha_0': [], 'alpha_p': [],
        'alpha_w': [],
        'loss': [], 'loss_net': [], 'loss_out': [],
        'omega_std': [], 'omega_mean': [],
    }

    for epoch in range(epoch):
        model.train()
        optimizer.zero_grad()
        
        z_norm = (model.omega - model.omega.mean(0)) / (model.omega.std(0) + 1e-8)
        
        D = torch.cdist(z_norm, z_norm, p=1) 
        V = model.beta_0 - model.beta_w * D
        A_pred = torch.sigmoid(V) 
        loss_net = F.binary_cross_entropy(A_pred, A_dense) * 10
        
        peer_avg = torch.mv(A_dense, y0) / (degree + 1e-8)
        y1_pred = model.alpha_0 + (z_norm @ model.alpha_w) + model.alpha_p * peer_avg
        loss_out = F.mse_loss(y1_pred, y1)
        
        loss = loss_out + loss_net 
        
        loss.backward()
        optimizer.step()

        history['beta_0'].append(model.beta_0.item())
        history['beta_w'].append(model.beta_w.item())
        history['alpha_0'].append(model.alpha_0.item())
        history['alpha_p'].append(model.alpha_p.item())
        history['alpha_w'].append(model.alpha_w.detach().cpu().numpy().copy())
        history['loss'].append(loss.item())
        history['loss_net'].append(loss_net.item())
        history['loss_out'].append(loss_out.item())
        history['omega_std'].append(model.omega.detach().cpu().numpy().std(0).copy())
        history['omega_mean'].append(model.omega.detach().cpu().numpy().mean(0).copy())

    if args.history:
        history['true_theta'] = np.asarray(theta, dtype=float)
        return history
    elif args.z_dim==1:
        return [
            model.beta_0.item(),
            model.beta_w.item(),
            model.alpha_0.item(),
            model.alpha_p.item(),
            model.alpha_w.item(),
        ]
    else:
        return [
            model.beta_0.item(),
            model.beta_w.item(),
            model.alpha_0.item(),
            model.alpha_p.item()
        ]

def emb_gen(data):
    basket_theta = data['basket_theta']
    network = data['network']
    label_name = data['label_name']

    n = network.shape[0]
    econmodel = PeerIModel(n)

    R = basket_theta.shape[0]
    results = Parallel(n_jobs=16, backend="multiprocessing", verbose=10)(
        delayed(simulate_moment)(econmodel, basket_theta[t, :], args.z_dim, args.epoch) 
        for t in range(R)
    )
    test_pred = torch.tensor(results, dtype=torch.float32)
    print(test_pred.shape, basket_theta[:R, :].shape)

    Test_error_summary_emb(test_pred, basket_theta[:R, :], label_name)

    # R = 1
    # results = Parallel(n_jobs=-1, backend="multiprocessing", verbose=10)(
    #     delayed(simulate_moment)(econmodel, basket_theta[t, :], args.z_dim, args.epoch) 
    #     for t in range(R)
    # )
    # if args.history:
    #     for t, hist in enumerate(results):
    #         print(f"\n=== plotting trajectory for sample t={t} "
    #             f"(alpha_p final={hist['alpha_p'][-1]:.4f}, "
    #             f"true={hist['true_theta'][3]:.4f}) ===")
    #         plot_history(hist, label_name)



if __name__ == "__main__":
    args = get_args()
    with open('training_set_small.pkl', 'rb') as f:
        data = pickle.load(f)
    emb_gen(data)