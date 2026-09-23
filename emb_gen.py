import scipy.sparse as sp
import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed
import pickle
from tqdm import tqdm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
from torch_geometric.nn import Node2Vec
from Model import PeerIModel


def get_args():
    parser = argparse.ArgumentParser('nneGen')
    parser.add_argument('--model', type=str, help='econ model', default='pi') # nf: network formation, pi: peer influence
    parser.add_argument('--epoch', type=int, help='total nodes', default=2)
    parser.add_argument('--z_dim', type=int, help='embedding dim', default=32)
    parser.add_argument('--gpu', type=int, help='gpu', default=0)
    return parser.parse_args()


def Test_error_summary_emb(y_hat, basket_theta, label_name, figure=True, table=True):
    y_hat = y_hat.detach().cpu().numpy()
    err = y_hat - basket_theta[:, -1]

    if figure:
        fig, axes = plt.subplots(2, 1, figsize=(3*2, 8))  # approximate figure size
        ax = axes[1]
        ax.hist(err, bins=30, color=[0, 0.4, 0.7], edgecolor='none')
        ax.set_xlabel(f"$\\hat{{{label_name[-1]}}}-{label_name[-1]}$")
        ax.set_ylim([0, ax.get_ylim()[1]*1.1])

        ax = axes[0]
        ax.scatter(basket_theta[:, -1], err + basket_theta[:, -1], s=3, color=[0, 0.4, 0.7], marker='.')
        ax.set_xlabel(f"${label_name[-1]}$")
        ax.set_ylabel(f"$\\hat{{{label_name[-1]}}}$")

        ax.plot([basket_theta[:, -1].min(), basket_theta[:, -1].max()],
                [basket_theta[:, -1].min(), basket_theta[:, -1].max()], 'r')  # reference line y=x
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.show()

    # Print result table
    if table:
        bias = [f"{np.mean(err):.3f} ({np.std(err)/np.sqrt(len(err)):.1f})" for j in range(len(err))]
        rmse = [f"{np.sqrt(np.mean(err**2)):.3f} ({0.5/np.sqrt(np.mean(err**2))*np.std(err**2)/np.sqrt(len(err)):.1f})"
                for j in range(len(err))]
        mean_SD = ["nan"]*len(err)

        result = pd.DataFrame({
            'bias': bias,
            'rmse': rmse,
            'mean_SD': mean_SD
        }, index=label_name)
        print("Test results:", result)

def simulate_moment(econmodel, theta):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    network_simul, feature_simul = econmodel.get_data(theta)
    A = sp.coo_matrix(network_simul)
    edge_index = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
    embAlgo = Node2Vec(
        edge_index,
        embedding_dim=args.z_dim,
        walks_per_node=10,
        walk_length=20,
        context_size=10,
        p=1,
        q=1,
        num_negative_samples=1,
    )

    optimizer = torch.optim.Adam(embAlgo.parameters(), lr=0.01)
    loader = embAlgo.loader(batch_size=128, shuffle=True)

    for epoch in range(args.epoch):
        embAlgo.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = embAlgo.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        z = embAlgo() 
    
    z = z.detach().cpu().numpy() # （N, z_dim)
    Z = pd.DataFrame(z)
    col_name = ['z'+str(i) for i in range(args.z_dim)]
    Z.columns = col_name
    behavior = pd.DataFrame({'y0': feature_simul[0].squeeze(-1), 'y1': feature_simul[1].squeeze(-1), 'T': feature_simul[2].squeeze(-1)})
    behavior = pd.concat([behavior,Z], axis=1)
    string = ''
    for j in col_name:
        string = string + ' + ' + j
    res = smf.ols(formula='y1 ~ T' + string, data=behavior).fit()

    return res.params['T']

def emb_gen(data):
    
    basket_theta = data['basket_theta']
    network = data['network']
    feature = data['feature']
    lb = data['lb']
    ub = data['ub']
    label_name = data['label_name']

    n = network.shape[0]
    econmodel = PeerIModel(n)
    R = basket_theta.shape[0]

    results = Parallel(n_jobs=-1, verbose=0)(
                  delayed(simulate_moment)(econmodel, basket_theta[t, :]) for t in tqdm(range(R))
    )     # for parallel computing
    test_pred = torch.tensor(results, dtype=torch.float32)
    Test_error_summary_emb(test_pred, basket_theta, label_name)





if __name__ == "__main__":
    args = get_args()
    with open('training_set.pkl', 'rb') as f:  # data from "set_up.py"
        data = pickle.load(f)
    emb_gen(data)

