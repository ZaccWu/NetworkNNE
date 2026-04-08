import pickle
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from Positive_transform import Positive_transform
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd

def getTrainArgs():
    parser = argparse.ArgumentParser('nneTrain')
    # neural network algorithm/training settings
    parser.add_argument('--num_nodes', type=int, help='layer width', default=128)   # 128
    parser.add_argument('--batch_size', type=int, help='training sample batch', default=32)    # 256
    parser.add_argument('--max_epochs', type=int, help='training epoches', default=100)         # 100
    parser.add_argument('--initial_lr', type=int, help='initial learning rate', default=0.01)   # 0.01

    # display settings
    parser.add_argument('--disp_test_summary', type=bool, help='display test summary', default=True)
    parser.add_argument('--display_fig', type=bool, help='display figure', default=True)

    try:
        args = parser.parse_args()  # Pass empty list to ignore kernel args
        return args
    except:
        args, _ = parser.parse_known_args()
        return args

def set_train_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)

def forward_loss(Y, T): # y_pred, y_true
    #print(Y.shape)
    #print(T.shape)
    k = Y.shape[1] // 2
    n = Y.shape[0]
    U = Y[:, :k]
    S = torch.tensor(Positive_transform(Y[:, k:2*k].detach().numpy()))
    S = S+1e-8
    X = T[:, :k]
    squared_err = 2 * torch.log(S) + ((U - X) / S) ** 2
    # sum all elements, then divide by n
    loss = squared_err.sum() / n
    return loss



def nne_train_specify(args, input_train, input_test, input_real):
    label_train = data['label_train']
    label_test = data['label_test']
    lb, ub = data['lb'], data['ub']
    label_name = data['label_name']

    R_train, R_test = input_train.shape[0], input_test.shape[0] # num of train/test 'samples'
    R = R_train + R_test
    M, L = input_train.shape[1], label_train.shape[1]


    output_dim = L
    scaler = StandardScaler()
    input_train, input_test = scaler.fit_transform(input_train), scaler.fit_transform(input_test)
    input_real = scaler.fit_transform(input_real.reshape(1, -1))

    # Dataset and DataLoader
    train_dataset = TensorDataset(torch.tensor(input_train, dtype=torch.float32),
                                    torch.tensor(label_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    net = NeuralNet(M, args.num_nodes, output_dim)


    device = torch.device("cpu")
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.initial_lr)

    # Optional: learning rate scheduler similar to MATLAB piecewise schedule
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1) # 40, 0.1

    for epoch in range(args.max_epochs):
        net.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        scheduler.step()

    # Validation / Test summary (optional)
        net.eval()
        with torch.no_grad():
            test_preds = net(torch.tensor(input_test, dtype=torch.float32))
            loss_va = criterion(test_preds, torch.tensor(label_test, dtype=torch.float32))
            #print(f"Epoch {epoch + 1}/{args.max_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {loss_va.item():.4f}")

    # table results
    net.eval()
    with torch.no_grad():
        nts, kts = label_test.shape # num of samples, num of output_dim
        mts = len(label_name)     # num of parameters
        y_hat = net(torch.tensor(input_test, dtype=torch.float32))
        y_hat = y_hat.detach().cpu().numpy()
        err = y_hat[:, :mts] - label_test[:, :mts]
        # bias = [f"{np.mean(err[:, j]):.3f} ({np.std(err[:, j])/np.sqrt(nts):.1f})" for j in range(mts)]
        # rmse = [f"{np.sqrt(np.mean(err[:, j]**2)):.3f} ({0.5/np.sqrt(np.mean(err[:, j]**2))*np.std(err[:, j]**2)/np.sqrt(nts):.1f})"
        #         for j in range(mts)]
        # result = pd.DataFrame({
        #     'bias': bias,
        #     'rmse': rmse,
        # }, index=label_name)
        bias = [np.mean(err[:, j]) for j in range(mts)]
        rmse = [np.sqrt(np.mean(err[:, j]**2)) for j in range(mts)]
        result = pd.DataFrame({
            'bias': bias,
            'rmse': rmse,
        }, index=label_name)
    #print("Test results:", result)

    # Estimate on original data
    net.eval()
    with torch.no_grad():
        temp = net(torch.tensor(input_real.squeeze(0), dtype=torch.float32)).numpy()
    # 截断 theta
    theta = np.clip(temp[:L], lb, ub)
    #print(theta)
    return result, theta


def get_mask_scheme(mask_col):
    mask_train = np.ones(data['input_train'].shape[1], dtype=bool)
    mask_test = np.ones(data['input_test'].shape[1], dtype=bool)
    mask_real = np.ones(data['input_real'].shape[0], dtype=bool)
    
    #mask[[2, 5, 10]] = False
    mask_train[mask_col], mask_test[mask_col], mask_real[mask_col] = False, False, False
    return data['input_train'][:, mask_train], data['input_test'][:, mask_test], data['input_real'][mask_real]


if __name__ == "__main__":
    with open('results/results-20260322/lambda-3~-2, alpha2~12/training_set_gen.pkl', 'rb') as f:  # data from "set_up.py"
        data = pickle.load(f)
    args = getTrainArgs()
    # input: 20 moment1 (4 period, 5 statistics), 28 moment2 (3-period-average, D1: 4+10, D2: 4+10)


    # systematic experiments
    results_all = {}
    mask_scheme = {'full': [],
                    'no moment1': [int(i) for i in np.arange(0,20)],
                    'no moment2': [int(i) for i in np.arange(20,48)],
                    'no D1': [int(i) for i in np.arange(20,34)],
                    'no D2': [int(i) for i in np.arange(34,48)],
                    'no deg': [0, 1, 5, 6, 10, 11, 15, 16],
                    'no fea mean': [2, 7, 12, 17],
                    'no fea var': [3, 8, 13, 18],
                    'no clu glo': [4, 9, 14, 19],
                    'no D1 cov': [int(i) for i in np.arange(24,34)],
                    'no D2 cov': [int(i) for i in np.arange(38,48)],
                    'no Y0': [20, 24, 25, 26, 27, 34, 38, 39, 40, 41],
                    'no dist': [21, 25, 28, 29, 30, 35, 39, 42, 43, 44],
                    'no logdegsum': [22, 26, 29, 31, 32, 36, 40, 43, 45, 46],
                    'no feacov': [23, 27, 30, 32, 33, 37, 41, 44, 46, 47],}

    for k, v in mask_scheme.items():
        print(k, v)
        input_train, input_test, input_real = get_mask_scheme([v])
        result, theta = nne_train_specify(args, input_train, input_test, input_real)

        # define the checking parameter in interest
        res_all = [result.loc[r'\lambda_f', 'bias'], result.loc[r'\lambda_f', 'rmse'], theta[0],
            result.loc[r'\alpha_f', 'bias'], result.loc[r'\alpha_f', 'rmse'], theta[1],
            result.loc[r'\beta_f', 'bias'], result.loc[r'\beta_f', 'rmse'], theta[2],
            result.loc[r'\delta_f', 'bias'], result.loc[r'\delta_f', 'rmse'], theta[3],
            result.loc[r'\theta_f', 'bias'], result.loc[r'\theta_f', 'rmse'], theta[4],
            result.loc[r'\gamma_f', 'bias'], result.loc[r'\gamma_f', 'rmse'], theta[5],
            result.loc[r'\tau', 'bias'], result.loc[r'\tau', 'rmse'], theta[6]]

        res_label_name = ['lambda_f_bias', 'lambda_f_rmse', 'lambda_f_real_pred',
            'alpha_f_bias', 'alpha_f_rmse', 'alpha_f_real_pred',
            'beta_f_bias', 'beta_f_rmse', 'beta_f_real_pred',
            'delta_f_bias', 'delta_f_rmse', 'delta_f_real_pred',
            'theta_f_bias', 'theta_f_rmse', 'theta_f_real_pred',
            'gamma_f_bias', 'gamma_f_rmse', 'gamma_f_real_pred',
            'tau_bias', 'tau_rmse', 'tau_real_pred']

        results_all[k] = res_all

    results_all = pd.DataFrame(results_all, index=res_label_name)
    print(results_all)
    results_all.to_csv('results_all.csv')