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
        bias = [f"{np.mean(err[:, j]):.3f} ({np.std(err[:, j])/np.sqrt(nts):.1f})" for j in range(mts)]
        rmse = [f"{np.sqrt(np.mean(err[:, j]**2)):.3f} ({0.5/np.sqrt(np.mean(err[:, j]**2))*np.std(err[:, j]**2)/np.sqrt(nts):.1f})"
                for j in range(mts)]
        result = pd.DataFrame({
            'bias': bias,
            'rmse': rmse,
        }, index=label_name)
    print("Test results:", result)

    # Estimate on original data
    net.eval()
    with torch.no_grad():
        temp = net(torch.tensor(input_real.squeeze(0), dtype=torch.float32)).numpy()
    # 截断 theta
    theta = np.clip(temp[:L], lb, ub)
    print(theta)



if __name__ == "__main__":
    with open('results/results-20260322/lambda-3~-2, alpha2~12/training_set_gen.pkl', 'rb') as f:  # data from "set_up.py"
        data = pickle.load(f)
    args = getTrainArgs()
    # input: 20 moment1 (4 period, 5 statistics), 28 moment2 (3-period-average, D1: 4+10, D2: 4+10)

    print("Full")
    input_train = data['input_train']
    input_test = data['input_test']
    input_real = data['input_real']
    nne_train_specify(args, input_train, input_test, input_real)

    print("\n w/o moment 1:")
    input_train = data['input_train'][:,20:]
    input_test = data['input_test'][:,20:]
    input_real = data['input_real'][20:]
    nne_train_specify(args, input_train, input_test, input_real)

    print("\n w/o moment 2:")
    input_train = data['input_train'][:,:20]
    input_test = data['input_test'][:,:20]
    input_real = data['input_real'][:20]
    nne_train_specify(args, input_train, input_test, input_real)