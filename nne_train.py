
import numpy as np
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Positive_transform import Positive_transform
from sklearn.preprocessing import StandardScaler
from normalRegressionLayer import NormalRegressionLayer
import pickle
from Test_error_summary import Test_error_summary
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser('nneTrain')
# neural network algorithm/training settings
parser.add_argument('--num_nodes', type=int, help='layer width', default=128)   # 128
parser.add_argument('--batch_size', type=int, help='training sample batch', default=256)    # 256
parser.add_argument('--max_epochs', type=int, help='training epoches', default=100)         # 100
parser.add_argument('--initial_lr', type=int, help='initial learning rate', default=0.01)   # 0.01

# display settings
parser.add_argument('--disp_test_summary', type=bool, help='display test summary', default=True)
parser.add_argument('--display_fig', type=bool, help='display figure', default=True)
parser.add_argument('--disp_iter', type=bool, help='display iteration', default=True)
parser.add_argument('--learn_standard_error', type=bool, help='learn standard error', default=True)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


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
    X = T[:, :k]
    squared_err = 2 * torch.log(S) + ((U - X) / S) ** 2
    # sum all elements, then divide by n
    loss = squared_err.sum() / n
    return loss

def nne_train(data):
    # load training/validation/real data
    input_train, label_train = data['input_train'], data['label_train']
    input_test, label_test = data['input_test'], data['label_test']
    input_real = data['input_real']
    lb, ub = data['lb'], data['ub']
    label_name = data['label_name']

    R_train, R_test = input_train.shape[0], input_test.shape[0] # num of train/test 'samples'
    R = R_train + R_test
    M, L = input_train.shape[1], label_train.shape[1]

    if args.learn_standard_error:   # sd as labels
        label_train = np.hstack([label_train, np.zeros((R_train, L))])
        label_test = np.hstack([label_test, np.zeros((R_test, L))])
        output_dim = 2 * L
    else:
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
    if args.learn_standard_error:
        criterion = forward_loss
    else:
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

        if args.disp_iter:
            print(f"Epoch {epoch + 1}/{args.max_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {loss_va.item():.4f}")


    if args.disp_test_summary:
        net.eval()
        with torch.no_grad():
            test_preds = net(torch.tensor(input_test, dtype=torch.float32))
            Test_error_summary(torch.tensor(input_test, dtype=torch.float32), label_test, label_name, net, figure=args.display_fig, table=1)

    # Estimate on original data
    net.eval()
    with torch.no_grad():
        temp = net(torch.tensor(input_real.squeeze(0), dtype=torch.float32)).numpy()

    # 截断 theta
    theta = np.clip(temp[:L], lb, ub)

    # 正向变换标准误（如果学习）
    if args.learn_standard_error:
        se = Positive_transform(temp[L:2 * L])

    print(theta)
    print(se)


if __name__ == "__main__":
    with open('training_set_gen.pkl', 'rb') as f:  # data from "set_up.py"
        data = pickle.load(f)
    nne_train(data)