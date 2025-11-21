
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Positive_transform import Positive_transform
import pickle
from Test_error_summary import Test_error_summary

# Settings
disp_test_summary = True
display_fig = True
disp_iter = True
learn_standard_error = True

# Data preparation
with open('training_set_gen.pkl', 'rb') as f:  # data from "set_up.py"
    data = pickle.load(f)

input_train = data['input_train']
label_train = data['label_train']
input_test = data['input_test']
label_test = data['label_test']
input_real = data['input_real']

lb, ub = data['lb'], data['ub']
label_name = data['label_name']

T_train = input_train.shape[0]
T_test = input_test.shape[0]
T = T_train + T_test
M = input_train.shape[1]
L = label_train.shape[1]

num_nodes = 128
batch_size = 256

if learn_standard_error:
    # 扩展标签以学习标准误
    label_train = np.hstack([label_train, np.zeros((T_train, L))])
    label_test = np.hstack([label_test, np.zeros((T_test, L))])
    output_dim = 2 * L
else:
    output_dim = L


max_epochs = 100
initial_lr = 0.01

# ----------------------------
# Dataset and DataLoader
# ----------------------------
train_dataset = TensorDataset(torch.tensor(input_train, dtype=torch.float32),
                              torch.tensor(label_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(torch.tensor(input_test, dtype=torch.float32),
                            torch.tensor(label_test, dtype=torch.float32))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# ----------------------------
# Define network
# ----------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learn_se=True):
        super().__init__()
        self.learn_se = learn_se
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


net = NeuralNet(M, num_nodes, output_dim, learn_standard_error)

# ----------------------------
# Training
# ----------------------------
device = torch.device("cpu")
net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=initial_lr)

# Optional: learning rate scheduler similar to MATLAB piecewise schedule
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

for epoch in range(max_epochs):
    net.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = net(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    scheduler.step()

    if disp_iter:
        print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item():.4f}")

# ----------------------------
# Validation / Test summary (optional)
# ----------------------------
if disp_test_summary:
    net.eval()
    with torch.no_grad():
        test_preds = net(torch.tensor(input_test, dtype=torch.float32))
    # 调用自定义 Test_error_summary 函数或写可视化/表格代码
    Test_error_summary(torch.tensor(input_test, dtype=torch.float32), label_test, label_name, net, figure=display_fig, table=1)

# ----------------------------
# Estimate on original data
# ----------------------------
net.eval()
with torch.no_grad():
    temp = net(torch.tensor(input_real, dtype=torch.float32)).numpy()

# 截断 theta
theta = np.clip(temp[:L], lb, ub)

# 正向变换标准误（如果学习）
if learn_standard_error:
    se = Positive_transform(temp[L:2 * L])
