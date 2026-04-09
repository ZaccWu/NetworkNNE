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
import shap
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")


def _run_shapley_analysis(net, input_train, input_test):
    # Limit sample size to keep SHAP runtime manageable.
    bg_n = min(len(input_train), max(1, 128))
    ev_n = min(len(input_test), max(1, 256))
    bg_idx = np.random.choice(len(input_train), size=bg_n, replace=False)
    ev_idx = np.random.choice(len(input_test), size=ev_n, replace=False)
    background = torch.tensor(input_train[bg_idx], dtype=torch.float32) # (128, 38)
    eval_x = torch.tensor(input_test[ev_idx], dtype=torch.float32)  # (256, 38)

    net.eval()
    with torch.no_grad():
        pred_dim = net(eval_x[:1]).shape[1] # 7

    target_idx = int(np.clip(0, 0, pred_dim - 1))  # 0
    explainer = shap.DeepExplainer(net, background)
    shap_values = explainer.shap_values(eval_x) # (256, 38, 7)

    if isinstance(shap_values, list):
        # Multi-output model: pick one output dimension to explain.
        shap_matrix = np.array(shap_values[target_idx])
    else:
        shap_matrix = np.array(shap_values)
        if shap_matrix.ndim == 3:
            shap_matrix = shap_matrix[:, :, target_idx]
    
    # shap_matrix: (256, 38)

    importance = np.mean(np.abs(shap_matrix), axis=0)
    order = np.argsort(-importance)
    top_k = min(max(1, 15), len(importance))

    print(f"\nSHAP feature importance (target output idx={target_idx}, top {top_k}):")
    for rank, feat_idx in enumerate(order[:top_k], start=1):
        print(f"{rank:>2}. x{feat_idx:<3} | mean(|SHAP|) = {importance[feat_idx]:.6g}")

    # 1. SHAP 摘要图
    shap.summary_plot(shap_values, eval_x, plot_type="bar", show=True)
    shap.summary_plot(shap_values, eval_x, plot_type="dot", show=True)


    # 2. 特征重要性条形图
    indices = order[:top_k]
    labels = [f"x{idx}" for idx in indices] # 如果有特征名称，可以替换这里
    values = importance[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(values)), values[::-1]) # 倒序排列，最大的在上面
    plt.yticks(range(len(values)), [f"x{idx}" for idx in indices[::-1]])
    plt.xlabel("Mean(|SHAP Value|)")
    plt.title(f"Top {top_k} Feature Importance")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 3. 依赖关系图
    top_feature_idx = order[0] 
    # eval_x[:, top_feature_idx] 取出该特征的列
    # shap_values[:, top_feature_idx] 取出该特征对应的 SHAP 值列
    shap.dependence_plot(
        top_feature_idx, # top_feature_idx
        shap_values[:, :, 0], # parameter at position 0
        eval_x.numpy()  # shape (256, 38)
    )

    #  4. 单样本力图
    # 选取 eval_x 中的第一个样本进行解释
    sample_idx = 0
    sample_data = eval_x[sample_idx:sample_idx+1,:]
    sample_shap = shap_matrix[sample_idx:sample_idx+1,:]

    # base value（多输出时取 target_idx）
    base = explainer.expected_value
    if isinstance(base, (list, tuple, np.ndarray)):
        base_value = float(np.array(base).reshape(-1)[target_idx])
    else:
        base_value = float(base)


    shap.force_plot(
        base_value,          # 第一个参数必须是 base value
        sample_shap.flatten(),         # 该样本的 SHAP 向量 (38)
        sample_data.flatten(),         # 该样本特征 torch.Size([38])
        matplotlib=True,
    )
    
    # 5. 热力图
    # 为了可视化效果，通常只对最重要的特征画图
    top_k_features = 20
    top_indices = order[:top_k_features]

    # 提取子矩阵
    shap_matrix_subset = shap_matrix[:, top_indices]

    plt.figure(figsize=(12, 8))
    # 对样本进行聚类或排序（可选，这里按 SHAP 值总和排序以便观察）
    row_order = np.argsort(np.sum(np.abs(shap_matrix_subset), axis=1))[::-1]

    sns.heatmap(shap_matrix_subset[row_order], 
                cmap="RdBu_r", # 红蓝配色，红正蓝负
                center=0,
                xticklabels=[f"x{i}" for i in top_indices],
                yticklabels=False) # 样本太多时不显示 y 轴标签

    plt.xlabel("Features")
    plt.ylabel("Samples (Sorted)")
    plt.title(f"SHAP Values Heatmap (Top {top_k_features} Features)")
    plt.tight_layout()
    plt.show()


def getTrainArgs():
    parser = argparse.ArgumentParser('nneTrain')
    # neural network algorithm/training settings
    parser.add_argument('--num_nodes', type=int, help='layer width', default=128)   # 128
    parser.add_argument('--batch_size', type=int, help='training sample batch', default=32)    # 256
    parser.add_argument('--max_epochs', type=int, help='training epoches', default=10)         # 100
    parser.add_argument('--initial_lr', type=int, help='initial learning rate', default=0.01)   # 0.01

    # display settings
    parser.add_argument('--enable_shapley', type=bool, help='shapley value analysis', default=True)
    parser.add_argument('--disp_test_summary', type=bool, help='display test summary', default=True)
    parser.add_argument('--display_fig', type=bool, help='display figure', default=False) # True
    parser.add_argument('--disp_iter', type=bool, help='display iteration', default=True)
    parser.add_argument('--learn_standard_error', type=bool, help='learn standard error', default=False)

    try:
        args = parser.parse_args()  # Pass empty list to ignore kernel args
        return args
    except:
        parser.print_help()
        sys.exit(0)
    

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

def nne_train(data, args):
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

    if args.enable_shapley:
        _run_shapley_analysis(net, input_train, input_test)

    # Estimate on original data
    net.eval()
    with torch.no_grad():
        temp = net(torch.tensor(input_real.squeeze(0), dtype=torch.float32)).numpy()

    # 截断 theta
    theta = np.clip(temp[:L], lb, ub)
    print(theta)
    # 正向变换标准误（如果学习）
    if args.learn_standard_error:
        se = Positive_transform(temp[L:2 * L])
        print(se)


if __name__ == "__main__":
    args = getTrainArgs()
    #with open('results/results-20260409//training_set_gen_p2.pkl', 'rb') as f:
    with open('training_set_gen.pkl', 'rb') as f:  # data from "set_up.py"
        data = pickle.load(f)
    set_train_seed(101)
    nne_train(data, args)
