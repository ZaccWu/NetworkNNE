import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.sparse import issparse
import shap
from .functional import Positive_transform



def PeerDataDescriptive(network):
    """
    Compute descriptive statistics for network and guild over multiple periods.
    Returns a pandas DataFrame similar to MATLAB table.
    """
    n = network[0].shape[0]
    period = len(network)

    # Initialize DataFrame
    stat = pd.DataFrame({
        'period': np.arange(1, period+1),
        'net_density': np.nan,
        'net_max_deg': np.nan,
    })
    # Compute statistics for each period
    for p in range(period):
        Y = network[p].toarray() if issparse(network[p]) else network[p]
        stat.at[p, 'net_density'] = np.count_nonzero(Y) / (n * (n - 1))
        stat.at[p, 'net_max_deg'] = np.max(np.sum(Y, axis=1))
    # Round to 3 significant figures
    stat.iloc[:, 1:] = stat.iloc[:, 1:].apply(lambda x: np.round(x, 3))
    print(stat)
    return stat


def Test_error_summary(input_test, label_test, label_name, net, figure=True, table=True):
    n, k = label_test.shape # num of samples, num of output_dim
    m = len(label_name)     # num of parameters

    # Determine label mode
    if m == k:
        mode = 1
    elif 2*m == k:
        mode = 2
    else:
        raise ValueError("label format not recognized")

    y_hat = net(input_test)
    y_hat = y_hat.detach().cpu().numpy()
    err = y_hat[:, :m] - label_test[:, :m]

    sdd = None
    if mode == 2:
        y_hat[:, m:k] = Positive_transform(y_hat[:, m:k])
        sdd = y_hat[:, m:k] - label_test[:, m:k]

    # Plot histogram and scatter
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
        if mode == 1:
            mean_SD = ["nan"]*m
        else:
            mean_SD = [f"{np.mean(sdd[:, j]):.3f} ({np.std(sdd[:, j])/np.sqrt(n):.1f})" for j in range(m)]

        result = pd.DataFrame({
            'bias': bias,
            'rmse': rmse,
            'mean_SD': mean_SD
        }, index=label_name)

        print("Test results:", result)

    return err, sdd


def run_shapley_analysis(net, input_train, input_test):
    # Limit sample size to keep SHAP runtime manageable.
    bg_n = min(len(input_train), max(1, 500))
    ev_n = len(input_test)
    bg_idx = np.random.choice(len(input_train), size=bg_n, replace=False)
    ev_idx = np.random.choice(len(input_test), size=ev_n, replace=False)
    background = torch.tensor(input_train[bg_idx], dtype=torch.float32) # (500, 38)
    eval_x = torch.tensor(input_test[ev_idx], dtype=torch.float32)  # (N_test, 38)

    net.eval()
    with torch.no_grad():
        pred_dim = net(eval_x[:1]).shape[1] # 7

    target_idx = int(np.clip(0, 0, pred_dim - 1))  # 0
    explainer = shap.DeepExplainer(net, background)
    shap_values = explainer.shap_values(eval_x) # (N_test, 38, 7)

    if isinstance(shap_values, list):
        # Multi-output model: pick one output dimension to explain.
        shap_matrix = np.array(shap_values[target_idx])
    else:
        shap_matrix = np.array(shap_values)
        if shap_matrix.ndim == 3:
            shap_matrix = shap_matrix[:, :, target_idx]
    
    # shap_matrix: (N_test, 38)

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
        eval_x.numpy()  # shape (N_test, 38)
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
        sample_data.numpy().flatten(),         # 该样本特征 torch.Size([38])
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

def upper(input_matrix):
    input_matrix = np.asarray(input_matrix)
    k = input_matrix.shape[0]
    # 获取上三角布尔索引
    mask = np.triu(np.ones((k, k), dtype=bool))
    # 提取上三角元素
    output = input_matrix[mask]
    return output