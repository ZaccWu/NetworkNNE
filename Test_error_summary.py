import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from Positive_transform import Positive_transform


def Test_error_summary(input_test, label_test, label_name, net, figure=True, table=True):
    """
    Measure misclassification / prediction error and optionally plot histograms and scatter plots.

    Parameters
    ----------
    input_test : torch.tensor
    label_test : np.ndarray
    label_name : list of str
    net : trained network object with .predict() method
    figure : bool, optional
    table : bool, optional

    Returns
    -------
    err : np.ndarray
        Error for main predictions
    sdd : np.ndarray or None
        Error for standard deviation predictions (if applicable)
    """

    n, k = label_test.shape # num of samples, num of parameters
    m = len(label_name)     # num of parameter

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
            ax.scatter(label_test[:, j], err[:, j] + label_test[:, j], s=30, color=[0, 0.4, 0.7], marker='.')
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
