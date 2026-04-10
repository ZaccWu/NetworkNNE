import numpy as np
import time
import pickle
from Model import PeerModelwithFeature
from Descriptive import PeerDataDescriptive
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser('SetUp')
# 'peer' or 'peer+community'
parser.add_argument('--r', type=int, help='model type', default=10000) # number of parameter sample (default: 1e4, try: 10~1e3)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


def set_beta():
    bounds = [
        [-2.5, -3, -2, r'\c_f'],
        [9, 3, 12, r'\alpha_f'],
        [1, 0, 2, r'\beta_f'],  # beta (default: 1, 0, 2)
        [4, 0, 10, r'\delta_f'],  # delta # check here
        [0.75, 0, 1, r'\lambda_f'],  # theta
        [0.25, 0, 1, r'\gamma_f'],  # gamma # check here
        [4, 1, 5, r'\tau']  # tau
    ]
    return bounds


def set_up():
    R = args.r       # 测试时可减少样本量
    bounds = set_beta()

    bounds = np.array(bounds, dtype=object)
    theta = bounds[:, 0].astype(float)
    lb = bounds[:, 1].astype(float)
    ub = bounds[:, 2].astype(float)
    label_name = bounds[:, 3].tolist()

    n = 2511  # num of individual
    period = 2
    econmodel = PeerModelwithFeature(n, period)
    network, feature = econmodel.get_data(theta)
    PeerDataDescriptive(network)


    # --- PARAMETER BASKET ---
    print("Sampling parameter basket...")
    start_time = time.time()

    basket_theta = np.full((R, len(label_name)), np.nan)

    # Calculate network density
    # rho = np.count_nonzero(network[0]) / n / (n - 1)
    rho = network[0].sum() / n / (n - 1) # 利用csr matrix的特性计算网络密度

    n0, m0 = 500, 100 # 500, 100
    for t in range(R):
        while True:
            theta_sample = np.random.uniform(lb, ub) # uniform sampling within bounds
            econmodel = PeerModelwithFeature(n0, period=1)
            network_simul, feature_simul = econmodel.get_data(theta_sample)
            density = network_simul[0].sum() / n0 / (n0 - 1)
            if rho / 5 < density < rho * 5: # 选择密度在一定初始网络一定范围内的样本
                basket_theta[t, :] = theta_sample
                #PeerDataDescriptive(network_simul)
                #print("tau: ", int(round(theta_sample[-1])), ", lambda: ", theta_sample[0])
                break

        if t%10 == 0:
            print("Generate sample: ", t)

    print(f" Done. Time spent: {time.time() - start_time:.2f} seconds")

    save_dict = {
        'network': network, # csr sparse matrix
        'feature': feature,
        'label_name': label_name, # parameter name
        'lb': lb,           # parameter lower bound
        'ub': ub,           # parameter upper bound
        'basket_theta': basket_theta    # true parameter values (for out-of-sample evaluation)
    }
    return save_dict


if __name__ == "__main__":
    save_dict = set_up()
    with open('training_set.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
