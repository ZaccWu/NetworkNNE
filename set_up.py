import numpy as np
import time
import pickle
from Model import SimplePeerModel, PeerModelwithFeature, MixModel
from Descriptive import PeerDataDescriptive, MixDataDescriptive
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser('SetUp')
# 'peer' or 'peer+community'
parser.add_argument('--mod', type=str, help='model type', default='peerf') # speer, peerf, mix
parser.add_argument('--r', type=int, help='model type', default=100) # number of parameter sample (default: 1e4, try: 10~1e3)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


def set_beta(mod):
    if mod == 'speer': # simple peer model
        bounds = [
            [-6.5, -10, -5, r'\lambda_f'],
            [12, 5, 15, r'\alpha_f'],
            [4, 0, 10, r'\delta_f'],  # delta
            [0.75, 0, 1, r'\theta_f'],  # theta
            [0.25, 0, 1, r'\gamma_f'],  # gamma
            [4, 1, 5, r'\tau']  # tau
        ]
        return bounds

    elif mod == 'peerf': # peer model with features
        bounds = [
            [-6.5, -10, -5, r'\lambda_f'],
            [12, 5, 15, r'\alpha_f'],
            [1, 0, 2, r'\beta_f'],  # beta
            [4, 0, 10, r'\delta_f'],  # delta
            [0.75, 0, 1, r'\theta_f'],  # theta
            [0.25, 0, 1, r'\gamma_f'],  # gamma
            [4, 1, 5, r'\tau']  # tau
        ]
        return bounds

    elif mod == 'mix': # mixed model (peer + community)
        bounds = [
            [-6.5, -10, -5, r'\beta_0'],  # lambda
            [12, 5, 15, r'\beta_1'],  # alpha
            [6, 0, 10, r'\beta_2'],  # beta
            [4, 0, 10, r'\beta_3'],  # delta
            [0.75, 0, 1, r'\beta_4'],  # theta
            [0.25, 0, 1, r'\beta_5'],  # gamma
            [9, 5, 10, r'\gamma_1'],  # alpha
            [2.5, 0, 5, r'\gamma_2'],  # beta
            [1.5, 0, 5, r'\gamma_3'],  # delta
            [3.5, 0, 5, r'\gamma_4'],  # theta
            [4, 1, 5, r'\tau']  # tau
        ]
        return bounds

    else:
        raise ValueError("Model Not Specified")


def set_up():
    R = args.r       # 测试时可减少样本量
    bounds = set_beta(args.mod)

    bounds = np.array(bounds, dtype=object)
    theta = bounds[:, 0].astype(float)
    lb = bounds[:, 1].astype(float)
    ub = bounds[:, 2].astype(float)
    label_name = bounds[:, 3].tolist()

    # simulate 'real_data'
    if args.mod in ['speer', 'peerf']:
        n = 2511  # num of individual
        period = 4
        if args.mod == 'speer':
            econmodel = SimplePeerModel(n, period)  # network: list of sparse numpy matrix (len=period)
            network = econmodel.get_data(theta)
        else:
            econmodel = PeerModelwithFeature(n, period)
            network, _ = econmodel.get_data(theta)
        PeerDataDescriptive(network)

    else:
        n = 2511  # num of individual
        m = 537
        period = 4
        econmodel = MixModel(n, m, period) # network, guild: list of sparse numpy matrix (len=period)
        network, guild = econmodel.get_data(theta)
        MixDataDescriptive(network, guild)


    # --- PARAMETER BASKET ---
    print("Sampling parameter basket...")
    start_time = time.time()

    basket_theta = np.full((R, len(label_name)), np.nan)

    # Calculate network density
    # rho = np.count_nonzero(network[0]) / n / (n - 1)
    rho = network[0].sum() / n / (n - 1) # 利用csr matrix的特性计算网络密度

    n0, m0 = 500, 100
    for t in range(R):
        while True:
            theta_sample = np.random.uniform(lb, ub) # uniform sampling within bounds
            if args.mod == 'speer':
                econmodel = SimplePeerModel(n0, period=1)
                network_simul = econmodel.get_data(theta_sample)
            elif args.mod == 'peerf':
                econmodel = PeerModelwithFeature(n0, period=1)
                network_simul, feature_simul = econmodel.get_data(theta_sample)
            else:
                econmodel = MixModel(n0, m0, period=1)
                network_simul, guild_simul = econmodel.get_data(theta_sample)

            density = network_simul[0].sum() / n0 / (n0 - 1)

            #if rho / 5 < density < rho * 5:
            if rho / 5 < density < rho * 5:  # 选择密度在一定初始网络一定范围内的样本
                basket_theta[t, :] = theta_sample
                break

        if t%10 == 0:
            print("Generate sample: ", t)

    print(f" Done. Time spent: {time.time() - start_time:.2f} seconds")

    if args.mod == 'speer':
        save_dict = {
            'network': network, # csr sparse matrix
            'label_name': label_name, # parameter name
            'lb': lb,           # parameter lower bound
            'ub': ub,           # parameter upper bound
            'basket_theta': basket_theta    # true parameter values (for out-of-sample evaluation)
        }

    elif args.mod == 'peerf':
        save_dict = {
            'network': network, # csr sparse matrix
            'feature': feature_simul,
            'label_name': label_name, # parameter name
            'lb': lb,           # parameter lower bound
            'ub': ub,           # parameter upper bound
            'basket_theta': basket_theta    # true parameter values (for out-of-sample evaluation)
        }

    else:
        save_dict = {
            'network': network, # csr sparse matrix
            'guild': guild,     # csr sparse matrix
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