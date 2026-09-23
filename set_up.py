import numpy as np
import time
import pickle
import sys
import argparse
import warnings

from torch import Value
warnings.filterwarnings("ignore")

from utils.analysis import FormationIFDtDescriptive, PeerIDtDescriptive
from Model import FormationIFModel, PeerIModel

def get_args():
    parser = argparse.ArgumentParser('SetUp')
    # 'peer' or 'peer+community'
    parser.add_argument('--r', type=int, help='num samples', default=5000) # number of parameter sample (default: 5000, try: 10~1e3)
    parser.add_argument('--model', type=str, help='econ model', default='pi') # nf: network formation, pi: peer influence
    parser.add_argument('--printdt', type=bool, help='print simu data detail', default=False)
    return parser.parse_args()


def set_beta_FormationIF():
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

def set_beta_PeerI():
    bounds = [
        [1.5, 2, 1, r'\beta_0'],
        [9, 5, 12, r'\delta_f'],  # beta_w

        [0, -1, 1, r'\alpha_0'],
        [0.75, 0, 1.5, r'\alpha_p'], 
        [0.75, 0, 1.5, r'\alpha_w'],  
    ]
    return bounds


def set_up():
    R = args.r   
    n = 2511  # num of individual

    # initialize setup
    if args.model == 'nf':
        bounds = set_beta_FormationIF()
        econmodel = FormationIFModel(n, period=2)
        descri = FormationIFDtDescriptive()
    elif args.model == 'pi':
        bounds = set_beta_PeerI()
        econmodel = PeerIModel(n)
        descri = PeerIDtDescriptive()
    else:
        assert ValueError('econ model not specified')

    bounds = np.array(bounds, dtype=object)
    theta = bounds[:, 0].astype(float)
    lb = bounds[:, 1].astype(float)
    ub = bounds[:, 2].astype(float)
    label_name = bounds[:, 3].tolist()
    network, feature = econmodel.get_data(theta)

    if args.model == 'nf': descri.getDescriptive(network)
    elif args.model == 'pi': descri.getDescriptive(network, feature)
    else: assert ValueError('econ model not specified')

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
            if args.model == 'nf':
                econmodel = FormationIFModel(n0, period=1)
            elif args.model == 'pi':
                econmodel = PeerIModel(n0)
            else:
                assert ValueError('econ model not specified')
            
            network_simul, feature_simul = econmodel.get_data(theta_sample)
            density = network_simul[0].sum() / n0 / (n0 - 1)
            if rho / 5 < density < rho * 5: 
                basket_theta[t, :] = theta_sample
                if args.printdt:
                    if args.model == 'nf': descri.getDescriptive(network_simul)
                    elif args.model == 'pi': descri.getDescriptive(network_simul, feature_simul)
                    else: assert ValueError('econ model not specified')

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
    args = get_args()
    save_dict = set_up()
    with open('training_set.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
