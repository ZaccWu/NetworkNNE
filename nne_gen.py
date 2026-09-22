import numpy as np
import time
from joblib import Parallel, delayed
import pickle
from Model import FormationIFModel, PeerIModel
from Moments import FormationIFMoments, PeerIMoments
from tqdm import tqdm
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser('nneGen')
    parser.add_argument('--model', type=str, help='econ model', default='pi') # nf: network formation, pi: peer influence
    return parser.parse_args()


def simulate_moment(econmodel, dtmoments, theta):
    network_simul, feature_simul = econmodel.get_data(theta)
    moment = dtmoments.getMoments(network_simul, feature_simul)
    return moment, theta

def nne_gen(data):
    basket_theta = data['basket_theta']
    network = data['network']
    feature = data['feature']
    lb = data['lb']
    ub = data['ub']
    label_name = data['label_name']

    if args.model == 'nf':
        n = network[0].shape[0]
        econmodel = FormationIFModel(n, period=len(network))
        dtmoments = FormationIFMoments()
    elif args.model == 'pi':
        n = network.shape[0]
        econmodel = PeerIModel(n)
        dtmoments = PeerIMoments()
    else:
        assert ValueError('econ model not specified')


    

    print("Simulating training data...")
    start_time = time.time()

    R = basket_theta.shape[0]
    results = Parallel(n_jobs=-1, verbose=0)(
                  delayed(simulate_moment)(econmodel, dtmoments, basket_theta[t, :]) for t in tqdm(range(R))
    )     # for parallel computing

    # 拆分 input 和 label
    input_list, label_list = zip(*results) # [moments, theta]
    input_array = np.vstack(input_list)
    label_array = np.vstack(label_list)

    # train test split
    split_idx = int(0.9 * R)
    input_train = input_array[:split_idx, :]
    label_train = label_array[:split_idx, :]

    input_test = input_array[split_idx:, :]
    label_test = label_array[split_idx:, :]

    moment_real = dtmoments.getMoments(network, feature)

    input_real = moment_real
    label_real = np.full_like(lb, np.nan)
    print(f" Done. Time spent: {time.time() - start_time:.2f} seconds")

    save_dict = {
        # for out-of-sample evaluation
        'input_train': input_train,
        'label_train': label_train,
        'input_test': input_test,
        'label_test': label_test,
        # for monte carlo (bias-variance analysis)
        'input_real': input_real,
        'label_real': label_real,
        'label_name': label_name,
        'ub': ub,
        'lb': lb
    }
    return save_dict


if __name__ == "__main__":
    args = get_args()
    with open('training_set.pkl', 'rb') as f:  # data from "set_up.py"
        data = pickle.load(f)
    save_dict = nne_gen(data)
    with open('training_set_gen.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
