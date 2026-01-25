import numpy as np
import time
from joblib import Parallel, delayed
import pickle
from Model import SimplePeerModel, PeerModelwithFeature, MixModel
from Moments import SimplePeerMoments, PeerFeatureMoments, MixMoments
from tqdm import tqdm
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser('nneGen')
# 'peer' or 'peer+community'
parser.add_argument('--mod', type=str, help='model type', default='peerf') # speer, peerf, mix


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


def simulate_moment(mod, econmodel, theta):
    if mod == 'peerf':
        network_simul, feature_simul = econmodel.get_data(theta)
        moment = PeerFeatureMoments(network_simul, feature_simul)
    elif mod == 'speer':
        network_simul = econmodel.get_data(theta)
        moment = SimplePeerMoments(network_simul)
    else:
        network_simul, guild_simul = econmodel.get_data(theta)
        moment = MixMoments(network_simul, guild_simul)
    return moment, theta


def nne_gen(data):
    if args.mod == 'speer':
        basket_theta = data['basket_theta']
        network = data['network']
        lb = data['lb']
        ub = data['ub']
        label_name = data['label_name']
        n = network[0].shape[0]
        period = len(network)
        econmodel = SimplePeerModel(n, period)
    elif args.mod == 'peerf':
        basket_theta = data['basket_theta']
        network = data['network']
        feature = data['feature']
        lb = data['lb']
        ub = data['ub']
        label_name = data['label_name']
        n = network[0].shape[0]
        period = len(network)
        econmodel = PeerModelwithFeature(n, period)
    elif args.mod == 'mix':
        basket_theta = data['basket_theta']
        guild = data['guild']
        network = data['network']
        lb = data['lb']
        ub = data['ub']
        label_name = data['label_name']
        n, m = guild[0].shape
        period = len(guild)
        econmodel = MixModel(n, m, period)
    else:
        raise ValueError("Model Not Specified")

    print("Simulating training data...")
    start_time = time.time()

    R = basket_theta.shape[0]
    results = Parallel(n_jobs=-1, verbose=0)(
                  delayed(simulate_moment)(args.mod, econmodel, basket_theta[t, :]) for t in tqdm(range(R))
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

    # recall 'real' data
    if args.mod == 'speer':
        moment_real = SimplePeerMoments(network)
    elif args.mod == 'peerf':
        moment_real = PeerFeatureMoments(network, feature)
    else:
        moment_real = MixMoments(network, guild)

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
    with open('training_set.pkl', 'rb') as f:  # data from "set_up.py"
        data = pickle.load(f)
    save_dict = nne_gen(data)
    with open('training_set_gen.pkl', 'wb') as f:
        pickle.dump(save_dict, f)