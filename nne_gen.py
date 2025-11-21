import numpy as np
import time
from joblib import Parallel, delayed
import pickle
from Model import Model
from Moments import Moments

# ----------------------------
# LOAD
# ----------------------------
with open('training_set.pkl', 'rb') as f:  # data from "set_up.py"
    data = pickle.load(f)

basket_theta = data['basket_theta']
guild = data['guild']
network = data['network']
lb = data['lb']
ub = data['ub']
label_name = data['label_name']

print("Simulating training data...")
start_time = time.time()

# --- SETUP ---
T = basket_theta.shape[0]
n, m = guild[0].shape
period = len(guild)


def simulate_moment(theta):
    network_simul, guild_simul = Model(theta, n, m, period)
    moment = Moments(network_simul, guild_simul)
    return moment, theta

# 并行化
results = Parallel(n_jobs=-1)(delayed(simulate_moment)(basket_theta[t, :]) for t in range(T))

# 拆分 input 和 label
input_list, label_list = zip(*results)
input_array = np.vstack(input_list)
label_array = np.vstack(label_list)

# train test split
split_idx = int(0.9 * T)
input_train = input_array[:split_idx, :]
label_train = label_array[:split_idx, :]

input_test = input_array[split_idx:, :]
label_test = label_array[split_idx:, :]

# --- REAL DATA ---
moment_real = Moments(network, guild)
input_real = moment_real
label_real = np.full_like(lb, np.nan)

print(f" Done. Time spent: {time.time() - start_time:.2f} seconds")

save_dict = {
    'input_train': input_train,
    'label_train': label_train,
    'input_test': input_test,
    'label_test': label_test,
    'input_real': input_real,
    'label_real': label_real,
    'label_name': label_name,
    'ub': ub,
    'lb': lb
}

with open('training_set_gen.pkl', 'wb') as f:
    pickle.dump(save_dict, f)
