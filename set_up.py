import numpy as np
import time
import pickle
from Model import Model
from Moments import Moments
from Descriptive import Descriptive

# ----------------------------
# LOAD / SET-UP
# ----------------------------
n = 2511
m = 537
period = 4

# T = int(1e4)  # number of parameter samples
T = 10       # 可以用于测试

# ----------------------------
# BOUNDS
# ----------------------------
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

bounds = np.array(bounds, dtype=object)
theta = bounds[:, 0].astype(float)
lb = bounds[:, 1].astype(float)
ub = bounds[:, 2].astype(float)
label_name = bounds[:, 3].tolist()

# ----------------------------
# SIMULATE "REAL DATA"
# ----------------------------
network, guild = Model(theta, n, m, period)
moment = Moments(network, guild)
Descriptive(network, guild)

# ----------------------------
# PARAMETER BASKET
# ----------------------------
print("Sampling parameter basket...")
start_time = time.time()

basket_theta = np.full((T, len(label_name)), np.nan)

# 计算真实网络密度
rho = np.count_nonzero(network[0]) / n / (n - 1)

n0 = 500
m0 = 100

for i in range(T):
    while True:
        # uniform sampling within bounds
        theta_sample = np.random.uniform(lb, ub)

        network_simul, guild_simul = Model(theta_sample, n0, m0, 1)

        density = np.count_nonzero(network_simul[0]) / n0 / (n0 - 1)

        if rho / 5 < density < rho * 5:
            basket_theta[i, :] = theta_sample
            break

print(f" Done. Time spent: {time.time() - start_time:.2f} seconds")

# ----------------------------
# SAVE
# ----------------------------
save_dict = {
    'network': network,
    'guild': guild,
    'label_name': label_name,
    'lb': lb,
    'ub': ub,
    'basket_theta': basket_theta
}

with open('training_set.pkl', 'wb') as f:
    pickle.dump(save_dict, f)
