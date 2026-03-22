import numpy as np
import scipy.sparse as sp
from scipy.stats import poisson, expon

class PeerModelwithFeature():
    def __init__(self, n, period, *args):
        super(PeerModelwithFeature, self).__init__()
        self.ceiling = 50
        self.n = n
        self.period = period

    def get_data(self, theta):
        self.beta0 = theta[0]   # lambda_f
        self.beta1 = theta[1]   # alpha_f
        self.beta2 = theta[2]   # beta_f
        self.beta3 = theta[3]   # delta_f
        self.beta4 = theta[4]   # theta_f
        self.beta5 = theta[5]   # gamma_f
        self.tau = int(round(theta[6]))

        # Preallocate
        network = [sp.csr_matrix((self.n, self.n), dtype=bool) for _ in range(self.tau + self.period)]
        # Random initial w, g
        w = np.random.randn(self.n, 1)
        g = np.random.randn(self.n, 1)
        x = np.random.randn(self.n, 1)

        # Pre-compute w−w' and g+g'
        w_diff = np.abs(w - w.T)    # (n, n)
        g_summ = g + g.T            # (n, n)
        x_diff = np.abs(x - x.T)  # (n, n)
        U_constant = self.beta0 - self.beta3 * w_diff + self.beta4 *  g_summ

        #  Main loop: simulate from p = 2 : tau+period
        for p in range(1, self.tau + self.period):
            Y0 = network[p-1].toarray() # (n, n)
            # Degree and log-degree
            deg = np.sum(Y0, axis=1, keepdims=True)
            log_deg = np.log1p(deg)

            #  Compute U for network linking
            U = (
                self.beta1 * Y0
                - self.beta2 * x_diff
                + U_constant #+ p
                - self.beta5 * (log_deg + log_deg.T)
            )

            #i = np.tril(U > -4.5951, -1) # Screening sets i, j (lower triangular)
            i = np.tril(U > np.mean(U), -1)  # Screening sets i, j (lower triangular)
            density = poisson.rvs(self.n**2 * 0.01) / (self.n**2) # sprandsym equivalent
            j_rand = sp.rand(self.n, self.n, density)
            j = np.tril((j_rand != 0).toarray(), -1)

            #  Generate Y
            Y = np.zeros((self.n, self.n), dtype=bool)
            if np.any(j):       # Case j: probability = 100/(1+exp(-U))
                prob_j = 100.0 / (1.0 + np.exp(-U[j]))
                Y[j] = np.random.rand(np.sum(j)) < prob_j
            if np.any(i):       # Case i: probability = 1/(1+exp(-U))
                prob_i = 1.0 / (1.0 + np.exp(-U[i]))
                Y[i] = np.random.rand(np.sum(i)) < prob_i
            # Symmetrize
            Y = np.logical_or(Y, Y.T)
            Y = sp.csr_matrix(Y)
            # Save results
            network[p] = Y

        network = network[self.tau:] #  Remove first tau entries
        return network, x