import numpy as np
import scipy.sparse as sp
from scipy.stats import poisson, expon

class FormationIFModel(): # Network formation model with individual features
    def __init__(self, n, period, *args):
        super(FormationIFModel, self).__init__()
        self.n = n
        self.period = period

    def get_data(self, theta):

        self.beta0 = theta[0]   # c_f
        self.beta1 = theta[1]   # alpha_f
        self.beta2 = theta[2]   # beta_f
        self.beta3 = theta[3]   # delta_f
        self.beta4 = theta[4]   # lambda_f
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

            #  Generate A
            A = np.zeros((self.n, self.n), dtype=bool)
            if np.any(j):       # Case j: probability = 100/(1+exp(-U))
                prob_j = 100.0 / (1.0 + np.exp(-U[j]))
                A[j] = np.random.rand(np.sum(j)) < prob_j
            if np.any(i):       # Case i: probability = 1/(1+exp(-U))
                prob_i = 1.0 / (1.0 + np.exp(-U[i]))
                A[i] = np.random.rand(np.sum(i)) < prob_i
            # Symmetrize
            A = np.logical_or(A, A.T)
            A = sp.csr_matrix(A)
            # Save results
            network[p] = A

        network = network[self.tau:] #  Remove first tau entries
        return network, x


class PeerIModel(): # Peer influence model
    def __init__(self, n, *args):
        super(PeerIModel, self).__init__()
        self.n = n

    def get_data(self, theta):
        self.beta0 = theta[0]   # c_f
        self.beta_w = theta[1]   # alpha
        self.alpha0 = theta[2]  
        self.alpha_p = theta[3]
        self.alpha_w = theta[4]


        # Preallocate
        network = sp.csr_matrix((self.n, self.n), dtype=bool)   # initialize network
        w = np.random.randn(self.n, 1)
        w_diff = np.abs(w - w.T)    # (n, n)

        A_U = self.beta0 - self.beta_w * w_diff # network formation likelihood
        i = np.tril(A_U > np.mean(A_U), -1)  # Screening sets i, j (lower triangular)

        density = poisson.rvs(self.n**2 * 0.01) / (self.n**2) # sprandsym equivalent
        j_rand = sp.rand(self.n, self.n, density)
        j = np.tril((j_rand != 0).toarray(), -1)

        #  Generate A
        A = np.zeros((self.n, self.n), dtype=bool)
        if np.any(j):       # Case j: probability = 100/(1+exp(-U))
            prob_j = 100.0 / (1.0 + np.exp(-A_U[j]))
            A[j] = np.random.rand(np.sum(j)) < prob_j
        if np.any(i):       # Case i: probability = 1/(1+exp(-U))
            prob_i = 1.0 / (1.0 + np.exp(-A_U[i]))
            A[i] = np.random.rand(np.sum(i)) < prob_i
        # Symmetrize
        A = np.logical_or(A, A.T)
        A = sp.csr_matrix(A)
        network = A

        y0 = self.alpha0 + self.alpha_w * w
        degree = np.asarray(A.sum(axis=1)).reshape(-1, 1)
        peer_sum = np.asarray(A @ y0).reshape(-1, 1)
        peer_avg = np.zeros_like(y0)
        nz = degree.ravel() > 0
        peer_avg[nz] = peer_sum[nz] / degree[nz]
        y = (
            self.alpha0
            + self.alpha_w * w
            + self.alpha_p * peer_avg
        )
        return network, [y0, y]

