import numpy as np
import scipy.sparse as sp
from scipy.stats import poisson, expon



class SimplePeerModel():
    def __init__(self, n, period, *args):
        super(SimplePeerModel, self).__init__()
        self.ceiling = 50
        self.n = n
        self.period = period

    def get_data(self, theta):
        self.beta0 = theta[0]   # lambda_f
        self.beta1 = theta[1]   # alpha_f
        self.beta3 = theta[2]   # delta_f
        self.beta4 = theta[3]   # theta_f
        self.beta5 = theta[4]   # gamma_f
        self.tau = int(round(theta[5]))

        # Preallocate
        network = [sp.csr_matrix((self.n, self.n), dtype=bool) for _ in range(self.tau + self.period)]
        # Random initial w, g
        w = np.random.randn(self.n, 1)
        g = np.random.randn(self.n, 1)

        # Pre-compute w−w' and g+g'
        w_diff = np.abs(w - w.T)    # (n, n)
        g_summ = g + g.T            # (n, n)
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
                + U_constant
                - self.beta5 * (log_deg + log_deg.T)
            )

            #i = np.tril(U > -4.5951, -1) # Screening sets i, j (lower triangular)
            i = np.tril(U > -11.5951, -1)  # Screening sets i, j (lower triangular)
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
        return network






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
                + U_constant
                - self.beta5 * (log_deg + log_deg.T)
            )

            #i = np.tril(U > -4.5951, -1) # Screening sets i, j (lower triangular)
            i = np.tril(U > -11.5951, -1)  # Screening sets i, j (lower triangular)
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





class MixModel():
    def __init__(self, n, m, period, *args):
        super(MixModel, self).__init__()
        self.ceiling = 50
        # Simulate Network and Guilds
        self.period = period
        self.n = n
        self.m = m

    def get_data(self, theta):
        self.beta0 = theta[0]
        self.beta1 = theta[1]
        self.beta2 = theta[2]
        self.beta3 = theta[3]
        self.beta4 = theta[4]
        self.beta5 = theta[5]
        self.gamma1 = theta[6]
        self.gamma2 = theta[7]
        self.gamma3 = theta[8]
        self.gamma4 = theta[9]
        self.tau = int(round(theta[10]))

        # Preallocate
        network = [sp.csr_matrix((self.n, self.n), dtype=bool) for _ in range(self.tau + self.period)]
        guild   = [sp.csr_matrix((self.n, self.m), dtype=bool) for _ in range(self.tau + self.period)]
        track   = [sp.csr_matrix((self.n, self.n), dtype=bool) for _ in range(self.tau + self.period)]

        # Random initial w, g
        w = np.random.randn(self.n, 1)
        g = np.random.randn(self.n, 1)

        # Pre-compute w−w' and g+g'
        w_diff = np.abs(w - w.T)    # (n, n)
        g_summ = g + g.T            # (n, n)
        U_constant = self.beta0 - self.beta3 * w_diff + self.beta4 *  g_summ

        # =====================================================
        #   Main loop: simulate from p = 2 : tau+period
        # =====================================================
        for p in range(1, self.tau + self.period):

            Y0 = network[p-1].toarray() # (n, n)
            L0 = guild[p-1].toarray()   # (n, m)

            # Degree and log-degree
            deg = np.sum(Y0, axis=1, keepdims=True)
            log_deg = np.log1p(deg)

            # Guild statistics
            same_guild = L0 @ L0.T                  # (n, m) × (m, n) -> (n, n)
            num_friend = Y0 @ L0                     # (n, n) × (n, m) -> (n, m)
            fac_friend = num_friend / (deg + 1e-15)
            siz_guild = np.ones((self.n, 1)) @ np.sum(L0, axis=0, keepdims=True)

            # ---------------------------------------------------
            #  Compute U for network linking
            # ---------------------------------------------------
            U = (
                self.beta1 * Y0
                + self.beta2 * same_guild
                + U_constant
                - self.beta5 * (log_deg + log_deg.T)
            )

            # Screening sets i, j (lower triangular)
            i = np.tril(U > -4.5951, -1)
            # sprandsym equivalent
            density = poisson.rvs(self.n**2 * 0.01) / (self.n**2)
            j_rand = sp.rand(self.n, self.n, density)
            j = np.tril((j_rand != 0).toarray(), -1)

            # ---------------------------------------------------
            #  Generate Y
            # ---------------------------------------------------
            Y = np.zeros((self.n, self.n), dtype=bool)

            # Case j: probability = 100/(1+exp(-U))
            if np.any(j):
                prob_j = 100.0 / (1.0 + np.exp(-U[j]))
                Y[j] = np.random.rand(np.sum(j)) < prob_j

            # Case i: probability = 1/(1+exp(-U))
            if np.any(i):
                prob_i = 1.0 / (1.0 + np.exp(-U[i]))
                Y[i] = np.random.rand(np.sum(i)) < prob_i

            # Symmetrize
            Y = np.logical_or(Y, Y.T)
            Y = sp.csr_matrix(Y)

            # ---------------------------------------------------
            #  Guild choice U for each node
            # ---------------------------------------------------
            w_avg_diff = (w_diff @ L0) / (siz_guild + 1e-15)   # (n, n) × (n, m) -> (n, m)
            g_avg = (g.T @ L0) / (siz_guild + 1e-15)           # (1, n) × (n, m) -> (1, m)

            U = (
                self.gamma1 * L0
                + self.gamma2 * fac_friend
                - self.gamma3 * w_avg_diff
                + self.gamma4 * g_avg
            )


            # Add extreme-value noise (Gumbel)
            U = U - expon.rvs(scale=1.0, size=(self.n, self.m)) # (n, m)

            # Winner-take-all choice of guild
            L = (U == np.max(U, axis=1, keepdims=True)).astype(bool) # (n, m)


            # ---------------------------------------------------
            #  Enforce maximum guild size
            # ---------------------------------------------------
            while True:
                guild_sizes = np.sum(L, axis=0)
                j = np.argmax(guild_sizes)
                max_size = guild_sizes[j]

                k = max_size - self.ceiling
                if k > 0:
                    U[:, j] = -np.inf
                    # nodes that newly choose j
                    diff = (L[:, j] & (~L0[:, j]))
                    idx = np.where(diff)[0]
                    if len(idx) > 0:
                        drop_idx = np.random.choice(idx, size=k, replace=False)
                        L[drop_idx, :] = (
                            U[drop_idx, :] == np.max(U[drop_idx, :], axis=1, keepdims=True)
                        )
                else:
                    break

            L = sp.csr_matrix(L)
            # Save results
            network[p] = Y
            guild[p] = L

            # track updating
            if True:
                Y_dense = Y.toarray()
                Y0_dense = Y0
                same_dense = same_guild.astype(bool)

                prev = track[p-1].toarray()
                new_track = (Y_dense & Y0_dense) * prev + (Y_dense & (~Y0_dense)) * same_dense
                track[p] = sp.csr_matrix(new_track)

        #  Remove first tau entries
        network = network[self.tau:]
        guild = guild[self.tau:]
        return network, guild