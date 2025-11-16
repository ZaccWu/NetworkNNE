import numpy as np
import scipy.sparse as sp
from scipy.stats import poisson, expon


def Model(theta, n, m, period, *args):
    """
    Python translation of MATLAB's Model.m
    -------------------------------------------------------
    [network, guild, track] = Model(theta, n, m, period, varargin)
    """

    ceiling = 50

    # Simulate Network and Guilds
    beta0 = theta[0]
    beta1 = theta[1]
    beta2 = theta[2]
    beta3 = theta[3]
    beta4 = theta[4]
    beta5 = theta[5]
    gamma1 = theta[6]
    gamma2 = theta[7]
    gamma3 = theta[8]
    gamma4 = theta[9]
    tau = int(round(theta[10]))

    # Preallocate
    network = [sp.csr_matrix((n, n), dtype=bool) for _ in range(tau + period)]
    guild   = [sp.csr_matrix((n, m), dtype=bool) for _ in range(tau + period)]
    track   = [sp.csr_matrix((n, n), dtype=bool) for _ in range(tau + period)]

    # Random initial w, g
    w = np.random.randn(n, 1)
    g = np.random.randn(n, 1)

    # Optional policy_fun
    has_policy = len(args) > 0
    if has_policy:
        policy_fun = args[0]
        age = np.zeros((n, 1))

    # Pre-compute w−w' and g+g'
    w_diff = np.abs(w - w.T)                # n×n
    U_constant = beta0 - beta3 * w_diff + beta4 * (g + g.T)

    # =====================================================
    #   Main loop: simulate from p = 2 : tau+period
    # =====================================================
    for p in range(1, tau + period):

        Y0 = network[p-1].toarray()
        L0 = guild[p-1].toarray()

        # Degree and log-degree
        deg = np.sum(Y0, axis=1, keepdims=True)
        log_deg = np.log1p(deg)

        # Guild statistics
        same_guild = L0 @ L0.T                   # n×n
        num_friend = Y0 @ L0                     # n×m
        fac_friend = num_friend / (deg + 1e-15)
        siz_guild = np.ones((n, 1)) @ np.sum(L0, axis=0, keepdims=True)

        # ---------------------------------------------------
        #  Compute U for network linking
        # ---------------------------------------------------
        U = (
            beta1 * Y0
            + beta2 * same_guild
            + U_constant
            - beta5 * (log_deg + log_deg.T)
        )

        # Screening sets i, j (lower triangular)
        i = np.tril(U > -4.5951, -1)
        # sprandsym equivalent
        density = poisson.rvs(n**2 * 0.01) / (n**2)
        j_rand = sp.rand(n, n, density)
        j = np.tril((j_rand != 0).toarray(), -1)

        # ---------------------------------------------------
        #  Generate Y
        # ---------------------------------------------------
        Y = np.zeros((n, n), dtype=bool)

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
        w_avg_diff = (w_diff * L0) / (siz_guild + 1e-15)   # n×m
        g_avg = (g.T @ L0) / (siz_guild + 1e-15)           # 1×m

        U = (
            gamma1 * L0
            + gamma2 * fac_friend
            - gamma3 * w_avg_diff
            + gamma4 * g_avg
        )

        if has_policy:
            U = U + policy_fun(age) * L0

        # Add extreme-value noise (Gumbel)
        U = U - expon.rvs(scale=1.0, size=(n, m))

        # Winner-take-all choice of guild
        L = (U == np.max(U, axis=1, keepdims=True)).astype(bool)

        # ---------------------------------------------------
        #  Enforce maximum guild size
        # ---------------------------------------------------
        while True:
            guild_sizes = np.sum(L, axis=0)
            j = np.argmax(guild_sizes)
            max_size = guild_sizes[j]

            k = max_size - ceiling
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

        # ---------------------------------------------------
        #  Update age
        # ---------------------------------------------------
        if has_policy:
            guild_change = np.any(L.toarray() - L0, axis=1, keepdims=True)
            age = (1 - guild_change) * (age + 1)

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

    # =====================================================
    #  Remove first tau entries
    # =====================================================
    network = network[tau:]
    guild = guild[tau:]
    track = track[tau:]

    return network, guild, track