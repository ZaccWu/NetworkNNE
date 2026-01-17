import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def multinomial(Y, XT, sampling_prob=0.5, option=''):
    # 初始化信息表
    Ninfo = pd.DataFrame(np.nan, index=['raw', 'missing_removed', 'sampled'],
                         columns=['total_obs', 'total_options'])
    Ninfo.loc['raw', 'total_obs'] = Y.shape[0]
    Ninfo.loc['raw', 'total_options'] = Y.shape[1]

    fields = list(XT.keys())
    c = len(fields)

    # --- REMOVE MISSING ROWS AND COLUMNS ---
    n, m = Y.shape
    i = np.zeros(n, dtype=bool)
    j = np.zeros(m, dtype=bool)

    for field in fields:
        Q = np.isnan(XT[field]) | np.isinf(XT[field])
        i |= np.all(Q, axis=1)
        j |= np.all(Q, axis=0)

    Y = np.delete(Y, np.where(j)[0], axis=1)
    i |= np.sum(Y, axis=1) == 0
    Y = Y[~i, :]

    for field in fields:
        if XT[field].shape[0] > 1:
            XT[field] = XT[field][~i, :]
        XT[field] = np.delete(XT[field], np.where(j)[0], axis=1)

    Ninfo.loc['missing_removed', 'total_obs'] = Y.shape[0]
    Ninfo.loc['missing_removed', 'total_options'] = Y.shape[1]

    # --- SAMPLING ---
    n, m = Y.shape
    t = int(np.ceil(sampling_prob * m))

    Q = np.random.rand(n, m) * (1 - Y)
    thresholds = np.quantile(Q, t / m, axis=1)
    i_mask = Q <= thresholds[:, None]

    flat_idx = np.flatnonzero(i_mask.T)
    Y_sampled = Y.flat[flat_idx]

    X = np.empty((n, t, c))
    for k, field in enumerate(fields):
        Qk = np.ones((n, 1)) * XT[field]
        X[:, :, k] = Qk.flat[flat_idx].reshape(n, t)
        XT[field] = None

    Ninfo.loc['sampled', 'total_obs'] = Y_sampled.shape[0]
    Ninfo.loc['sampled', 'total_options'] = t

    print(Ninfo)

    if np.any(np.isnan(X)):
        raise ValueError('There are still NaN values in X.')

    # --- WINSORIZE ---
    if 'winsorize' in option:
        for k in range(c):
            Qk = X[:, :, k].reshape(-1)
            lower = np.quantile(Qk, 0.01)
            upper = np.quantile(Qk, 0.99)
            X[:, :, k] = np.clip(X[:, :, k], lower, upper)

    # --- MAXIMUM LIKELIHOOD ---
    y_idx = np.flatnonzero(Y_sampled)

    def likelihood(beta, y_idx, X):
        n, m, c = X.shape
        coef = beta.reshape(1, 1, -1)
        V = np.sum(X * coef, axis=2)
        V = np.exp(V)
        sV = np.sum(V, axis=1)
        p = V.flat[y_idx] / sV.repeat(m)[:len(y_idx)]
        val = -np.mean(np.log(p))

        score = np.zeros((n, c))
        for k in range(c):
            Xk = X[:, :, k]
            score[:, k] = np.sum(Xk * V, axis=1) / sV - Xk.flat[y_idx]
        grad = np.mean(score, axis=0)
        return val, grad

    def obj(beta):
        val, grad = likelihood(beta, y_idx, X)
        return val, grad

    res = minimize(lambda b: obj(b)[0], np.zeros(c), jac=lambda b: obj(b)[1],
                   method='BFGS', options={'disp': True, 'maxiter': 300, 'gtol': 1e-6})

    beta = res.x
    _, score = likelihood(beta, y_idx, X)
    V = np.linalg.inv(score.T @ score)
    se = np.sqrt(np.diag(V))

    # --- DISPLAY ---
    p_vals = 2 * (1 - norm.cdf(np.abs(beta) / se))
    sig = np.array(['   '] * len(p_vals), dtype=object)
    sig[p_vals < 0.10] = '*  '
    sig[p_vals < 0.05] = '** '
    sig[p_vals < 0.01] = '***'

    vnames = [f.replace('__', ' (').replace('_', ' ') for f in fields]

    print(pd.DataFrame({'beta': np.round(beta, 3),
                        'sig': sig,
                        'se': np.round(se, 3),
                        'p': np.round(p_vals, 3)}, index=vnames))

    logL = -res.fun
    print(pd.DataFrame({' ': [-logL]}, index=['log_likelihood']))

    return beta, se
