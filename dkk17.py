import numpy as np
import scipy.linalg
import scipy.stats
import scipy.sparse.linalg
from sklearn.decomposition import PCA
from utils import pca, svd_pow, k_lowest_ind, step_vec, flatten_matrix, reshape_vector

def cov_Tail(T, d, eps, tau):
    if T <= 10 * np.log(1 / eps):
        return 1
    return 3 * eps / (T * np.log(T))**2

def Q(G, P):
    return 2 * np.linalg.norm(P)**2

def cov_estimation_filter(S, eps, tau=0.1, limit=None, method='krylov'):
    d, n = S.shape
    C = 10
    C_prime = 0
    Sigma = (S @ S.T) / n
    G = scipy.stats.multivariate_normal(mean=np.zeros(d), cov=Sigma)
    invsqrt_Sigma = scipy.linalg.inv(scipy.linalg.sqrtm(Sigma))
    Y = invsqrt_Sigma @ S
    xinv_Sigma_x = np.array([y.T @ y for y in Y.T])
    mask = xinv_Sigma_x >= C * d * np.log(n / tau)

    if np.any(mask):
        print("early filter")
        if limit is None:
            return ~mask
        else:
            return ~mask | np.argsort(xinv_Sigma_x)[:max(0, n - limit)]
    
    if method == 'arpack':
        Z = np.array([np.kron(y, y) for y in Y.T]).T
        Id_flat = np.eye(d).flatten()
        TS = np.linalg.eigvalsh(-np.outer(Id_flat, Id_flat) + (Z @ Z.T) / n)
        lambda_, v = TS[0], np.linalg.eigh(TS)[1][:, 0]
    else:
        Z = lambda v: np.array([np.kron(y, y) @ v for y in Y.T])
        Id_flat = np.eye(d).flatten().reshape(-1, 1)
        TS = -np.outer(Id_flat, Id_flat.T) + (Z(np.eye(n)) @ Z(np.eye(n)).T) / n
        lambda_, v = scipy.sparse.linalg.eigs(TS, k=1, which='LM')
    
    if lambda_ <= (1 + C * eps * np.log(1 / eps)**2) * Q(invsqrt_Sigma @ G, v) / 2:
        return G
    
    V = 0.5 * (v.reshape(d, d) + v.reshape(d, d).T)
    ps = np.array([1 / np.sqrt(2) * (y.T @ V @ y - np.trace(V)) for y in Y.T])
    mu = np.median(ps)
    diffs = np.abs(ps - mu)

    for i, diff in enumerate(sorted(diffs)):
        shift = 3
        if diff < shift:
            continue
        T = diff - shift
        if T <= C_prime:
            continue
        if i / n >= cov_Tail(T, d, eps, tau):
            if limit is None:
                return diffs <= T
            else:
                return (diffs <= T) | np.argsort(diffs)[:max(0, n - limit)]

def cov_estimation_iterate(S, eps, tau=0.1, k=None, iters=None, limit=None):
    _, n = S.shape
    idxs = np.arange(n)
    i = 0
    
    while True:
        if iters is not None and i >= iters:
            break
        if k is None:
            Sk = S
        else:
            Sk = PCA(n_components=k).fit_transform(S.T).T
        
        select = cov_estimation_filter(Sk, eps, tau, limit=limit)
        
        if isinstance(select, scipy.stats._multivariate.multivariate_normal_frozen):
            print(f"Terminating early {i} success...")
            break
        if select is None:
            print(f"Terminating early {i} fail...")
            break
        
        if limit is not None:
            limit -= len(select) - np.sum(select)
            assert limit >= 0
        
        S = S[:, select]
        idxs = idxs[select]
        i += 1
        if limit == 0:
            break

    select = np.zeros(n, dtype=bool)
    select[idxs] = True
    return select

def rcov(S, eps, tau=0.1, k=None, iters=None, limit=None):
    select = cov_estimation_iterate(S, eps, tau, k, iters=iters, limit=limit)
    selected = S[:, select]
    return selected @ selected.T

def mean_Tail(T, d, eps, delta, tau, nu=1):
    return 8 * np.exp(-T**2 / (2 * nu)) + 8 * eps / (T**2 * np.log(d * np.log(d / (eps * tau))))

def mean_estimation_filter(S, eps, tau=0.1, nu=1, limit=None):
    d, n = S.shape
    mu = np.mean(S, axis=1, keepdims=True)
    Sigma = np.cov(S, rowvar=True, bias=True)
    lambda_, v = scipy.linalg.eigh(Sigma, subset_by_index=[d-1, d-1])
    
    if lambda_[0] - 1 <= eps * np.log(1 / eps):
        return None
    
    delta = 3 * np.sqrt(eps * (lambda_[0] - 1))
    lambda_mags = np.abs((S - mu).T @ v).flatten()

    for i, mag in enumerate(sorted(lambda_mags)):
        if mag < delta:
            continue
        T = mag - delta
        if (n - i) / n > mean_Tail(T, d, eps, delta, tau, nu):
            if limit is None:
                return lambda_mags <= mag
            else:
                return (lambda_mags <= mag) | np.argsort(lambda_mags)[:max(0, n - limit)]

def mean_estimation_iterate(A, eps, tau=0.1, nu=1, iters=None, limit=None):
    d, n = A.shape
    idxs = np.arange(n)
    i = 0
    
    while True:
        if iters is not None and i >= iters:
            break
        select = mean_estimation_filter(A, eps, tau, nu, limit=limit)
        
        if select is None:
            print(f"Terminating early {i}...")
            break
        
        if limit is not None:
            limit -= len(select) - np.sum(select)
            assert limit >= 0
        
        A = A[:, select]
        idxs = idxs[select]
        i += 1

    select = np.zeros(n, dtype=bool)
    select[idxs] = True
    return select

def rmean(A, eps, tau=0.1, nu=1, iters=None, limit=None):
    select = mean_estimation_iterate(A, eps, tau, nu, iters=iters, limit=limit)
    return np.mean(A[:, select], axis=1, keepdims=True)
