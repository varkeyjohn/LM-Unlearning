import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import svd
from scipy.sparse.linalg import svds

def pca(A, k):
    """
    Performs Principal Component Analysis (PCA) on matrix A.

    Args:
        A (ndarray): Input matrix of shape (d, n)
        k (int): Number of principal components

    Returns:
        (ndarray, ndarray): PCA transformed data, and principal components
    """
    assert k <= min(A.shape)
    A_centered = A - np.mean(A, axis=1, keepdims=True)

    if k == min(A.shape):
        U, _, _ = svd(A_centered, full_matrices=False)
    else:
        U, _, _ = svds(A_centered, k=k)

    return U.T @ A_centered, U.T

def svd_pow(A, p):
    """
    Computes matrix power using SVD decomposition.

    Args:
        A (ndarray): Square matrix
        p (float): Exponent for the singular values

    Returns:
        ndarray: A^p computed via SVD
    """
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    U, S, Vt = svd(A)
    assert np.all(S > 0), "All singular values must be positive"
    
    return U @ np.diag(S ** p) @ Vt

def k_lowest_ind(A, k):
    """
    Returns a boolean mask for the k smallest values in A.

    Args:
        A (ndarray): Input array
        k (int): Number of smallest elements to select

    Returns:
        ndarray: Boolean mask
    """
    assert 0 <= k <= A.size
    if k == 0:
        return np.zeros_like(A, dtype=bool)
    
    sorted_vals = np.sort(A.flatten())
    cutoff = sorted_vals[k - 1]
    
    return A <= cutoff

def step_vec(n, k):
    """
    Creates a step vector of length n with the first k elements as True.

    Args:
        n (int): Length of the vector
        k (int): Number of leading True values

    Returns:
        ndarray: Boolean array of length n
    """
    v = np.zeros(n, dtype=bool)
    v[:k] = True
    return v

def sb_pairplot(A, clean=5000):
    """
    Generates a Seaborn pairplot of the input data.

    Args:
        A (ndarray): Input matrix of shape (d, n)
        clean (int): Number of non-poisoned data points
    """
    d, n = A.shape
    df = pd.DataFrame(A.T)
    df["poison"] = ~step_vec(n, clean)
    sns.pairplot(df, diag_kind="kde", hue="poison")

def flatten_matrix(A):
    """
    Flattens a square matrix into a vector.

    Args:
        A (ndarray): Square matrix

    Returns:
        ndarray: Flattened 1D array
    """
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    return A.flatten()

def reshape_vector(v):
    """
    Reshapes a vector into a square matrix.

    Args:
        v (ndarray): Input 1D array

    Returns:
        ndarray: Reshaped square matrix
    """
    n = v.size
    m = int(np.sqrt(n))
    assert m * m == n, "Vector length must be a perfect square"
    return v.reshape(m, m)
