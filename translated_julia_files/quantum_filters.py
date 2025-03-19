import numpy as np
import tensorflow as tf
import dkk17

def normalize_states(S):
    """
    Normalizes each column of S to unit length (L2 norm).
    
    Args:
        S (ndarray): Data matrix of shape (d, n)
    
    Returns:
        Tensor: Normalized data matrix
    """
    S_tf = tf.convert_to_tensor(S, dtype=tf.float32)
    return tf.linalg.l2_normalize(S_tf, axis=0)

def quantum_fidelity(S, i, j):
    """
    Computes the quantum fidelity between two state vectors.

    Args:
        S (Tensor): Normalized data matrix of shape (d, n)
        i (int): Index of the first state
        j (int): Index of the second state

    Returns:
        Tensor: Fidelity measure
    """
    return tf.square(tf.tensordot(S[:, i], S[:, j], axes=1))

def quantum_filter(S, threshold=0.9):
    """
    Filters vectors based on quantum fidelity.

    Args:
        S (ndarray): Data matrix of shape (d, n)
        threshold (float): Fidelity threshold for filtering
    
    Returns:
        ndarray: Filtered state vectors
    """
    S_norm = normalize_states(S)
    d, n = S.shape
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            if keep[j] and quantum_fidelity(S_norm, i, j).numpy() > threshold:
                keep[j] = False  # Remove duplicate or highly similar states

    return S[:, keep]

def density_matrix(S):
    """
    Computes the quantum density matrix for the given states.

    Args:
        S (ndarray): Data matrix of shape (d, n)

    Returns:
        Tensor: Density matrix
    """
    S_norm = normalize_states(S)
    return tf.linalg.matmul(S_norm, S_norm, transpose_b=True)

def project_onto_eigenvectors(S, num_vectors=5):
    """
    Projects the data onto the top eigenvectors of its density matrix.

    Args:
        S (ndarray): Data matrix of shape (d, n)
        num_vectors (int): Number of eigenvectors to retain

    Returns:
        ndarray: Projected dataset
    """
    rho = density_matrix(S).numpy()
    eigvals, eigvecs = np.linalg.eigh(rho)
    top_indices = np.argsort(eigvals)[-num_vectors:]  # Select top eigenvectors
    projection_matrix = eigvecs[:, top_indices]
    
    return np.dot(projection_matrix.T, S)

