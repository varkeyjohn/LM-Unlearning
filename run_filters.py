import numpy as np
import tensorflow as tf
from kmeans_filters import kmeans_filter
from quantum_filters import quantum_filter, project_onto_eigenvectors
from utils import pca, sb_pairplot, step_vec


def generate_random_data(d=10, n=100, seed=42):
    """
    Generates a random dataset.

    Args:
        d (int): Dimension of data
        n (int): Number of samples
        seed (int): Random seed for reproducibility

    Returns:
        ndarray: Random dataset of shape (d, n)
    """
    np.random.seed(seed)
    return np.random.randn(d, n).astype(np.float32)

def main():
    # Load or generate data
    S = generate_random_data(d=10, n=100)

    # Apply K-means filtering
    kmeans_filtered_S = kmeans_filter(S, num_clusters=5)

    # Apply Quantum filtering
    quantum_filtered_S = quantum_filter(kmeans_filtered_S, threshold=0.9)

    # Project onto top eigenvectors
    projected_S = project_onto_eigenvectors(quantum_filtered_S, num_vectors=5)

    # Print shapes of filtered outputs
    print("Original Data Shape:", S.shape)
    print("K-Means Filtered Shape:", kmeans_filtered_S.shape)
    print("Quantum Filtered Shape:", quantum_filtered_S.shape)
    print("Projected Shape:", projected_S.shape)

if __name__ == "__main__":
    main()
