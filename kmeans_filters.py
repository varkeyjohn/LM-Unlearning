import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def kmeans_filter(S, k, max_iters=100):
    """
    Performs k-means filtering on the dataset S.
    
    Args:
        S (ndarray): Data matrix of shape (d, n)
        k (int): Number of clusters
        max_iters (int): Maximum iterations for k-means
    
    Returns:
        ndarray: Boolean mask indicating selected samples
    """
    d, n = S.shape
    S_T = S.T  # Transpose to match sklearn format (n, d)
    
    kmeans = KMeans(n_clusters=k, max_iter=max_iters, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(S_T)
    centers = kmeans.cluster_centers_
    
    # Compute distances to cluster centers
    distances = cdist(S_T, centers, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    
    # Compute mean distance per cluster
    cluster_means = np.array([np.mean(min_distances[labels == i]) for i in range(k)])
    
    # Filtering step: Keep points within a reasonable threshold
    threshold = np.mean(cluster_means)
    return min_distances <= threshold

def robust_kmeans(S, k, eps, max_iters=100):
    """
    Performs robust k-means clustering with outlier filtering.

    Args:
        S (ndarray): Data matrix of shape (d, n)
        k (int): Number of clusters
        eps (float): Tolerance for outliers
        max_iters (int): Maximum iterations for k-means
    
    Returns:
        ndarray: Filtered dataset after robust k-means
    """
    select = kmeans_filter(S, k, max_iters)
    return S[:, select]

def tf_kmeans(S, k, max_iters=100):
    """
    TensorFlow implementation of k-means clustering.
    
    Args:
        S (ndarray): Data matrix of shape (d, n)
        k (int): Number of clusters
        max_iters (int): Maximum iterations
    
    Returns:
        ndarray: Boolean mask indicating selected samples
    """
    d, n = S.shape
    S_T = tf.convert_to_tensor(S.T, dtype=tf.float32)  # Convert to TensorFlow tensor
    
    kmeans = tf.compat.v1.estimator.experimental.KMeans(
        num_clusters=k, use_mini_batch=False
    )
    kmeans.train(lambda: tf.data.Dataset.from_tensor_slices(S_T).batch(n), max_steps=max_iters)
    
    cluster_indices = list(kmeans.predict_cluster_index(S_T).numpy().flatten())
    centers = kmeans.cluster_centers()
    
    distances = tf.norm(S_T[:, None, :] - centers[None, :, :], axis=2)
    min_distances = tf.reduce_min(distances, axis=1)
    
    threshold = tf.reduce_mean(min_distances)
    return min_distances.numpy() <= threshold.numpy()
