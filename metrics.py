import numpy as np


def metric1(H1, H2):
    """
    Calculate M^{-1} \sum_{i=1}^M ||h_i^1 - h_i^2||_2^2
    :param H1: numpy array of shape (d, M) for Case 1, where d is the feature space dimension, M is the number of columns
    :param H2: numpy array of shape (d, M) for Case 2
    :return: scalar value representing the metric
    """
    if H1.shape != H2.shape:
        raise ValueError("H1 and H2 must have the same shape")

    # Compute the Euclidean distance squared for each column and then take the average
    distances = np.linalg.norm(H1 - H2, axis=0) ** 2
    metric = np.mean(distances)

    return metric


def metric2(H1, H2):
    """
    Calculate the squared Frobenius norm of the difference of normalized H^T H matrices
    :param H1: numpy array of shape (d, M) for Case 1, where d is the feature space dimension, M is the number of columns
    :param H2: numpy array of shape (d, M) for Case 2
    :return: scalar value representing the metric
    """
    if H1.shape != H2.shape:
        raise ValueError("H1 and H2 must have the same shape")

    # Compute H^T H for both cases
    HT1H1 = H1.T @ H1
    H2TH2 = H2.T @ H2

    # Normalize by Frobenius norm
    H1TH1_normalized = HT1H1 / np.linalg.norm(HT1H1, 'fro')
    H2TH2_normalized = H2TH2 / np.linalg.norm(H2TH2, 'fro')

    # Compute the squared Frobenius norm of the difference
    difference_matrix = H1TH1_normalized - H2TH2_normalized
    metric = np.linalg.norm(difference_matrix, 'fro') ** 2

    return metric
