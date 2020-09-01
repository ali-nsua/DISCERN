import numpy as np


def normalize(XI):
    """
    Normalize input matrix XI by dividing rows by their norms.
    """
    XC = XI.copy()
    norms = np.einsum('ij,ij->i', XI, XI)
    np.sqrt(norms, norms)
    XC /= norms[:, np.newaxis]
    return XC


def cosine_similarity(X, Y=None):
    """
    Creates a self-similarity matrix using cosine similarity
    """
    X_normalized = normalize(X)
    if Y is None:
        return ((1 + np.dot(X_normalized, X_normalized.T)) / 2), X_normalized
    Y_normalized = normalize(Y)
    K = np.dot(X_normalized, Y_normalized.T)
    return ((1 + K) / 2), X_normalized
