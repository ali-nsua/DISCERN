import numpy as np
import torch
from torch.nn import functional as F


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


def torch_unravel_index(index, shape):
    """
    Unravel index for torch tensors
    By ModarTensai -- PyTorch Forums
    https://discuss.pytorch.org/u/ModarTensai

    Parameters
    ---------
    index : int
    shape : tuple

    Returns
    -------
    index : tuple
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def torch_cosine_similarity(x):
    """
    Creates a self-similarity matrix using cosine similarity using PyTorch (CUDA supported)

    Parameters
    ---------
    x : torch.Tensor

    Returns
    -------
    cosine_similarity_matrix : torch.Tensor
    x_normalized : torch.Tensor
    """
    x_normalized = F.normalize(x, p=1, dim=1)
    xxt = x_normalized.matmul(x_normalized.T)
    return ((1 + xxt) / 2), x_normalized
