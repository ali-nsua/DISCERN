import numpy as np
from sklearn import preprocessing


def normalize(XI):
    XC = XI.copy()
    norms = np.einsum('ij,ij->i', XI, XI)
    np.sqrt(norms, norms)
    XC /= norms[:, np.newaxis]
    return XC


def cosine_similarity(X, Y=None):
    X_normalized = normalize(X)
    if Y is None:
        return ((1 + np.dot(X_normalized, X_normalized.T)) / 2), X_normalized
    Y_normalized = normalize(Y)
    K = np.dot(X_normalized, Y_normalized.T)
    return ((1 + K) / 2), X_normalized


def purity_score(y_true, y_pred):
    # Encoding the true labels, just to be on the safe side
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y_true)

    # Calculate purity score
    num_class = len(np.unique(y_true))
    num_clusters = len(np.unique(y_pred))
    lbl = np.unique(y_pred)
    scores = np.zeros((num_class, num_clusters))
    for i in range(0, len(y)):
        scores[y[i], np.where(lbl == y_pred[i])[0]] += 1
    acc = np.sum(np.max(scores, axis=0))
    acc /= len(y)
    return acc
