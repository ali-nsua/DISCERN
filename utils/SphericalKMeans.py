import numpy as np
from .ops import cosine_similarity


def spherical_k_means(data, cluster_centers_, max_iter=100):
    delta = 1
    itr = 0

    sim_matrix, _ = cosine_similarity(data, cluster_centers_)
    assignments = np.argmax(sim_matrix, axis=1)
    intertia_ = np.sum(np.max(sim_matrix, axis=1))

    while ((delta > 0) and itr < max_iter):
        itr += 1
        previous_intertia_ = intertia_

        new_centers = np.zeros((cluster_centers_.shape))
        cluster_counter = np.zeros(new_centers.shape[0])
        for i in range(0, data.shape[0]):
            lbl = int(assignments[i])
            new_centers[lbl, :] += data[i, :]
            cluster_counter[lbl] += 1

        for j in range(0, new_centers.shape[0]):
            if (cluster_counter[j] > 0):
                new_centers[lbl, :] /= cluster_counter[j]

        cluster_centers_ = new_centers
        sim_matrix, _ = cosine_similarity(data, cluster_centers_)
        assignments = np.argmax(sim_matrix, axis=1)
        intertia_ = np.sum(np.max(sim_matrix, axis=1))

        delta = np.abs(intertia_ - previous_intertia_)

    return assignments.astype(int), cluster_centers_
