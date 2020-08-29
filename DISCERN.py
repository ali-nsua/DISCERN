import numpy as np
from .utils import cosine_similarity
from sklearn.cluster import KMeans


class DISCERN:

    def __init__(self, n_clusters=0, max_iter=None, metric='euclidean'):
        """
        Initialization
        """
        self.num_clusters = n_clusters  # Manually fix the number of clusters
        self.max_iter = max_iter
        self.metric = 1 if metric == 'cosine' else 0

        self.similarity_matrix = None
        self.labels_ = None
        self.inertia_ = None
        self.cluster_centers_ = None
        self.n_iter_ = None

    def _prep(self, input_data):
        """
        Checks the input data and computes the cosine similarity matrix
        """
        data = np.asarray(input_data)
        if len(data.shape) < 2:
            data = np.expand_dims(data, axis=0)
        while len(data.shape) > 2:
            data = np.squeeze(data, axis=len(data.shape) - 1)
        self.similarity_matrix, data_normalized = cosine_similarity(data)

        if self.metric > 0:
            return data_normalized

        return data

    def partial_fit(self, input_data):
        """
        Runs DISCERN initialization alone
        """
        data = self._prep(input_data)

        # Run DISCERN
        init_centroid_idx = self._run_discern()
        self.cluster_centers_ = data[init_centroid_idx, :]

    def fit(self, input_data):
        """
        Fits K-Means on the input data using DISCERN initialization
        """
        data = self._prep(input_data)

        # Run DISCERN
        init_centroid_idx = self._run_discern()
        self.cluster_centers_ = data[init_centroid_idx, :]

        # Run K-Means
        self.km_instance = KMeans(n_clusters=len(self.cluster_centers_),
                             init=self.cluster_centers_, n_init=1)
        self.km_instance.fit(data)

        self.labels_ = self.km_instance.labels_
        self.inertia_ = self.km_instance.inertia_
        self.cluster_centers_ = self.km_instance.cluster_centers_
        self.n_iter_ = self.km_instance.n_iter_

    def predict(self, input_data):
        data = self._prep(input_data)
        return self.km_instance.predict(data)

    def _run_discern(self):
        centroid_idx_0, centroid_idx_1 = np.unravel_index(
            self.similarity_matrix.argmin(), self.similarity_matrix.shape
        )

        centroid_idx = [centroid_idx_0, centroid_idx_1]

        remaining = [y for y in range(0, len(self.similarity_matrix)) if y not in centroid_idx]

        similarity_submatrix = self.similarity_matrix[centroid_idx, :][:, remaining]

        ctr = 2
        max_iter = len(self.similarity_matrix) if self.max_iter is None else self.max_iter
        find_n_clusters = self.num_clusters < 2

        if find_n_clusters:
            membership_values = np.zeros(max_iter, dtype=float)

        while len(remaining) > 1 and ctr <= max_iter:
            if 1 < self.num_clusters <= len(centroid_idx):
                break

            max_vector = np.max(similarity_submatrix, axis=0)
            min_vector = np.min(similarity_submatrix, axis=0)
            diff_vector = max_vector - min_vector

            membership_vector = np.square(max_vector) * min_vector * diff_vector

            min_idx = np.argmin(membership_vector)
            new_centroid_idx = remaining[min_idx]

            if find_n_clusters:
                membership_values[ctr] = membership_vector[min_idx]

            centroid_idx.append(new_centroid_idx)
            remaining.remove(new_centroid_idx)
            similarity_submatrix = self.similarity_matrix[centroid_idx, :][:, remaining]
            ctr += 1

        membership_values = membership_values[:ctr]
        if find_n_clusters:
            x = range(0, len(membership_values))

            dy = np.gradient(membership_values, x)
            d2y = np.gradient(dy, x)

            kappa = (d2y / ((1 + (dy ** 2)) ** (3 / 2)))
            predicted_n_clusters = np.argmin(kappa)
            n_clusters = max(predicted_n_clusters, 2)
        else:
            n_clusters = self.num_clusters

        centroid_idx = centroid_idx[:n_clusters]

        return centroid_idx

