import numpy as np
from .utils.ops import cosine_similarity
from sklearn.cluster import KMeans


class DISCERN:
    """
    DISCERN is a K-Means initializer, finding suitable centroids for K-Means to begin with and therefore increase
    the probability for quicker convergence. Moreover, DISCERN does not require the number of clusters (K), as it
    can estimate a suitable number on its own.

    NOTE: DISCERN has a very large complexity w.r.t the number of samples, therefore if an upper bound is known for
    the number of clusters, it should be set to improve speed (max_n_clusters).
    """
    def __init__(self, n_clusters=0, max_n_clusters=None, metric='euclidean'):
        """
        Initialization
        """
        self.num_clusters = n_clusters  # Manually fix the number of clusters
        self.max_n_clusters = max_n_clusters
        self.metric = 1 if metric == 'cosine' else 0

        self.similarity_matrix = None
        self.labels_ = None
        self.inertia_ = None
        self.cluster_centers_ = None
        self.n_iter_ = None
        self.km_instance = None

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
            """
            Using normalized data (unit length vectors) with the Euclidean norm is very similar to
            spherical clustering in theory, but the actual results may slightly differ.
            """
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

    def fit_predict(self, input_data):
        """
        Fits K-Means on the input data using DISCERN initialization, returns the clustering assignments
        """
        self.fit(input_data)
        return self.labels_

    def predict(self, input_data):
        """
        Returns the clustering assignments on new input data
        """
        data = self._prep(input_data)
        return self.km_instance.predict(data)

    def _run_discern(self):
        """
        Runs DISCERN;
        This part may seem very confusing if you haven't read the paper yet:
        https://arxiv.org/pdf/1910.05933.pdf
        """
        centroid_idx_0, centroid_idx_1 = np.unravel_index(
            self.similarity_matrix.argmin(), self.similarity_matrix.shape
        )

        centroid_idx = [centroid_idx_0, centroid_idx_1]

        remaining = [y for y in range(0, len(self.similarity_matrix)) if y not in centroid_idx]

        similarity_submatrix = self.similarity_matrix[centroid_idx, :][:, remaining]

        ctr = 2
        max_n_clusters = len(self.similarity_matrix) if self.max_n_clusters is None else self.max_n_clusters
        find_n_clusters = self.num_clusters < 2

        if find_n_clusters:
            membership_values = np.zeros(max_n_clusters+1, dtype=float)

        # DISCERN initialization for c_3, c_4, ... , c_l, ...
        while len(remaining) > 1 and ctr <= max_n_clusters:
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

        if find_n_clusters:
            # K-Estimation
            membership_values = membership_values[:ctr]
            x = range(0, len(membership_values))

            # Compute the curvature of the function
            dy = np.gradient(membership_values, x)
            d2y = np.gradient(dy, x)
            kappa = (d2y / ((1 + (dy ** 2)) ** (3 / 2)))

            predicted_n_clusters = np.argmin(kappa)
            n_clusters = max(predicted_n_clusters, 2)
        else:
            n_clusters = self.num_clusters

        centroid_idx = centroid_idx[:n_clusters]

        return centroid_idx

