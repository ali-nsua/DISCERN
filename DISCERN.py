import torch
import numpy as np
from .utils.ops import cosine_similarity, torch_cosine_similarity, torch_unravel_index
from sklearn.cluster import KMeans


class _DISCERN:
    """
    DISCERN is a K-Means initializer, finding suitable centroids for K-Means to begin with and therefore increase
    the probability for quicker convergence. Moreover, DISCERN does not require the number of clusters (K), as it
    can estimate a suitable number on its own.

    NOTE: DISCERN has a very large complexity w.r.t the number of samples, therefore if an upper bound is known for
    the number of clusters, it should be set to improve speed (max_n_clusters).

    Parameters
    ----------
    n_clusters : int or NoneType, default=None
        The target number of clusters. Ignored if not set or set to less than 2.

    max_n_clusters : int or NoneType, default=None
        The maximum number of clusters to consider when estimating the number of clusters, ignored if not set or
        set to less than 2. Setting can largely improve performance when dealing with a large number of samples.

    metric : str, default='euclidean'
        Clustering distance metric, can be either 'euclidean' or 'cosine'.

    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : np.ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.
    """
    def __init__(self, n_clusters=None, max_n_clusters=None, metric='euclidean'):
        self.num_clusters = n_clusters if n_clusters is not None and n_clusters > 1 else None
        self.max_n_clusters = max_n_clusters if max_n_clusters is not None and max_n_clusters > 1 else None
        if metric not in ['cosine', 'euclidean']:
            raise NotImplementedError("Metric `{}` is not supported.".format(metric))
        self.metric = 1 if metric == 'cosine' else 0

        self.similarity_matrix = None
        self.initial_cluster_centers_ = None
        self.km_instance = None

    @property
    def cluster_centers_(self):
        if self.km_instance is None:
            if self.initial_cluster_centers_ is not None:
                return self.initial_cluster_centers_
            return None
        return self.km_instance.cluster_centers_

    @property
    def labels_(self):
        if self.km_instance is None:
            return None
        return self.km_instance.labels_

    @property
    def inertia_(self):
        if self.km_instance is None:
            return None
        return self.km_instance.inertia_

    @property
    def n_iter_(self):
        if self.km_instance is None:
            return None
        return self.km_instance.n_iter_

    def _prep(self, input_data):
        """
        Checks the input data and computes the cosine similarity matrix, returns the data in the correct type if
        the clustering metric is euclidean, and the row-normalized data if the metric is cosine.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        data : array-like
        """
        raise NotImplementedError

    def partial_fit(self, input_data):
        """
        Runs DISCERN initialization alone

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        self
        """
        data = self._prep(input_data)

        init_centroid_idx = self._run_discern()
        self.initial_cluster_centers_ = data[init_centroid_idx, :]

        return self

    def fit(self, input_data):
        """
        Fits K-Means on the input data using DISCERN initialization

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        self
        """
        raise NotImplementedError

    def fit_predict(self, input_data):
        """
        Fits K-Means on the input data using DISCERN initialization, returns the clustering assignments

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        labels_ : array-like
        """
        self.fit(input_data)
        return self.labels_

    def predict(self, input_data):
        """
        Returns the clustering assignments on new input data

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        labels_ : array-like
        """
        data = self._prep(input_data)
        return self.km_instance.predict(data)

    def _run_discern(self):
        """
        Runs DISCERN and estimates the number of clusters and returns the indices of samples fit to be the
        initial centroids.

        This part may seem very confusing if you haven't read the paper yet:
        https://arxiv.org/abs/1910.05933

        Returns
        -------
        centroid_indices : list
        """
        raise NotImplementedError


class DISCERN(_DISCERN):
    """
    DISCERN is a K-Means initializer, finding suitable centroids for K-Means to begin with and therefore increase
    the probability for quicker convergence. Moreover, DISCERN does not require the number of clusters (K), as it
    can estimate a suitable number on its own.

    This implementation is completely based on scipy and therefore runs on the CPU.
    If your device has CUDA-enabled GPUs, the required libraries and PyTorch consider using the torch-based
    implementation.

    NOTE: DISCERN has a very large complexity w.r.t the number of samples, therefore if an upper bound is known for
    the number of clusters, it should be set to improve speed (max_n_clusters).

    Parameters
    ----------
    n_clusters : int or NoneType, default=None
        The target number of clusters. Ignored if not set or set to less than 2.

    max_n_clusters : int or NoneType, default=None
        The maximum number of clusters to consider when estimating the number of clusters, ignored if not set or
        set to less than 2. Setting can largely improve performance when dealing with a large number of samples.

    metric : str, default='euclidean'
        Clustering distance metric, can be either 'euclidean' or 'cosine'.

    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : np.ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.
    """
    def _prep(self, input_data):
        """
        Checks the input data and computes the cosine similarity matrix, returns the data in the correct type if
        the clustering metric is euclidean, and the row-normalized data if the metric is cosine.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        data : array-like
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

    def fit(self, input_data):
        """
        Fits K-Means on the input data using DISCERN initialization

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        self
        """
        data = self._prep(input_data)

        init_centroid_idx = self._run_discern()
        self.initial_cluster_centers_ = data[init_centroid_idx, :]

        self.km_instance = KMeans(n_clusters=len(self.cluster_centers_), init=self.cluster_centers_, n_init=1)
        self.km_instance.fit(data)
        return self

    def _run_discern(self):
        """
        Runs DISCERN and estimates the number of clusters and returns the indices of samples fit to be the
        initial centroids.

        This part may seem very confusing if you haven't read the paper yet:
        https://arxiv.org/abs/1910.05933

        Returns
        -------
        centroid_indices : list
        """
        centroid_idx_0, centroid_idx_1 = np.unravel_index(
            self.similarity_matrix.argmin(), self.similarity_matrix.shape
        )

        centroid_idx = [centroid_idx_0, centroid_idx_1]

        remaining = [y for y in range(0, len(self.similarity_matrix)) if y not in centroid_idx]

        similarity_submatrix = self.similarity_matrix[centroid_idx, :][:, remaining]

        ctr = 2
        max_n_clusters = len(self.similarity_matrix) if self.max_n_clusters is None else self.max_n_clusters
        find_n_clusters = self.num_clusters is None or self.num_clusters < 2

        membership_values = None if not find_n_clusters else np.zeros(max_n_clusters+1, dtype=float)

        # DISCERN initialization for c_3, c_4, ... , c_l, ...
        while len(remaining) > 1 and ctr <= max_n_clusters:
            if self.num_clusters is not None and 1 < self.num_clusters <= len(centroid_idx):
                break

            max_vector = np.max(similarity_submatrix, axis=0)
            min_vector = np.min(similarity_submatrix, axis=0)
            diff_vector = max_vector - min_vector

            membership_vector = np.square(max_vector) * min_vector * diff_vector

            min_idx = int(np.argmin(membership_vector))
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

            predicted_n_clusters = int(np.argmin(kappa))
            n_clusters = max(predicted_n_clusters, 2)
        else:
            n_clusters = self.num_clusters

        centroid_idx = centroid_idx[:n_clusters]

        return centroid_idx


class TorchDISCERN(_DISCERN):
    """
    DISCERN is a K-Means initializer, finding suitable centroids for K-Means to begin with and therefore increase
    the probability for quicker convergence. Moreover, DISCERN does not require the number of clusters (K), as it
    can estimate a suitable number on its own.

    This implementation is completely based on torch and therefore runs on the GPU.
    If your device does not have CUDA-enabled GPUs, the required libraries or PyTorch consider using the scipy-based
    implementation.

    NOTE: DISCERN has a very large complexity w.r.t the number of samples, therefore if an upper bound is known for
    the number of clusters, it should be set to improve speed (max_n_clusters).

    Parameters
    ----------
    n_clusters : int or NoneType, default=None
        The target number of clusters. Ignored if not set or set to less than 2.

    max_n_clusters : int or NoneType, default=None
        The maximum number of clusters to consider when estimating the number of clusters, ignored if not set or
        set to less than 2. Setting can largely improve performance when dealing with a large number of samples.

    metric : str, default='euclidean'
        Clustering distance metric, can be either 'euclidean' or 'cosine'.

    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : np.ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.
    """

    def _prep(self, input_data):
        """
        Checks the input data and computes the cosine similarity matrix, returns the data in the correct type if
        the clustering metric is euclidean, and the row-normalized data if the metric is cosine.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        data : array-like
        """
        data = torch.from_numpy(np.asarray(input_data))
        data.requires_grad = False
        if torch.cuda.is_available():
            data = data.cuda()
        if len(data.shape) < 2:
            data = torch.unsqueeze(data, dim=0)
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        self.similarity_matrix, data_normalized = torch_cosine_similarity(data)

        if self.metric > 0:
            """
            Using normalized data (unit length vectors) with the Euclidean norm is very similar to
            spherical clustering in theory, but the actual results may slightly differ.
            """
            return data_normalized

        return data

    def fit(self, input_data):
        """
        Fits K-Means on the input data using DISCERN initialization

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        self
        """
        data = self._prep(input_data)

        init_centroid_idx = self._run_discern()
        self.initial_cluster_centers_ = data[init_centroid_idx, :]
        # Switch to CPU mode since the matrices should be cast into np.ndarrays
        self.initial_cluster_centers_ = self.initial_cluster_centers_.cpu()
        data = data.cpu()
        self.km_instance = KMeans(n_clusters=len(self.cluster_centers_), init=self.cluster_centers_, n_init=1)
        self.km_instance.fit(data)
        return self

    def _run_discern(self):
        """
        Runs DISCERN and estimates the number of clusters and returns the indices of samples fit to be the
        initial centroids.

        This part may seem very confusing if you haven't read the paper yet:
        https://arxiv.org/abs/1910.05933

        Returns
        -------
        centroid_indices : list
        """
        centroid_idx_0, centroid_idx_1 = torch_unravel_index(int(torch.argmin(self.similarity_matrix)),
                                                             self.similarity_matrix.shape)

        centroid_idx = [centroid_idx_0, centroid_idx_1]

        remaining = [y for y in range(0, len(self.similarity_matrix)) if y not in centroid_idx]

        similarity_submatrix = self.similarity_matrix[centroid_idx, :][:, remaining]

        ctr = 2
        max_n_clusters = len(self.similarity_matrix) if self.max_n_clusters is None else self.max_n_clusters
        find_n_clusters = self.num_clusters is None or self.num_clusters < 2

        membership_values = None if not find_n_clusters else np.zeros(max_n_clusters+1, dtype=float)

        # DISCERN initialization for c_3, c_4, ... , c_l, ...
        while len(remaining) > 1 and ctr <= max_n_clusters:
            if self.num_clusters is not None and 1 < self.num_clusters <= len(centroid_idx):
                break

            max_vector = torch.max(similarity_submatrix, dim=0).values
            min_vector = torch.min(similarity_submatrix, dim=0).values
            diff_vector = max_vector - min_vector

            membership_vector = torch.square(max_vector) * min_vector * diff_vector

            min_idx = int(torch.argmin(membership_vector).item())
            new_centroid_idx = remaining[min_idx]

            if find_n_clusters:
                membership_values[ctr] = float(membership_vector[min_idx].data)

            centroid_idx.append(new_centroid_idx)
            remaining.remove(new_centroid_idx)
            similarity_submatrix = self.similarity_matrix[centroid_idx, :][:, remaining]
            ctr += 1

        if find_n_clusters:
            # TODO: Torch-based finite differences, curvature estimation and K estimation
            # K-Estimation
            membership_values = membership_values[:ctr]
            x = range(0, len(membership_values))

            # Compute the curvature of the function
            dy = np.gradient(membership_values, x)
            d2y = np.gradient(dy, x)
            kappa = (d2y / ((1 + (dy ** 2)) ** (3 / 2)))

            predicted_n_clusters = int(np.argmin(kappa))
            n_clusters = max(predicted_n_clusters, 2)
        else:
            n_clusters = self.num_clusters

        centroid_idx = centroid_idx[:n_clusters]

        return centroid_idx
