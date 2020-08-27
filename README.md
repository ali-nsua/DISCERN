# DISCERN: Diversity-based Selection of Centroids for k-Estimation and Rapid Non-stochastic Clustering
This repository contains the implementation of DISCERN in Python.
You can find the paper on [arXiv](https://arxiv.org/abs/1910.05933).

## Examples
```python3
X = load_data() # This is assumed to be a 2-dimensional numpy array, where rows represent data samples.
```
Basic DISCERN instance:
```python3
from DISCERN import DISCERN

di = DISCERN()
di.fit(X)

clustering_labels = di.labels_
cluster_centers = di.cluster_centers_
sse_loss = di.inertia_
```
Fix the number of clusters to a specific number (only use DISCERN to initialize K-Means)
```python3
di = DISCERN(n_clusters=K)
```
Use Spherical K-Means
```python3
di = DISCERN(metric='cosine')
```
Specify an upper bound for the number of clusters
```python3
di = DISCERN(max_iter=max_n_clusters)
```

## Notebooks

A Jupyter notebook is also provided in this repository which applies DISCERN to two of the multivariate datasets in the paper. Stay tuned for another notebook on applying DISCERN to images embedded via ResNet.
