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
di = DISCERN(max_n_clusters=1000)
```

## Notebooks

Two Jupyter notebooks are also provided in this repository (see `examples/`). Multivariate applies DISCERN to two of the multivariate datasets in the paper.
The other (ImageNette) applies it to <a href="https://github.com/fastai/imagenette">one of the image datasets in the paper, ImageNette</a>. However, unlike the paper, the notebook uses MoCo <a href="#moco">[1]</a><a href="#mocov2">[2]</a> instead of a labeled-imagenet pretrained ResNet.

Stay tuned for more notebooks.

## References

<div id="moco">
[1] He, Kaiming, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. "Momentum contrast for unsupervised visual representation learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729-9738. 2020. (<a href="https://arxiv.org/abs/1911.05722">arXiv</a> | <a href="https://github.com/facebookresearch/moco/">GitHub</a>) 
</div>
<div id="mocov2">
[2] Chen, Xinlei, Haoqi Fan, Ross Girshick, and Kaiming He. "Improved baselines with momentum contrastive learning." <a href="https://arxiv.org/abs/1911.05722">arXiv preprint arXiv:2003.04297</a> (2020).
</div>
