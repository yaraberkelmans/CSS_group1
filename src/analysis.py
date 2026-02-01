import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler


def hdbscan_cluster_labels_xytheta(
    model, theta_scale=1.0, min_cluster_size=10, min_samples=None
):
    """Perform HDBSCAN clustering on agents' social positions and opinions."""

    X = np.array([agent.x for agent in model.agents], dtype=float)  # shape (N, d)
    theta = np.array([agent.theta[0] for agent in model.agents], dtype=float).reshape(
        -1, 1
    )

    feats = np.hstack([X, theta_scale * theta])  # (N, d+1)

    feats = StandardScaler().fit_transform(feats)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(feats)
    return labels


def count_big_hdbscan_clusters(labels, min_size=10):
    """Count number of big clusters and their sizes from HDBSCAN labels."""

    labels = np.asarray(labels)
    mask = labels >= 0
    if not mask.any():
        return 0, {}

    ids, counts = np.unique(labels[mask], return_counts=True)
    size_by_id = dict(zip(ids.tolist(), counts.tolist()))
    n_big = sum(c >= min_size for c in counts)
    return n_big, size_by_id
