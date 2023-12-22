cimport cython
import numpy as np
cimport numpy as cnp
from src.utils cimport helpers
from src.utils import helpers
from sklearn.cluster import kmeans_plusplus

cdef _build( cnp.float64_t[:, :] X, int n_clusters):
    """This function select the initial k medoids, using the BUILD heuristic.

    Args:
        X (float64): Dataset
        n_clusters (int): number of clusters
    
    The code is available in scikit-learn-extra package at:
        https://github.com/scikit-learn-contrib/scikit-learn-extra/blob/main/sklearn_extra/cluster/_k_medoids_helper.pyx#L72
    """    

    cdef int[:] medoid_idxs = np.zeros(n_clusters, dtype = np.intc)
    cdef int sample_size = X.shape[0]
    cdef int[:] not_medoid_idxs = np.arange(sample_size, dtype = np.intc)
    cdef int i, j,  id_i, id_j

    cdef float minimum_dist = np.inf
    cdef int minimum_idx = -1

    for o in range(sample_size):
        D = helpers.pairwise_distance_euclidean_single_point(X[o], X)

        if np.sum(D) < minimum_dist:
            minimum_dist = np.sum(D)
            minimum_idx = o
            Dj = D.copy()
    
    medoid_idxs[0] = minimum_idx
    not_medoid_idxs = np.delete(not_medoid_idxs, medoid_idxs[0])

    cdef int n_medoids_current = 1
    cdef (int, int) new_medoid = (0,0)
    cdef cnp.float64_t cost_change_max
    cdef cnp.float64_t cost_change

    for _ in range(n_clusters -1):
        cost_change_max = 0
        for i in range(sample_size - n_medoids_current):
            id_i = not_medoid_idxs[i]
            cost_change = 0

            for j in range(sample_size - n_medoids_current):
                id_j = not_medoid_idxs[j]
                # print(Dj, id_j)
                Dij = helpers.euclidean_distance(X[id_i], X[id_j])
                cost_change +=   max(0, Dj[id_j] - Dij)

            if cost_change >= cost_change_max:
                cost_change_max = cost_change
                new_medoid = (id_i, i)


        medoid_idxs[n_medoids_current] = new_medoid[0]
        n_medoids_current +=  1
        not_medoid_idxs = np.delete(not_medoid_idxs, new_medoid[1])

        for id_j in range(sample_size):
            # print(X[new_medoid[0])
            Djm = helpers.euclidean_distance(X[id_j], X[new_medoid[0]])
            Dj[id_j] = min(Dj[id_j], Djm)
 
    return np.array(medoid_idxs)

cdef _kmeans_plusplus(X, n_clusters):
    """
    This is a scikit-learn package:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.kmeans_plusplus.html#sklearn.cluster.kmeans_plusplus
    """

    X = np.asarray(X)
    _, medoids_idx = kmeans_plusplus(X, n_clusters)
    return np.array(medoids_idx)

def InitializeMedoids(cnp.float64_t[:, :] X, int n_clusters, str heuristic='BUILD'):
    if heuristic == 'BUILD':
        return _build(X, n_clusters).astype(dtype=np.int64)
    elif heuristic=='k-means++':
        return _kmeans_plusplus(X, n_clusters).astype(dtype=np.int64)
    else:
        raise Exception("Method not implemented. Please select 'BUILD' or 'k-means++'")