cimport cython
import numpy as np
cimport numpy as cnp
from src.vptree cimport vptree
from src.vptree import vptree

cnp.import_array()

cdef class KluSIM():
    cdef cnp.int32_t n_clusters 
    cdef str metric
    cdef cnp.int32_t max_iter 
    cdef public cnp.float64_t[:, :] cluster_centers_
    cdef public cnp.int64_t[:] medoid_indices_
    cdef public cnp.int64_t N
    cdef public cnp.float64_t total_deviation_
    cdef public object data
    cdef public vptree.VPTree tree
    cdef bint medoids_is_set

    def __init__(self, cnp.int32_t n_clusters, str metric='euclidean', cnp.int32_t max_iter=1000):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter

    cdef vptree.VPTree build_tree(self, cnp.float64_t[:,:] X):
        self.tree = vptree.VPTree(X)

    def set_medoids(self, cnp.int64_t[:] medoids):
        self.medoid_indices_ = medoids
        self.medoids_is_set = True

    def swap(self, cnp.float64_t[:,:] X, int p=3):
        self.N = X.shape[0]

        if self.medoids_is_set:
            medoids_idxs = np.copy(self.medoid_indices_) 
        else:
            raise Exception('Medoids are not set.')
        
        dataset_idx = np.arange(self.N, dtype=np.int64)
        
        self.build_tree(X)

        total_deviation, cluster = self.tree.assurance_similarity_query(X,  medoids_idxs, dataset_idx, self.n_clusters)

        for _ in range(self.max_iter):
            swap = self._compute_optimal_swap(
                    X,
                    medoids_idxs,
                    cluster,
                    dataset_idx,
                    total_deviation,
                    self.n_clusters,
                    p
                )

            if swap:
                total_deviation = swap[0]
                medoids_idxs = swap[1]
                cluster = swap[2]
            else:
                break

        self.cluster_centers_ = np.array(X)[medoids_idxs]
        self.medoid_indices_ = medoids_idxs
        self.total_deviation_ = total_deviation
        self.data = cluster
        return self
    
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    cdef _compute_optimal_swap(
            self, 
            cnp.float64_t[:,:] X, 
            cnp.int64_t[:] medoids_idxs, 
            list cluster_idx, 
            cnp.int64_t[:] dataset_idx,  
            cnp.float64_t total_deviation, 
            int k, 
            int p
        ):

        cdef cnp.float64_t best_cost = total_deviation
        cdef list cluster = cluster_idx
        cdef cnp.int64_t[:] best_swaps_medoids = medoids_idxs.copy()
        cdef int m

        for m in range(k):
            id_i = medoids_idxs[m]

            u_i = np.mean(X.base[cluster_idx[m]], axis=0)
            S_p = self.tree.knn(u_i, p)
        
            for o_j in S_p:
    
                medoids_idxs[m] = o_j[0]

                new_cost, new_cluster = self.tree.assurance_similarity_query(X, medoids_idxs, dataset_idx, k)

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_swaps_medoids[m] = o_j[0]
                    cluster = new_cluster
                else:
                    medoids_idxs[m] = id_i

        if best_cost < total_deviation:
            return best_cost, best_swaps_medoids, cluster
        else:
            return None

    def get_labels(self):
        cdef cnp.int64_t[:] labels = np.zeros((self.N,), dtype=np.int64)
        cdef cnp.int64_t o, m

        for m in range(self.n_clusters):
            for o in self.data[m]:
                labels[o] = m

        return np.array(labels)