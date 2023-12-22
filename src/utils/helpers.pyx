cimport cython
import numpy as np
cimport numpy as cnp

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.float64_t[:,:] pairwise_distance_euclidean(cnp.float64_t[:,:] X):
    cdef int M = X.shape[0]
    cdef int N = X.shape[1]

    cdef cnp.float64_t[:, :] D = np.empty((M, M), dtype=np.float64)
    
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                d += (X[i, k] - X[j, k]) ** 2
            D[i, j] = d ** 0.5
    return D
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.float64_t[:] pairwise_distance_euclidean_single_point(cnp.float64_t[:] p, cnp.float64_t[:,:] X):
    cdef int M = X.shape[0]
    cdef int N = X.shape[1]
 
    cdef cnp.float64_t[:] D = np.empty((M,), dtype=np.float64)
    
    for i in range(M):
        d = 0.0

        for k in range(N):
            d += (p[k] - X[i, k]) ** 2
        
        D[i] = d ** 0.5

    return D

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.float64_t euclidean_distance(cnp.float64_t[:] a, cnp.float64_t[:] b) nogil:
    cdef double dist = 0.0
    cdef int i
    # print(np.asarray(a), np.asarray(b))
    for i in range(a.shape[0]):
        dist += (a[i] - b[i]) ** 2
    return dist ** 0.5

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.float64_t[:] find_radius(cnp.float64_t[:,:] X, cnp.int64_t[:] centroids_idx):
    cdef int M = centroids_idx.shape[0]
    cdef int N = X.shape[1]
    cdef cnp.float64_t d
    cdef cnp.float64_t[:] radius = np.empty(centroids_idx.shape[0], dtype=np.float64)

    for i in range(M):
        medoid_idx_i = centroids_idx[i]
        min_dist = np.inf

        for j in range(M):
            medoid_idx_j = centroids_idx[j]
            if i != j:
                d = 0.0
                for k in range(N):
                    d += (X[medoid_idx_i, k] - X[medoid_idx_j, k]) ** 2
                
                d = d ** 0.5
                if d < min_dist:
                    min_dist = d

        radius[i] = min_dist/2

    return radius

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.int64_t[:] set_diff(cnp.int64_t[:] arr1, cnp.int64_t[:] arr2):    
    return np.setdiff1d(arr1, arr2, True)
