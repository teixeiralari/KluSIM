cimport numpy as cnp

cdef cnp.float64_t[:,:] pairwise_distance_euclidean(cnp.float64_t[:,:] X)
cdef cnp.float64_t[:] find_radius(cnp.float64_t[:,:] X, cnp.int64_t[:] centroids_idx)
cdef cnp.float64_t euclidean_distance(cnp.float64_t[:] a, cnp.float64_t[:] b) nogil
cdef cnp.int64_t[:] set_diff(cnp.int64_t[:]arr1, cnp.int64_t[:] arr2)
cdef cnp.float64_t[:] pairwise_distance_euclidean_single_point(cnp.float64_t[:] p, cnp.float64_t[:,:] X)