cimport numpy as cnp
import numpy as np

cdef class VPTreeNode:
    cdef cnp.float64_t[:] pivot
    cdef double threshold
    cdef int index
    cdef VPTreeNode left
    cdef VPTreeNode right

cdef class VPTree:
    cdef VPTreeNode root

    cdef VPTreeNode build(self, cnp.float64_t[:,:] points, cnp.int64_t[:] index)
    cdef _knn_search(self, VPTreeNode node, list results, cnp.float64_t[:] target, int p)
    
    cdef _range_query_recursive(self, VPTreeNode node, cnp.float64_t[:] query_point, 
                                double radius, list objs, list distances)