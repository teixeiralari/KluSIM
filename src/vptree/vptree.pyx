cimport cython
cimport numpy as cnp
import numpy as np
from src.utils cimport helpers
from src.utils import helpers

cnp.import_array()

cdef class VPTreeNode:
    def __init__(self, cnp.float64_t[:] pivot, double threshold, int index=-1, VPTreeNode left=None, VPTreeNode right=None):
        self.pivot = pivot
        self.left = left
        self.right = right
        self.index = index
        self.threshold = threshold 
        

cdef class VPTree:
    def __init__(self, cnp.float64_t[:,:] points):
        self.root = self.build(points, np.arange(points.shape[0], dtype=np.int64))
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef VPTreeNode build(self, cnp.float64_t[:,:] points, cnp.int64_t[:] index):
        
        cdef int x_max = points.shape[0] - 1
        cdef int y_max = points.shape[1]
        
        if points.shape[0] == 0:
            return None

        # print(np.asarray(points))
        cdef int pivot_index = index[0]
        cdef cnp.float64_t[:] pivot = points[0]

        cdef cnp.int64_t[:] points_index = index[1:]

        points = points[1:]

        cdef cnp.ndarray distances = np.array([helpers.euclidean_distance(points[iterator], pivot) for iterator in range(points.shape[0])])

        cdef cnp.float64_t threshold = np.median(distances) if distances.shape[0] > 0 else 0
        cdef cnp.int64_t[:] right_index = np.zeros((x_max, ), dtype=np.int64) 
        cdef cnp.int64_t[:] left_index = np.zeros((x_max, ), dtype=np.int64)

        cdef cnp.float64_t[:, :] left_points = np.zeros((x_max, y_max), dtype=np.float64)
        cdef cnp.float64_t[:, :] right_points = np.zeros((x_max, y_max), dtype=np.float64)

        cdef int count_idx_right = 0
        cdef int count_idx_left = 0

        for row in range(x_max):
            if distances[row] < threshold:
                left_points[count_idx_left] = points[row]
                left_index[count_idx_left] = points_index[row]
                count_idx_left += 1
            else:
                right_points[count_idx_right] = points[row]
                right_index[count_idx_right] = points_index[row]
                count_idx_right += 1
                
        cdef VPTreeNode node = VPTreeNode(pivot, threshold, pivot_index)
  
        if count_idx_left > 0:
            node.left = self.build(left_points[:count_idx_left], left_index[:count_idx_left])

        if count_idx_right > 0:
            node.right = self.build(right_points[:count_idx_right], right_index[:count_idx_right])

        return node

    def knn(self, target, k):
        results = []
        self._knn_search(self.root, results, target, k)
        return results

    cdef _knn_search(self, VPTreeNode node, list results, cnp.float64_t[:] target, int near_u):
        if node is None:
            return

        distance = helpers.euclidean_distance(target, node.pivot)

        if len(results) < near_u:
            results.append([node.index, distance])
            results.sort(key=lambda x: x[1], reverse=True)

        elif distance < results[0][1]:
            results[0] = [node.index, distance]
            results.sort(key=lambda x: x[1], reverse=True)

        if distance - node.threshold < results[0][1]:
            self._knn_search(node.left, results, target, near_u)

        if distance + results[0][1] >= node.threshold:
            self._knn_search(node.right, results, target, near_u)

    def range_query(self, cnp.float64_t[:] query_point, double radius):
        distances, objs = [], []
        self._range_query_recursive(self.root, query_point, radius, objs, distances)
        return objs, distances
    
    cdef _range_query_recursive(self, VPTreeNode node, cnp.float64_t[:] query_point, 
                                double radius, list objs, list distances):

        if node is None:
            return 
        
        cdef cnp.float64_t distance = helpers.euclidean_distance(query_point, node.pivot)
       
        if distance <= radius:
            objs.append(node.index)
            distances.append(distance)
          
        if distance - node.threshold <= radius:
            self._range_query_recursive(node.left, query_point, radius, objs, distances)

        if node.threshold - distance <= radius:
            self._range_query_recursive(node.right, query_point, radius, objs, distances)
    
    def assurance_similarity_query(self, cnp.float64_t[:,:] X, cnp.int64_t[:] s_q, cnp.int64_t[:] X_idx, int k):

        cdef cnp.float64_t[:] radius = helpers.find_radius(X, s_q)
        cdef list clusters = [None] * k
        cdef double minimum_distances = 0.0
        cdef list all_idx = []
        cdef int obj_idx, medoids_idx, idx
        cdef double dist = 0.0
        cdef double min_dist = np.inf
        cdef int min_medoid_dist = -99

        for idx in range(k):
    
            objects_covered_idx, distances = self.range_query(X[s_q[idx]], radius[idx])
           
            all_idx.extend(objects_covered_idx)
            clusters[idx] = objects_covered_idx
            minimum_distances += np.sum(distances)
 
        cdef cnp.int64_t[:] all_idx_np = np.asarray(all_idx, dtype=np.int64)
        cdef cnp.int64_t[:] objs_not_covered = helpers.set_diff(X_idx, all_idx_np)

        cdef cnp.int64_t objs_not_covered_shape = objs_not_covered.shape[0]

        if objs_not_covered_shape > 0:

            for obj_idx in range(objs_not_covered_shape):
                min_dist = np.inf

                for medoids_idx in range(k):
                    dist = helpers.euclidean_distance(X[objs_not_covered[obj_idx]], X[s_q[medoids_idx]])
                 
                    if dist < min_dist:
                        min_medoid_dist = medoids_idx
                        min_dist = dist
                
                minimum_distances += min_dist
                clusters[min_medoid_dist].append(objs_not_covered[obj_idx])
        
        return minimum_distances, clusters