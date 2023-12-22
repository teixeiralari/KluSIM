cimport cython
import numpy as np
cimport numpy as cnp
from src.utils cimport helpers
from src.utils import helpers

cnp.import_array()

cdef class Data:
    cdef public cnp.float64_t nearest
    cdef public cnp.float64_t second
    cdef public cnp.float64_t distance_nearest
    cdef public cnp.float64_t distance_second_nearest

    def __init__(self, cnp.float64_t nearest=np.inf, cnp.float64_t second=np.inf, cnp.float64_t distance_nearest=np.inf, cnp.float64_t distance_second_nearest=np.inf):
        self.nearest = nearest
        self.second = second
        self.distance_nearest = distance_nearest
        self.distance_second_nearest = distance_second_nearest

    def get_nearest_idx(self):
        return int(self.nearest)

    def get_nearest_distance(self):
        return self.distance_nearest

    def get_second_nearest_idx(self):
        return int(self.second)
    
    def get_second_nearest_distance(self):
        return self.distance_second_nearest

    def set_nearest(self, cnp.float64_t i, cnp.float64_t d):
        self.nearest = i
        self.distance_nearest = d

    def set_second_nearest(self, cnp.float64_t i, cnp.float64_t d):
        self.second = i
        self.distance_second_nearest = d

cdef class FasterPAM():
    cdef cnp.int32_t n_clusters 
    cdef str metric
    cdef cnp.int32_t max_iter 
    cdef public cnp.float64_t[:, :] cluster_centers_
    cdef public cnp.int64_t[:] medoid_indices_
    cdef public cnp.float64_t total_deviation_
    cdef public cnp.ndarray data
    cdef bint medoids_is_set

    def __init__(self, cnp.int32_t n_clusters, str metric='euclidean', cnp.int32_t max_iter=1000):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter

    def set_medoids(self, cnp.int64_t[:] medoids):
        self.medoid_indices_ = medoids
        self.medoids_is_set = True

    def swap(self, cnp.float64_t[:,:] X):

        if self.medoids_is_set:
            medoids_idxs = np.copy(self.medoid_indices_) 
            loss, data = self.initial_assigment(X, medoids_idxs)
        else:
            raise Exception('Medoids are not set.')
                    
        cdef cnp.float64_t[:] removal_loss = self.update_removal_loss(data)
        cdef cnp.int32_t x_last = -99
        cdef cnp.int32_t n_swap = 0
        cdef cnp.int32_t x_c, swaps_before

        for _ in range(self.max_iter):
            swaps_before = n_swap

            for x_c in range(X.shape[0]):

                if x_c == x_last:
                    break

                if x_c == medoids_idxs[data[x_c].get_nearest_idx()]:
                    continue

                i, ploss = self.find_best_swap(removal_loss, X, x_c, data)

                if ploss >= 0:
                    continue

                x_last = x_c
                n_swap += 1

                newloss, medoids_idxs, data = self.do_swap(
                    X, medoids_idxs, data, i, x_c)

                if newloss >= loss:
                    break

                loss = newloss
                removal_loss = self.update_removal_loss(data)

            if n_swap == swaps_before:
                break

        self.cluster_centers_ = np.array(X)[medoids_idxs]
        self.medoid_indices_ = medoids_idxs
        self.total_deviation_ = loss
        self.data = data
        return self

    cdef initial_assigment(self, cnp.float64_t[:,:] X, cnp.int64_t[:] medoids_idx):

        cdef cnp.ndarray data = cnp.ndarray((X.shape[0],), dtype=Data)
        cdef cnp.int64_t first_medoid = medoids_idx[0]

        cdef cnp.float64_t loss = 0.0
        cdef cnp.int64_t o, m, medoid_idx

        for o in range(X.shape[0]):
            first_d = helpers.euclidean_distance(X[o], X[first_medoid])
            
            obj = Data(nearest=0, distance_nearest=first_d)
            
            for m in range(1, medoids_idx.shape[0]):        
                medoid_idx = medoids_idx[m]
                d = helpers.euclidean_distance(X[o], X[medoid_idx])
                
                if (d < obj.distance_nearest) or (o == medoid_idx):
                    obj.second = obj.nearest
                    obj.distance_second_nearest = obj.distance_nearest
                    obj.nearest = m
                    obj.distance_nearest = d
                elif (obj.distance_second_nearest == np.inf) or (d < obj.distance_second_nearest):
                    obj.second = m
                    obj.distance_second_nearest = d
                    
            loss += obj.distance_nearest
            data[o] = obj
        return loss, data

    cdef do_swap(self, cnp.float64_t[:,:] X, cnp.int64_t[:] medoids_idx, cnp.ndarray data, cnp.int32_t b, cnp.int32_t j):

        medoids_idx[b] = j
        cdef cnp.float64_t loss = 0.0
        cdef cnp.int64_t o

        for o in range(data.shape[0]):

            if o == j:
                if data[o].get_nearest_idx() != b:
                    data[o].set_second_nearest(data[o].get_nearest_idx(), data[o].get_nearest_distance())
                   
                data[o].set_nearest(b, 0.0)
            
            djo = helpers.euclidean_distance(X[j], X[o]) #X[j, o]
            
            # Nearest medoid is gone:
            if data[o].get_nearest_idx() == b:
                if djo < data[o].get_second_nearest_distance():
                    data[o].set_nearest(b, djo)
    
                else:
                    data[o].set_nearest(data[o].get_second_nearest_idx(), data[o].get_second_nearest_distance())

                    i, d = self.update_second_nearest(
                        X, medoids_idx, data[o].get_nearest_idx(), b, o, djo)
                    
                    data[o].set_second_nearest(i, d)
            else:
                if djo < data[o].get_nearest_distance():
                    data[o].set_second_nearest(data[o].get_nearest_idx(), data[o].get_nearest_distance())
                    data[o].set_nearest(b, djo)
    
                
                elif djo < data[o].get_second_nearest_distance():
                    data[o].set_second_nearest(b, djo)
    
                    
                elif data[o].get_second_nearest_idx() == b:
                    
                    i, d = self.update_second_nearest(
                        X, medoids_idx, data[o].get_nearest_idx(), b, o, djo)
                    
                    data[o].set_second_nearest(i, d)
                   
            loss += data[o].get_nearest_distance()

        return loss, medoids_idx, data

    cdef update_second_nearest(self, cnp.float64_t[:,:] X, cnp.int64_t[:] medoids_idx, cnp.int32_t nearest, cnp.int32_t b, cnp.int32_t o, cnp.float64_t djo):
        second_idx, second_distance = (b, djo)
        
        for m in range(self.n_clusters):
            if m == nearest or m == b:
                continue

            d = helpers.euclidean_distance(X[o], X[medoids_idx[m]])

            if d < second_distance:
                second_idx, second_distance = (m, d)

        return second_idx, second_distance

    cdef cnp.float64_t[:] update_removal_loss(self, cnp.ndarray data):
        cdef cnp.float64_t[:] loss = np.zeros(self.n_clusters, dtype=np.float64)

        for o in range(data.shape[0]):
            
            loss[data[o].get_nearest_idx()] += data[o].get_second_nearest_distance() - \
                data[o].get_nearest_distance()

        return loss

    cdef find_best_swap(self, cnp.float64_t[:] loss, cnp.float64_t[:,:] X, cnp.int32_t j, cnp.ndarray data):
        cdef cnp.float64_t[:] ploss = np.copy(loss)
        cdef cnp.float64_t delta_xc = 0.0
        cdef cnp.int32_t o
        
        for o in range(data.shape[0]):
            d = helpers.euclidean_distance(X[j], X[o]) #X[j, o]

            if d < data[o].get_nearest_distance():
                delta_xc += d - data[o].get_nearest_distance()
                ploss[data[o].get_nearest_idx()] += data[o].get_nearest_distance() - \
                    data[o].get_second_nearest_distance()

            elif d < data[o].get_second_nearest_distance():
                ploss[data[o].get_nearest_idx()] += d - data[o].get_second_nearest_distance()

        # print(ploss)
        i = np.argmin(ploss)
        bloss = ploss[i] + delta_xc
        return i, bloss

    def get_labels(self):
        cdef cnp.int64_t[:] labels = np.zeros((self.data.shape[0],), dtype=np.int64)
        cdef cnp.int64_t o

        for o in range(self.data.shape[0]):
            labels[o] = self.data[o].nearest

        return np.array(labels)