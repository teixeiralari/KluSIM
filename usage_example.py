import kmedoids_initialization
import fasterpam
import klusim
from sklearn.datasets import make_blobs
import time
import numpy as np

n_samples = 3000
n_features = 8
n_clusters = 5
random_state = 102

X, _ = make_blobs(n_samples=n_samples, n_features=n_features, 
                    centers=n_clusters, random_state=random_state)

# # Method BUILD
medoids = kmedoids_initialization.InitializeMedoids(X, n_clusters, heuristic='BUILD')
start = time.time()
fp = fasterpam.FasterPAM(n_clusters)
fp.set_medoids(medoids)
fp_results = fp.swap(X)

print("BUILD - FasterPAM took: %.2f ms" % ((time.time() - start)*1000))
print("BUILD - TD with FasterPAM: %.3f" % fp_results.total_deviation_)
print("BUILD - FasterPAM Medoids Indices: ", np.asarray(fp_results.medoid_indices_))

start = time.time()
ks = klusim.KluSIM(n_clusters)
ks.set_medoids(medoids)
ks_results = ks.swap(X)

print("BUILD - KluSIM took: %.2f ms" % ((time.time() - start)*1000))
print("BUILD - TD with KluSIM: %.3f" % ks_results.total_deviation_)
print("BUILD - KluSIM Medoids Indices: ", np.asarray(ks_results.medoid_indices_))
# # Method k-means++
medoids = kmedoids_initialization.InitializeMedoids(X, n_clusters, heuristic='k-means++')

start = time.time()
fp = fasterpam.FasterPAM(n_clusters)
fp.set_medoids(medoids)
fp_results = fp.swap(X)

print("k-means++ - FasterPAM took: %.2f ms" % ((time.time() - start)*1000))
print("k-means++ - TD with FasterPAM: %.3f" % fp_results.total_deviation_)
print("k-means++ - FasterPAM Medoids Indices: ", np.asarray(fp_results.medoid_indices_))

start = time.time()
ks = klusim.KluSIM(n_clusters)
ks.set_medoids(medoids)
ks_results = ks.swap(X)

print("k-means++ - KluSIM took: %.2f ms" % ((time.time() - start)*1000))
print("k-means++ - TD with KluSIM: %.3f" % ks_results.total_deviation_)
print("k-means++ - KluSIM Medoids Indices: ", np.asarray(ks_results.medoid_indices_))

