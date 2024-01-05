# KluSIM: Speeding Up k-Medoids Clustering Over Dimensional Data with Metric Access Method

This python package implements the KluSIM, an improvement of the SWAP step of k-medoids clustering.

## Installation

Make sure you have Python 3 installed. Then, execute the following commands:

```bash
pip3 install -r requirements.txt
python3 setup.py build_ext --inplace
```

## Usage
### Input Parameters
The algorithm takes the following input parameters:

- X: The input dataset to be clustered.
- k: The number of clusters to form.
- medoids: Indices of the initial medoids.

### Usage Example
To run the *KluSIM* algorithm, follow these steps:

1. Given a dataset, select the initial *k* medoids using *BUILD* or *k-means++* heuristics.

    ```python

    import kmedoids_initialization
    from sklearn.datasets import make_blobs
    import time
    import numpy as np

    n_samples = 3000
    n_features = 8
    n_clusters = 5
    random_state = 102

    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, 
                        centers=n_clusters, random_state=random_state)

    heuristic = 'BUILD' # e.g. 'BUILD' or 'k-means++'
    medoids = kmedoids_initialization.InitializeMedoids(X, n_clusters, heuristic=heuristic)

    ```

2. Then, set the medoids, and call swap method:

    ```python

    start = time.time()
    ks = klusim.KluSIM(n_clusters)
    ks.set_medoids(medoids)
    ks_results = ks.swap(X)

    print("KluSIM took: %.2f ms" % ((time.time() - start)*1000))
    print("TD with KluSIM: %.3f ms " % fp_results.total_deviation_)

    # Main Competitor
    start = time.time()
    fp = fasterpam.FasterPAM(n_clusters)
    fp.set_medoids(medoids)
    fp_results = fp.swap(X)

    print("FasterPAM took: %.2f ms" % ((time.time() - start)*1000))
    print("TD with FasterPAM: %.3f ms " % fp_results.total_deviation_)
    ```

### Usage Example Script

We have provided a script (*usage_example.py*) on how to use the *KluSIM* algorithm with a sample dataset. To run it, execute the following command:

```bash
    python3 usage_example.py
```