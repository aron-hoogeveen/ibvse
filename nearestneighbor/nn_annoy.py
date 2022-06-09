import annoy
import numpy as np
import time


def annoy_build_tree(image_features, forest_size, metric, filename):
    """
    Builds the ANNOY tree with given features
    :param image_features: The features with which the tree should be build
    :param forest_size: The size of the forest
    :param metric: The distance metric ("angular", "euclidean", "manhattan", "hamming", or "dot")
    :param filename: The name of the save file without extension
    :return: A saved file containing the data of the tree
    """
    t1 = time.time()
    t = annoy.AnnoyIndex(len(image_features[0]), metric)
    for i, feature in enumerate(image_features):
        t.add_item(i, feature)
    t.build(forest_size)
    t.save(rf'{filename}.ann')
    t2 = time.time()
    build_time = t2-t1
    return build_time


def annoy_search(image_features, metric, filename, k):
    """
    Searches the ANNOY tree for the closest neighbor to the query image
    :param image_features: The features that should be used for the search
    :param metric: The distance metric ("angular", "euclidean", "manhattan", "hamming", or "dot")
    :param filename: The name of the save file without extension
    :param k: The amount of returned NN
    :return: The k closest neighbors and their distance
    """
    t1 = time.time()

    queries, dim = image_features.shape

    idx = np.zeros((queries, k), dtype=np.int64)
    dist = np.zeros((queries, k), dtype=np.single)

    u = annoy.AnnoyIndex(dim, metric)
    u.load(rf'{filename}.ann')  # super fast, will just mmap the file

    for i, test_feature in enumerate(image_features):
        idx[i,:], dist[i,:] = u.get_nns_by_vector(test_feature, k, include_distances=True)  # will find the 1000 nearest neighbors

    t2 = time.time()
    time_per_query = (t2-t1)/queries
    return idx, dist, time_per_query
