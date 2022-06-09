import time
import hnswlib
import numpy as np
import pickle


def hnsw_add(image_feature, max_elements, filename="hnswresult", space='l2', ef=10, ef_const=10, init=True, M = 16):
    """
    :param image_feature: array of feature vectors
    :param max_elements: max allowed elements in graph, should be close to total amount of elements
    :param filename: save file name
    :param space: The space metric (l2, cosine or ip)
    :param ef: query time accuracy speed trade-off: higher ef leads to better accuracy but slower search
    :param ef_const: parameter that controls speed/accuracy trade-off during the index construction.
    :param init: Should a new graph be built?
    :return: Save file with encoded graph
    """
    t1 = time.time()
    _, dim = image_feature.shape
    p = hnswlib.Index(space=space, dim=dim)  # possible options are l2, cosine or ip
    if init:
        p.init_index(max_elements=max_elements, ef_construction=ef_const, M=M)
    else:
        p.load_index(f"{filename}.bin", max_elements=max_elements)
        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
    p.set_ef(ef)

    # Set number of threads used during batch search/construction
    # By default using all available cores

    # image_features_array = np.array(image_feature)
    p.add_items(image_feature)
    p.save_index(f"{filename}.bin")
    t2 = time.time()
    build_time = t2 - t1
    return build_time


def hnsw_search(image_features, k, filename="hnswresult", space='l2', ef=10):
    """
    Searches the HNSW Graph
    :param image_features: array of feature vectors
    :param k: amount of NN
    :param filename: save file name
    :param space: The space metric (l2, cosine or ip)
    :param ef: query time accuracy speed trade-off: higher ef leads to better accuracy but slower search
    :return: k closest neighbours and their distances
    """
    t1 = time.time()

    num_test, dim = image_features.shape
    p = hnswlib.Index(space=space, dim=dim)  # possible options are l2, cosine or ip
    p.load_index(f"{filename}.bin")
    p.set_ef(ef)
    idx, distances = p.knn_query(image_features, k)  # will find the k nearest neighbors

    t2 = time.time()
    time_per_query = (t2 - t1) / num_test
    return idx, distances, time_per_query
