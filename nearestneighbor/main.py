import sys
import os
import numpy as np
from scipy.interpolate import NearestNDInterpolator
sys.path.append(os.path.dirname(os.path.abspath('nearestneighbor\main.py')))

import nn_linear
import nn_faiss


def nns(frame_features_in, image_features_in):
    """
    Main function for the nearest neighbour search. Selects a method based on the inputs and performs the search
    :param frame_features_in: The n-dimensional feature vectors of the keyframes
    :param image_features_in: The n-dimensional feature vectors of the query images
    :return: The 7% nearest neighbour frames from the list of frame features
    """
    # allocate values
    k_percentage = 7
    min_k = 1
    min_n_frames = 40

    # set values
    n_frames, _ = frame_features_in.shape
    n_queries, _ = image_features_in.shape
    k = max(min_k, round(k_percentage/100 * n_frames))  # convert percentage to number with a minimum of 1
    if n_frames < min_n_frames:  # omit the k_percentage if below the threshold
        k = n_frames

    # Normalize the data
    frame_features_norm = np.linalg.norm(frame_features_in, axis=1)
    frame_features_norm = np.expand_dims(frame_features_norm, axis=1)
    image_features_norm = np.linalg.norm(image_features_in, axis=1)
    image_features_norm = np.expand_dims(image_features_norm, axis=1)
    frame_features_in = frame_features_in / frame_features_norm
    image_features_in = image_features_in / image_features_norm

    # select method and call the corresponding function
    method = method_selector(n_frames, n_queries)
    if method.lower() == 'linear':
        nns_res, dist, _ = nn_linear.matching_L2(k, frame_features_in, image_features_in)

    elif method.lower() == 'faiss_flat_cpu':
        nns_res, dist, _, _ = nn_faiss.faiss_flat(image_features_in, frame_features_in, k, False)

    elif method.lower() == 'faiss_flat_gpu':
        nns_res, dist, _, _ = nn_faiss.faiss_flat(image_features_in, frame_features_in, k, True)

    elif method.lower() == 'faiss_hnsw':
        nns_res, dist, _, _ = nn_faiss.faiss_hnsw(image_features_in, frame_features_in, k)

    elif method.lower() == 'faiss_lsh':
        nns_res, dist, _, _ = nn_faiss.faiss_lsh(image_features_in, frame_features_in, k)

    elif method.lower() == 'faiss_ivf':
        nns_res, dist, _, _ = nn_faiss.faiss_ivf(image_features_in, frame_features_in, k, False)

    else:
        raise Exception("No method available with that name")

    return nns_res, dist, method


def method_selector(n_frames_inter, n_queries_inter, use_indices = False):
    """
    Selects the method to be used for the NNS
    :param n_frames_inter: The number of keyframes
    :param n_queries_inter: The number of queries
    :param use_indices: Indicates if indices or string should be returned
    :return: The most optimal method that can be used for the search
    """

    # clip the data to fit in the bounds of the selector
    n_frames_inter = max(271, min(n_frames_inter, 50000))
    n_queries_inter = min(n_queries_inter, 1000)

    # load the selector data
    method_idx = np.load(os.path.abspath("./nearestneighbor/test_data/interp_data.npy"), allow_pickle=True)

    # create the x and y coordinates for the interpolation (different amounts of keyframes and query images)
    queries = np.array([np.arange(270,4050, 270)])
    queries = np.append(queries, np.array([np.arange(4050,50000,4050)]))
    queries = np.append(queries, 50000)

    n_frames = np.array([[queries[0]]*1000,[queries[1]]*1000,[queries[2]]*1000,[queries[3]]*1000,[queries[4]]*1000,
                         [queries[5]]*1000,[queries[6]]*1000,[queries[7]]*1000,[queries[8]]*1000,[queries[9]]*1000,
                         [queries[10]]*1000,[queries[11]]*1000,[queries[12]]*1000,[queries[13]]*1000,[queries[14]]*1000,
                         [queries[15]]*1000,[queries[16]]*1000,[queries[17]]*1000,[queries[18]]*1000,[queries[19]]*1000,
                         [queries[20]]*1000,[queries[21]]*1000,[queries[22]]*1000,[queries[23]]*1000,[queries[24]]*1000,
                         [queries[25]]*1000,[queries[26]]*1000]).flatten()
    n_queries = np.array([range(1, 1001)]*27).flatten()

    # Build the interpolation function and get the method for the given number of keyframes and query images
    interpolfunc_method = NearestNDInterpolator((n_queries, n_frames), method_idx)
    pts = np.array([n_queries_inter, n_frames_inter])
    interp_res = interpolfunc_method(pts)[0]
    print(interp_res)
    assert interp_res != -1  # Check if no bad results was given

    # convert index to a method name
    methods = ["linear", "faiss_flat_cpu", "faiss_flat_gpu", "faiss_hnsw", "faiss_lsh","faiss_ivf"]
    return interp_res if use_indices else methods[interp_res]

print(method_selector(270,1))

# WHAT TO DO WITH THESE?
# frames = np.load(r".\data\frames.npy")
# images = np.load(r".\data\images.npy")
# frame_labels = np.load(r".\data\frames_labels.npy")
# labels = np.load(r".\data\images_labels.npy")
#
# t1 = time.time()
# nns_res, dist, method = nns(frames,images)
# print(f"Time: {time.time()-t1}")
# print(cal_mAP(nns_res, frame_labels, labels))
# print(cal_recall(nns_res, frame_labels, labels))
#
# print(f"Method:{method}")
# print(f"result: {frame_labels[nns_res]}")
# print(f"distances: {dist}")
# print(labels)
# print(len(nns_res[0]))
