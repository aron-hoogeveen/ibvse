import sys
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
sys.path.append(os.path.dirname(os.path.abspath('nearestneighbor/main.py')))
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from nn_main_test import cal_mAP, cal_recall

import nn_linear
import nn_faiss


def nns(frame_features_in, image_features_in):
    """
    Performs the nearest neighbor search
    :param frame_features_in: the list of feature vectors from the frames
    :param image_features_in: the list of feature vectors from the images
    :param method: nearest neighbor method to be used
    :param k: the amount of nearest neighbors
    :param annoy_forest_size: forest size for using annoy
    :param annoy_metric: distance metric for using annoy
    :param hnsw_batch_size: batch size for hnsw
    :return: Resulting time needed
    """
    # allocate values
    k_percentage = 7
    min_k = 1

    # set values
    n_frames, _ = frame_features_in.shape
    n_queries, _ = image_features_in.shape

    k = max(min_k, round(k_percentage/100 * n_frames))

    if n_frames < 40:
        k = n_frames

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


def method_selector(n_frames_inter, n_queries_inter):
    methods = ["linear", "faiss_flat_cpu", "faiss_flat_gpu", "faiss_hnsw", "faiss_lsh","faiss_ivf"]
    n_frames_inter = max(271, min(n_frames_inter, 50000))
    n_queries_inter = min(n_queries_inter, 1000)

    method_idx = np.load(r".\test_data\interp_data.npy", allow_pickle=True)
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


    print(len(n_queries))
    print(len(n_frames))
    print(len(method_idx))

    interpolfunc_method = NearestNDInterpolator((n_queries, n_frames), method_idx)
    pts = np.array([n_queries_inter, n_frames_inter])
    interp_res = round(interpolfunc_method(pts)[0])

    assert interp_res != -1

    return methods[interp_res]


# if __name__ == "__main__":
#     methods = ["linear", "faiss_flat_cpu", "faiss_flat_gpu", "faiss_hnsw", "faiss_lsh", "faiss_ivf"]
#     datax = []
#     datay_sub = []
#     datay = []
#     data = []
#     Z = []
#     total = []
#     for i in range(1,50001,50):
#         data = []
#         for j in range(1,1002, 10):
#             data.append(method_selector(i,j))
#             print(f"{i,j}:/50000,1000")
#         total.append(data)
#
#
#     Y, X = np.meshgrid(range(1,1002, 10),range(1,50002,500))
#
#     cmap = mpl.cm.viridis
#     norm = mpl.colors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5,5.5], cmap.N)
#     ax = plt.pcolormesh(Y, X, total, cmap = cmap, clim = (0,6))
#
#     cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks = [0,1,2,3,4,5])
#     cbar.set_ticklabels(methods)
#     plt.title("Interpolation of method selector")
#     plt.ylabel("number of keyframes")
#     plt.xlabel("number of queries")
#     plt.xlim([0,1000])
#     plt.ylim([0,50000])
#     plt.savefig(r".\test_data\plots\method_selector.png")
#     plt.show()

#
#
    # frames = np.load(r".\data\embedded_features.npy")
    # images = np.load(r".\data\embedded_features_test.npy")
    # frame_labels = np.load(r".\data\labels.npy")
    # labels = np.load(r".\data\labels_test.npy")
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