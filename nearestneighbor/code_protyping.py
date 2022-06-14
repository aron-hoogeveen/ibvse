# from scipy.interpolate import NearestNDInterpolator
# import matplotlib.pyplot as plt
# import numpy as np
#
# def method_selector(n_frames_inter, n_queries_inter):
#     methods = ["linear", "faiss_flat_cpu", "faiss_flat_gpu", "faiss_hnsw", "faiss_lsh","faiss_ivf"]
#     n_frames_inter = max(271, min(n_frames_inter, 50000))
#     n_queries_inter = min(n_queries_inter, 1000)
#
#     method_idx = np.load(r".\test_data\interp_data.npy", allow_pickle=True)
#     queries = np.array([np.arange(270,4050, 270)])
#     queries = np.append(queries, np.array([np.arange(4050,50000,4050)]))
#     queries = np.append(queries, 50000)
#
#     n_frames = np.array( [[queries[0]]*1000,[queries[1]]*1000,[queries[2]]*1000,[queries[3]]*1000,[queries[4]]*1000,
#                          [queries[5]]*1000,[queries[6]]*1000,[queries[7]]*1000,[queries[8]]*1000,[queries[9]]*1000,
#                          [queries[10]]*1000,[queries[11]]*1000,[queries[12]]*1000,[queries[13]]*1000,[queries[14]]*1000,
#                          [queries[15]]*1000,[queries[16]]*1000,[queries[17]]*1000,[queries[18]]*1000,[queries[19]]*1000,
#                          [queries[20]]*1000,[queries[21]]*1000,[queries[22]]*1000,[queries[23]]*1000,[queries[24]]*1000,[queries[25]]*1000,[queries[26]]*1000]).flatten()
#     n_queries = np.array([range(1, 1001)]*27).flatten()
#
#
#     interpolfunc_method = NearestNDInterpolator((n_queries, n_frames), method_idx)
#     pts = np.array([n_queries_inter, n_frames_inter])
#     interp_res = round(interpolfunc_method(pts)[0])
#
#     assert interp_res != -1
#
#
#     return methods[interp_res]
#
# import matplotlib as mpl
# if __name__ == "__main__":
#     # queries  = [322, 506, 123, 75, 661, 838, 565, 2, 448, 588]
#     # frames  = [15519, 44, 28959, 9806, 41357, 23857, 6924, 7993, 3192, 932]
#     # for query,frame in zip(queries,frames):
#     #     print(f"{query}, {frame}: {method_selector(frame,query)}")
#     methods = ["linear", "faiss_flat_cpu", "faiss_flat_gpu", "faiss_hnsw", "faiss_lsh", "faiss_ivf"]
#     datax = []
#     datay_sub = []
#     datay = []
#     data = []
#     Z = []
#     total = []
#     for i in range(1, 50002, 500):
#         data = []
#         for j in range(1, 1002, 10):
#             data.append(method_selector(i, j))
#             print(f"{i, j}:/50000,1000")
#         total.append(data)
#
#     Y, X = np.meshgrid(range(1, 1002, 10), range(1, 50002, 500))
#
#     cmap = mpl.cm.Set1
#     norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
#     ax = plt.pcolormesh(Y, X, total, cmap=cmap, clim=(0, 6))
#
#     cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=[0, 1, 2, 3, 4, 5])
#     cbar.set_ticklabels(methods)
#     plt.title("Interpolation of method selector")
#     plt.ylabel("number of keyframes")
#     plt.xlabel("number of queries")
#     plt.xlim([0, 1000])
#     plt.ylim([0, 1000])
#     plt.savefig(r".\test_data\plots\method_selector.png")
#     plt.show()

from nn_main_test import nns
import numpy as np
import os

if __name__ == "__main__":
    np.random.seed(1234)
    frame_features = np.load((os.path.abspath(r'data/frames.npy')))
    frame_labels = np.load((os.path.abspath(r'data/frame_labels.npy')))
    image_features = np.load((os.path.abspath(r'data/images.npy')))
    image_labels = np.load((os.path.abspath(r'data/images_labels.npy')))


    tries = 10
    methods = ['linear', 'faiss_flat_cpu', 'faiss_flat_gpu', 'faiss_hnsw','faiss_lsh']
    n_queries = 100
    n_frames = np.array([np.arange(270,4050, 270)])
    n_frames = np.append(n_frames, np.array([np.arange(4050,50000,4050)]))
    n_frames = np.append(n_frames, 50000)

    filename = r".\test_data\interp_selector_data.csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(f"n_frames, method, buildtime, searchtime, mAP, recall\n")

    if n_queries < 10000:
        r_queries = np.random.choice(len(image_features), n_queries, False)
        image_features_sub = image_features[r_queries]
        image_labels_sub = image_labels[r_queries]

    for n_frame in n_frames:
        if n_frame <= 50000:
            dataset_new_size = n_frame
            dataset_n_classes = 10
            new_dataset_indices = []
            assert dataset_new_size % dataset_n_classes == 0

            for i in range(dataset_n_classes):
                new_dataset_indices.append(
                    np.random.choice(np.where(frame_labels == i)[0], int(dataset_new_size / dataset_n_classes),
                                     replace=False))

            new_dataset_indices_flatten = np.array(new_dataset_indices).flatten()
            frame_features_sub = frame_features[new_dataset_indices_flatten]
            frame_labels_sub = frame_labels[new_dataset_indices_flatten]

        for method in methods:
            data = np.zeros(4,tries)
            for i in range(tries):
                build_time, time_per_query, _, mAP, recall = nns(frame_features_sub, image_features_sub,
                                                                          method, k_percentage=7,
                                                                          frame_labels=frame_labels_sub,
                                                                          image_labels=image_labels_sub)
                build_time_ms = build_time * 1000
                time_per_query_ms = time_per_query * 1000
                data[1,i] = build_time_ms
                data[2,i] = time_per_query
                data[3,i] = mAP
                data[4,i] = recall
            data = data.mean(axis = 1)

            with open(filename, 'a') as f:
                f.write(f"{n_frame},{method},{data[1]},{data[2]},{data[3]},{data[4]}\n")