import os
import pandas as pd
import numpy as np
import argparse

import nn_main_test


def main():
    """
    Call the different plots
    :return: plots of the different metrics to be analyzed
    """
    #'faiss_ivf_cpu','faiss_pq',
    methods = ['linear', 'faiss_flat_cpu', 'faiss_flat_gpu','faiss_hnsw','faiss_lsh']
    n_queries = 100
    k_percentage = 7.5

    timevsk(methods, 25, 10,  'timevsk10')
    timevsk(methods, 25, 20,  'timevsk20')
    timevsk(methods, 25, 30,  'timevsk30')
    timevsk(methods, 25, 40,  'timevsk40')
    timevsk(methods, 25, 50,  'timevsk50')

    #timevsk(methods, 25, 8100, 'timevsk8100')
    # timevsk(methods, 25, 50000,'timevsk50000')
    # annoy_forest_size(k_percentage, n_queries)  # 100 queries at 7.5% nearest neighbors
    # hnsw_m(k_percentage,n_queries)
    # hnsw_batch_size(k_percentage,n_queries,4)


def to_csv(data_names, data, filename, mode = 'w'):
    """
    put data in a csv
    :param data_names: the names of the different data in the data list (["name1","name2"])
    :param data: data for to be stored ([[1,2,3],[4,5,6]])
    :param filename: the name of the savefile
    :return: a csv with the data
    """
    d = dict()
    if isinstance(data, list):
        for name, item in zip(data_names, data):
            d[name] = item
        length = len(data[0])
    else:
        d[data_names] = data
        length = len(data)

    df = pd.DataFrame(data=d, index=range(length))
    df.to_csv(f'test_data/{filename}.csv', index=True, header=True, mode=mode)


def timevsk(methods, max_percentage, size, filename):
    """
    Collect data for each method for the percentage of nearest neighbors to be taken
    :return: a csv with all the collected data
    """
    n_queries = 100
    # Create smaller dataset with equal distribution
    np.random.seed(1234)  # for generating consistent data
    # params to be set dependend on dataset (this case CIFAR-10)
    if size < 50000:
        dataset_n_classes = 10
        new_dataset_indices = []
        assert size % dataset_n_classes == 0

        for i in range(dataset_n_classes):
            new_dataset_indices.append(
                np.random.choice(np.where(frame_labels == i)[0], int(size / dataset_n_classes),
                                 replace=False))

        new_dataset_indices_flatten = np.array(new_dataset_indices).flatten()
        frame_features_subset = frame_features[new_dataset_indices_flatten]
        frame_labels_subset = frame_labels[new_dataset_indices_flatten]
    else:
        frame_features_subset = frame_features
        frame_labels_subset = frame_labels

    r_queries = np.random.choice(len(image_features), n_queries, False)
    image_features_subset = image_features[r_queries]
    image_labels_subset = image_labels[r_queries]

    search_arr = []
    mAP_arr = []
    recall_arr = []
    method_arr = []
    for method in methods:
        init = True
        for i in np.linspace(0.1, max_percentage,50):  # switch out for np.linspace(0.1, max_percentage, 25)? (50...50000)
            k_float = (i / 100) * len(frame_features_subset)
            k = max(1,round(k_float))
            if init:
                build_time, search, _, mAP, recall = nn_main_test.nns(frame_features_subset, image_features_subset, method, k,
                                                                      frame_labels=frame_labels_subset,
                                                                      image_labels=image_labels_subset, build=True)
                init = False
            else:
                build_time, search, _, mAP, recall = nn_main_test.nns(frame_features_subset, image_features_subset, method, k,
                                                                      frame_labels=frame_labels_subset,
                                                                      image_labels=image_labels_subset, build=False)
            search_arr.append(search)
            mAP_arr.append(mAP)
            method_arr.append(method)
            recall_arr.append(recall)
            print(f'Run {i} out of {max_percentage} completed for method {method}')

    data_names = ['method', 'searchtime', 'mAP', 'recall']
    data = [method_arr, search_arr, mAP_arr, recall_arr]
    to_csv(data_names, data, filename)


def annoy_forest_size(k_percentage, n_queries):
    forest_sizes = [1, 5, 10, 25, 50]
    build_times = []
    search_times = []
    mAP_arr = []
    n_frames = []
    recall_arr = []
    forest_size_arr = []
    for i in np.linspace(1, 50000, 10):
        i = round(i)
        rand_queries = np.random.choice(len(image_features), n_queries, False)
        image_features_annoy = image_features[rand_queries]
        image_labels_annoy = image_labels[rand_queries]

        rand_frames = np.random.choice(len(frame_features), round(i), False)
        frame_features_annoy = frame_features[rand_frames]
        frame_labels_annoy = frame_labels[rand_frames]
        for forest_size in forest_sizes:
            build_time, search, _, mAP, recall = nn_main.nns(frame_features_annoy, image_features_annoy , 'annoy',
                                                     k_percentage=k_percentage, annoy_forest_size= forest_size,
                                                     annoy_metric='angular', frame_labels=frame_labels_annoy,
                                                     image_labels=image_labels_annoy , build=True)
            build_times.append(build_time)
            search_times.append(search)
            mAP_arr.append(mAP)
            recall_arr.append(recall)
            n_frames.append(i)
            forest_size_arr.append(forest_size)
            print(f'Run {forest_sizes.index(forest_size)+1} out of {len(forest_sizes)} completed for n_frames {i} ')
    data_names = ['n_frames','forest_size', 'build_time', 'search_time', 'mAP', 'recall']
    data = [n_frames, forest_size_arr, build_times, search_times, mAP_arr, recall_arr]
    to_csv(data_names, data, 'annoy_forest_size')


def hnsw_m(k_percentage, n_queries):
    m_trials = range(4,15,2) # 10 runs
    build_times = []
    search_times = []
    mAP_arr = []
    n_frames = []
    recall_arr = []
    m_arr = []
    for i in np.linspace(1, 50000, 10):
        i = round(i)
        rand_queries = np.random.choice(len(image_features), n_queries, False)
        image_features_hnsw = image_features[rand_queries]
        image_labels_hnsw = image_labels[rand_queries]

        rand_frames = np.random.choice(len(frame_features), round(i), False)
        frame_features_hnsw = frame_features[rand_frames]
        frame_labels_hnsw = frame_labels[rand_frames]

        ef = round(i*k_percentage/100)
        for m in m_trials:
            build_time, search, _, mAP, recall = nn_main_test.nns(frame_features_hnsw, image_features_hnsw, 'hnsw',
                                                                  k_percentage=k_percentage, frame_labels=frame_labels_hnsw,
                                                                  image_labels=image_labels_hnsw, build=True, hnsw_m = m,
                                                                  ef = ef, ef_const= round(ef))
            build_times.append(build_time)
            search_times.append(search)
            mAP_arr.append(mAP)
            recall_arr.append(recall)
            n_frames.append(i)
            m_arr.append(m)
            print(f'Run {m_trials.index(m)+1} out of {len(m_trials)} completed for n_frames {i} ')
    data_names = ['n_frames', 'm', 'build_time', 'search_time', 'mAP', 'recall']
    data = [n_frames, m_arr, build_times, search_times, mAP_arr, recall_arr]
    to_csv(data_names, data, 'hnsw_m')


def hnsw_batch_size(k_percentage, n_queries, M):
    build_times = []
    search_times = []
    mAP_arr = []
    n_frames = []
    recall_arr = []
    batch_size_arr = []
    batch_sizes = np.linspace(1,60,15)
    for i in np.linspace(500, 50000, 10): # batches of minimum 5 (otherwise batches doesnt rlly do anything)
        i = round(i)
        rand_queries = np.random.choice(len(image_features), n_queries, False)
        image_features_annoy = image_features[rand_queries]
        image_labels_annoy = image_labels[rand_queries]

        rand_frames = np.random.choice(len(frame_features), round(i), False)
        frame_features_annoy = frame_features[rand_frames]
        frame_labels_annoy = frame_labels[rand_frames]

        ef = round(i * k_percentage / 100)
        for batch_size_percent in batch_sizes:
            batch_size = round(i*batch_size_percent/100)

            build_time, search, _, mAP, recall = nn_main.nns(frame_features_annoy, image_features_annoy , 'hnsw_batch',
                                                             k_percentage=k_percentage, frame_labels=frame_labels_annoy,
                                                             image_labels=image_labels_annoy, build=True, hnsw_batch_size=batch_size,
                                                             hnsw_m=M, ef_const=100,ef = ef)
            build_times.append(build_time)
            search_times.append(search)
            mAP_arr.append(mAP)
            recall_arr.append(recall)
            n_frames.append(i)
            batch_size_arr.append(batch_size_percent)
            print(f'Run {batch_size_percent} out 100% of {len(batch_sizes)} completed for n_frames {i} ')
    data_names = ['n_frames','batch_size', 'build_time', 'search_time', 'mAP', 'recall']
    data = [n_frames, batch_size_arr, build_times, search_times, mAP_arr, recall_arr]
    to_csv(data_names, data, 'hnsw_batch_size')


def performance_for_n_queries(methods, k_percentage, forest_size, ef, annoy_metric, batch_size):  # to be tested
    input_queries = [1, 10, 25, 50]
    method_arr = []
    n_query_arr = []
    build_times = []
    search_times = []
    mAP_arr = []
    n_frames_arr = []
    for n_queries in input_queries:
        for method in methods:
            for n_frames in np.linspace(1, 50000, 10):
                n_frames = round(n_frames)
                rand_frames = np.random.choice(len(frame_features), round(n_frames), False)
                frame_features_split = frame_features[rand_frames]
                frame_labels_split = frame_labels[rand_frames]

                build, search, _, mAP, recall = nn_main_test.nns(frame_features_split, frame_labels_split, method,
                                                                 k_percentage=k_percentage, annoy_forest_size=forest_size,
                                                                 annoy_metric=annoy_metric, frame_labels=frame_labels,
                                                                 image_labels=image_labels, ef=ef, hnsw_batch_size=batch_size)
                n_query_arr.append(n_queries)
                n_frames_arr.append(n_frames)
                method_arr.append(method)
                build_times.append(build)
                search_times.append(search)
                mAP_arr.append(mAP)

            print(f'Run {input_queries.index(n_queries)} out of {len(input_queries)} completed for method {method} ')
    data_names = ['n_queries','method','n_frames','build_time','search_time','mAP']
    data = [n_query_arr, method_arr, n_frames_arr, build_times, search_times, mAP_arr]
    to_csv(data_names, data, 'finalperformance')


if __name__ == "__main__":
    # definition of parser arguments
    a = argparse.ArgumentParser()
    a.add_argument("--n_frames", default=50000, type=int, help="amount of input frames")
    a.add_argument("--n_queries", default=100, type=int, help="amount of queries")
    a.add_argument("--method", default="hnsw_batch", type=str, help="The NNS method to be used")
    a.add_argument("--k", default=100, type=int, help="The amount of nearest neighbors")

    # more for testing purposes
    a.add_argument("--forest_size", default=10, type=int)
    a.add_argument("--annoy_metric", default="angular", type=str)
    a.add_argument("--batch_size", default=500, type=int)
    a.add_argument("--specify_n", default=False, type=bool)
    args = a.parse_args()

    frame_features = np.load((os.path.abspath(r'data/embedded_features.npy')))
    frame_labels = np.load((os.path.abspath(r'data/labels.npy')))
    image_features = np.load((os.path.abspath(r'data/embedded_features_test.npy')))
    image_labels = np.load((os.path.abspath(r'data/labels_test.npy')))


    # # select a random number of queries
    # if args.n_queries < 10000 :
    #     r_queries = np.random.choice(len(image_features), args.n_queries, False)
    #     image_features = image_features[r_queries]
    #     image_labels = image_labels[r_queries]
    #
    # # select a number of frames
    # if args.n_frames < 50000 and args.specify_n:
    #     r_frames = np.random.choice(len(frame_features), args.n_frames, False)
    #     frame_features = frame_features[r_frames]
    #     frame_labels = frame_labels[r_frames]

    main()



