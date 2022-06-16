from nn_main_test import nns
import numpy as np
import os

if __name__ == "__main__":
    np.random.seed(1234)
    frame_features = np.load((os.path.abspath(r'data\frames.npy')))
    frame_labels = np.load((os.path.abspath(r'data\frames_labels.npy')))
    image_features = np.load((os.path.abspath(r'data\images.npy')))
    image_labels = np.load((os.path.abspath(r'data\images_labels.npy')))

    tries = 10
    methods = ['linear', 'faiss_flat_cpu', 'faiss_hnsw','faiss_lsh']
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
            data = np.zeros((4,tries))
            for i in range(tries):
                build_time, time_per_query, _, mAP, recall = nns(frame_features_sub, image_features_sub,
                                                                          method, k_percentage=7,
                                                                          frame_labels=frame_labels_sub,
                                                                          image_labels=image_labels_sub)
                build_time_ms = build_time * 1000
                time_per_query_ms = time_per_query * 1000
                data[0,i] = build_time_ms
                data[1,i] = time_per_query_ms
                data[2,i] = mAP
                data[3,i] = recall
            data = data.mean(axis = 1)

            with open(filename, 'a') as f:
                f.write(f"{n_frame},{method},{data[0]},{data[1]},{data[2]},{data[3]}\n")
            print(f"=> n_frames: {np.where(n_frames == n_frame)[0][0]} of {len(n_frames)-1}\n"
                  f"   method:   {methods.index(method)} of {len(methods)-1} ({method})")


