import pandas as pd

from nn_main_test import nns
from main import method_selector
import numpy as np
import os
#
# if __name__ == "__main__":
#     np.random.seed(1234)
#     frame_features = np.load((os.path.abspath(r'data\frames.npy')))
#     frame_labels = np.load((os.path.abspath(r'data\frames_labels.npy')))
#     image_features = np.load((os.path.abspath(r'data\images.npy')))
#     image_labels = np.load((os.path.abspath(r'data\images_labels.npy')))
#
#     tries = 10
#     methods = ['linear', 'faiss_flat_cpu', 'faiss_hnsw','faiss_lsh']
#     n_queries = 100
#     n_frames = np.array([np.arange(270,4050, 270)])
#     n_frames = np.append(n_frames, np.array([np.arange(4050,50000,4050)]))
#     n_frames = np.append(n_frames, 50000)
#
#     filename = r".\test_data\interp_selector_data.csv"
#     if not os.path.exists(filename):
#         with open(filename, 'w') as f:
#             f.write(f"n_frames, method, buildtime, searchtime, mAP, recall\n")
#
#     if n_queries < 10000:
#         r_queries = np.random.choice(len(image_features), n_queries, False)
#         image_features_sub = image_features[r_queries]
#         image_labels_sub = image_labels[r_queries]
#
#     for n_frame in n_frames:
#         if n_frame <= 50000:
#             dataset_new_size = n_frame
#             dataset_n_classes = 10
#             new_dataset_indices = []
#             assert dataset_new_size % dataset_n_classes == 0
#
#             for i in range(dataset_n_classes):
#                 new_dataset_indices.append(
#                     np.random.choice(np.where(frame_labels == i)[0], int(dataset_new_size / dataset_n_classes),
#                                      replace=False))
#
#             new_dataset_indices_flatten = np.array(new_dataset_indices).flatten()
#             frame_features_sub = frame_features[new_dataset_indices_flatten]
#             frame_labels_sub = frame_labels[new_dataset_indices_flatten]
#
#         for method in methods:
#             data = np.zeros((4,tries))
#             for i in range(tries):
#                 build_time, time_per_query, _, mAP, recall = nns(frame_features_sub, image_features_sub,
#                                                                           method, k_percentage=7,
#                                                                           frame_labels=frame_labels_sub,
#                                                                           image_labels=image_labels_sub)
#                 build_time_ms = build_time * 1000
#                 time_per_query_ms = time_per_query * 1000
#                 data[0,i] = build_time_ms
#                 data[1,i] = time_per_query_ms
#                 data[2,i] = mAP
#                 data[3,i] = recall
#             data = data.mean(axis = 1)
#
#             with open(filename, 'a') as f:
#                 f.write(f"{n_frame},{method},{data[0]},{data[1]},{data[2]},{data[3]}\n")
#             print(f"=> n_frames: {np.where(n_frames == n_frame)[0][0]} of {len(n_frames)-1}\n"
#                   f"   method:   {methods.index(method)} of {len(methods)-1} ({method})")

if __name__ == "__main__":
    np.random.seed(1234)
    # Can be adjusted for random/uniform data and the amount of validations
    random = True
    n_validations = 100

    frame_features = np.load((os.path.abspath(r'data/frames.npy')))
    frame_labels = np.load((os.path.abspath(r'data/frames_labels.npy')))
    image_features = np.load((os.path.abspath(r'data/images.npy')))
    image_labels = np.load((os.path.abspath(r'data/images_labels.npy')))
    methods = ['linear', 'faiss_flat_cpu', 'faiss_hnsw','faiss_lsh','faiss_ivf_cpu']

    if random:
        n_queries = np.random.randint(1,1000, n_validations)
        n_frames = np.random.randint(1,50000, n_validations)
    else:
        assert int(np.sqrt(n_validations))**2 == n_validations
        step_size_queries, step_size_frames = int(1000/int(np.sqrt(n_validations))), int(50000/int(np.sqrt(n_validations)))
        n_queries, n_frames = np.meshgrid(range(1,1000,step_size_queries),range(1,50000, step_size_frames))

    fastest_count = n_validations  # decrease every failure
    mAP_recall_count = n_validations  # decrease every failure
    problematic_method_selection = []
    problematic_mAP_recall = []
    for i, (query, frame) in enumerate(zip(n_queries,n_frames)):
        r_queries = np.random.choice(len(image_features), query, False)
        image_features_selec = image_features[r_queries]
        image_labels_selec = image_labels[r_queries]

        r_frames = np.random.choice(len(frame_features), frame, False)
        frame_features_selec = frame_features[r_frames]
        frame_labels_selec = frame_labels[r_frames]

        method_selected = method_selector(frame,query)
        total_time = []

        for method in methods:
            build_time, time_per_query, _, mAP, recall = nns(frame_features_selec,image_features_selec,method,
                                                                      k_percentage=7, image_labels= image_labels_selec,
                                                                      frame_labels=frame_labels_selec)
            total_time.append(build_time + time_per_query*query)
            if method == method_selected:
                if mAP <= 0.65 or recall <= 0.5:
                    mAP_recall_count -= 1
                    problematic_mAP_recall.append([query, frame, method, mAP, recall])

        method_fastest = methods[total_time.index(min(total_time))]
        if method_fastest != method_selected:
            fastest_count -= 1
            problematic_method_selection.append([query,frame,method_selected,method_fastest])

        print(f"Validation {i+1}/{n_validations} completed")

    pms = pd.DataFrame(problematic_method_selection, columns = ["n_query","n_frames","method_selected","method_fastest"])
    pmr = pd.DataFrame(problematic_mAP_recall, columns = ["n_query","n_frames","method_selected","mAP", "recall"])
    pms.to_csv(r".\test_data\problematic_method_selection.csv", index= False)
    pmr.to_csv(r".\test_data\problematic_mAP_recall.csv",index= False)

    print(f"===============================================\n"
          f"Fastest method selection correct for :{fastest_count}/{n_validations}\n"
          f"mAP and recall correct for :{mAP_recall_count}/{n_validations}\n"
          f"===============================================")

