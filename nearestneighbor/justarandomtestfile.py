# def searchk():
#     # init = True
#     # for ef in range(0,100,50):
#     #     for ef_const in range(0,100,50):
#     #         build_time, start_time, total_time, mAP = nn_main.nns(frame_features, image_features, args.method, args.k, args.forest_size,
#     #         args.annoy_metric, f, args.batch_size, ef, ef_const)
#     #         df = pd.DataFrame({"build_time": build_time, "search_time": start_time, "total_time": total_time, "mAP": mAP, "ef": ef, "ef_const": ef_const}, index=[0])
#     #         df.to_csv((os.path.abspath(r'data/ef_res.csv')), mode='a', header=init)
#     #         init = False
#     start_time = time.process_time()
#     nn_annoy.annoy_build_tree(frame_features, args.forest_size, args.annoy_metric, 'AnnoyTree')  # Build the network
#     build_time = time.process_time()
#     init = True
#     for search_k in range(0, 1000, 50):
#         start_search = time.process_time()
#         nns_res = nn_annoy.annoy_search(image_features, args.annoy_metric, 'AnnoyTree', args.k, search_k)  # Search the network
#         search_time = time.process_time()-start_search
#         total_time = build_time - start_time + search_time
#         mAP = nn_main.cal_mAP(nns_res, frame_labels, image_labels)
#         df = pd.DataFrame({"build_time": build_time, "search_time": search_time, "total_time": total_time, "mAP": mAP}, index=[0])
#         df.to_csv((os.path.abspath(r'data/ef_res.csv')), mode='a', header=init)
#         init = False
#
# def framevstime(frame_features, image_features, method, k, frame_labels, image_labels):
#     init = True
#     for i in range(100,50000, 100):
#         r_frames = np.random.choice(len(frame_features), i, False)
#         r_images = np.random.choice(len(image_features), 1, False)
#         frame_cut = frame_features[r_frames]
#         image_cut = image_features[r_images]
#         labelf_cut = frame_labels[r_frames]
#         labeli_cut = image_labels[r_images]
#
#         build_time, search_time, total_time, mAP = nn_main.nns(frame_cut, image_cut, method, k, frame_labels = labelf_cut, image_labels = labeli_cut, hnsw_batch_size=100)
#         df = pd.DataFrame({"build_time": build_time, "search_time": search_time, "total_time": total_time, "mAP": mAP},
#                           index=[0])
#         df.to_csv((os.path.abspath(r'data/ef_res2.csv')), mode='a', header=init)
#         init = False
#
#
# def frametimeplot():
#     df = pd.read_csv(os.path.abspath(r'data/ef_res2.csv'))
#     df1 = df[['total_time']]
#     ls = np.arange(100, 50000, 100)
#     plt.plot(ls,df1, 'r.')
#     plt.savefig('data/hnswbatchsearchtime.png')
#     plt.show()
#
#
# def HNSWspeedtest():
#     init = True
#     build = 0
#     search_a = []
#     mAP_a = []
#     for i in range(1, 10, 1):
#         frame_features = np.load((os.path.abspath(r'data/embedded_features.npy')))
#         image_features = np.load((os.path.abspath(r'data/embedded_features_test.npy')))[:10, :].reshape(10, 4096)
#         labels = np.load((os.path.abspath(r'data/labels.npy')))
#         labels_test = np.load((os.path.abspath(r'data/labels_test.npy')))[:10]
#         build_time, search, _, mAP = nn_main.nns(frame_features, image_features, "hnsw", 100,
#                                                 hnsw_dim=4096, frame_labels=labels, image_labels=labels_test, build = init)
#         search_a.append(search)
#         mAP_a.append(mAP)
#         print(f'Run {i} out of 100')
#         if init:
#             build = build_time
#             init = False
#
#     print(build)
#     ls = np.arange(1, 200, 2)
#     plt.figure(1)
#     plt.plot(ls, mAP_a, 'r.')
#     plt.savefig('data/aaaaaaaa.png')
#
#     plt.figure(2)
#     ls = np.arange(1, 200, 2)
#     plt.plot(ls, search_a, 'r.')
#     plt.savefig('data/aaaaaaaa1.png')
# import numpy as np
# print(np.linspace(0.1,25,25))
# print(np.logspace(np.log10(0.1),np.log10(25),25))
#import faiss
# faiss.index_cpu_to_gpu()
import numpy as np







get_final_intersection_points()