import faiss
import sys
import time
import numpy as np
import argparse
import os
from scipy.interpolate import LinearNDInterpolator, interp1d

def faiss_flat(image_features, frame_features, k, use_gpu = False): # GPU and CPU
    queries, dim = image_features.shape
    t1 = time.time()

    index = faiss.IndexFlatL2(dim)
    index.metric_type = faiss.METRIC_L2
    if use_gpu:
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
        k = min(2048, k)

    index.add(frame_features)
    t2 = time.time()

    dist, idx = index.search(image_features, k = k)
    t3 = time.time()

    build_time = t2-t1
    time_per_query = (t3-t2)/queries

    return idx, dist, build_time, time_per_query


def faiss_hnsw(image_features, frame_features, k, m = 24, ef_const = 58): # does not allow GPU
    queries, dim = image_features.shape
    t1 = time.time()

    index = faiss.IndexHNSWFlat(dim, m)

    index.metric_type = faiss.METRIC_L2
    index.hnsw.efConstruction = ef_const
    index.hnsw.efSearch = k

    index.add(frame_features)
    t2 = time.time()

    dist, idx = index.search(image_features, k = k)
    t3 = time.time()

    build_time = t2-t1
    time_per_query = (t3-t2)/queries

    return idx, dist, build_time, time_per_query

def faiss_lsh(image_features, frame_features, k , bitlength_percentage = 0.25): # GPU and CPU
    queries, dim = image_features.shape
    n_frames, _ = frame_features.shape
    t1 = time.time()

    bitlength_percentage = interpol_lsh(n_frames)

    index = faiss.IndexLSH(dim, int(bitlength_percentage*dim))
    index.metric_type = faiss.METRIC_L2

    index.train(frame_features)
    index.add(frame_features)
    t2 = time.time()

    dist, idx = index.search(image_features, k = k)
    t3 = time.time()

    build_time = t2-t1
    time_per_query = (t3-t2)/queries

    return idx, dist, build_time, time_per_query


def faiss_pq(image_features, frame_features, k, vsplits = 8, nbits = 8): # GPU and CPU
    queries, dim = image_features.shape
    t1 = time.time()

    index = faiss.IndexPQ(dim, vsplits, nbits)
    index.metric_type = faiss.METRIC_L2

    index.train(frame_features)
    index.add(frame_features)
    t2 = time.time()

    dist, idx = index.search(image_features, k = k)
    t3 = time.time()

    build_time = t2-t1
    time_per_query = (t3-t2)/queries

    return idx, dist, build_time, time_per_query


def faiss_ivf(image_features, frame_features, k, use_gpu, splits = 2, nprobe = 1): # GPU and CPU
    queries, dim = image_features.shape
    n_frames, _ = frame_features.shape

    t1 = time.time()

    if n_frames >= 270:
        nprobe, splits = interpol_ivf(n_frames,queries)
        nprobe = int(nprobe)
        splits = int(splits)
    print(nprobe,splits)

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, splits)
    index.metric_type = faiss.METRIC_L2
    index.nprobe = nprobe

    if use_gpu:
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
        k = min(2048, k)

    index.train(frame_features)
    index.add(frame_features)
    t2 = time.time()

    dist, idx = index.search(image_features, k = k)
    t3 = time.time()

    build_time = t2-t1
    time_per_query = (t3-t2)/queries

    return idx, dist, build_time, time_per_query


def interpol_ivf(n_frames_inter, n_queries_inter):
    n_frames = [270, 270, 270, 270, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100]
    n_queries = [1, 122, 123, 1000, 1, 4, 5, 10, 11, 89, 90, 258, 259, 508, 509, 100]
    nprobe = [1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    splits = [2, 2, 5, 5, 4, 4, 4, 4, 4, 4, 6, 6, 8, 8, 12, 12]
    interpolfunc_nprobe = LinearNDInterpolator((n_queries, n_frames), nprobe, fill_value=1)
    interpolfunc_splits = LinearNDInterpolator((n_queries, n_frames), splits, fill_value=12)
    pts = np.array([n_queries_inter, n_frames_inter])
    return interpolfunc_nprobe(pts), interpolfunc_splits(pts)

def interpol_lsh(n_frames_inter):
    n_frames = [270, 8100, 50000]
    bitpercentage = [0.04, 0.09, 0.09]
    interpolfunc_bitpercentage = interp1d(n_frames, bitpercentage, kind='linear', bounds_error=False,
                                          fill_value=(0.04, 0.09))
    pts = np.array(n_frames_inter)
    return interpolfunc_bitpercentage(pts)