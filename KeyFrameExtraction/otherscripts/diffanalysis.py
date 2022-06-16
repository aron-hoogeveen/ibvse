# get_key_frames
# https://github.com/SuhailSaify/Key-Frame-Extraction

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np

# LBP FEATURE EXTRACTION

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb


# get LPBH from image
def lbp_histogram(img, radius):
    # img = color.rgb2gray(color_image)
    patterns = local_binary_pattern(img, 8, 1)
    hist, _ = np.histogram(patterns, bins=np.arange(radius ** 8 + 1), density=True)
    return hist


def lbp_one(im, radius):
    # radius = 2
    n_points = 8 * radius

    lbp = local_binary_pattern(im, n_points, radius)
    lbph = lbp_histogram(im, radius)
    return lbph


def lbp(im, radius):
    f = []
    for i in range(len(im)):
        f.append(lbp_one(im[i]), radius)

    # print(np.array(f))
    print('LBP feature extracted')
    return avg(np.array(f))


##chiSquared Distance
def chiSquared(p, q):
    return 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-6))


def divide_image(img, n, m):
    slice_total = []
    # List of n  image slices
    sliced = np.split(img, n, axis=0)

    # List of m lists of n image blocks
    blocks = [np.split(img_slice, m, axis=1) for img_slice in sliced]

    for i in blocks:
        for j in i:
            slice_total.append(j)

    return np.array(slice_total)


def get_key_frames_1(img_seq, show_plot=False, N=11, M=10, p_factor=0.5, block_size=(6), lbp_r=2, N_frames=20,
                     constant=False):
    # devide images into blocks
    # img_seq=images_dummy_vid
    max_ind = len(img_seq)
    interval = int((N + 1) / 2)

    img_blocks = []
    for i in img_seq:
        img_blocks.append(divide_image((i), block_size, block_size))
    img_blocks = np.array(img_blocks)

    # get LBP for each block
    lbp_image = []
    for i in img_blocks:
        lbp_block = []
        for j in i:
            lbp_block.append(lbp_one(j, lbp_r))
        lbp_image.append(lbp_block)
    lbp_image = np.array(lbp_image)

    # calculate diff from HF and TF value for each block
    diff_total = []

    for i in range(interval):
        diff_total.append(np.zeros((block_size * block_size)))

    for i in range(interval, max_ind - interval):
        sf = lbp_image[i - interval]
        ef = lbp_image[i + interval]
        cf = lbp_image[i]
        diff = []
        for j in range(len(cf)):
            diff.append(chiSquared(cf[j], (sf[j] + ef[j]) / 2))
        diff_total.append(diff)

    for i in range(max_ind - interval, max_ind):
        diff_total.append(np.zeros((block_size * block_size)))

    diff_total = np.array(diff_total)

    f = []
    for i in diff_total:
        f.append(np.sum(np.sort(i)[M:]) / M)
    f = np.array(f)

    c = []
    for i in range(N + 1):
        c.append(0)

    for i in range(N + 1, max_ind - N - 1):
        c.append(f[i] - (f[i - interval] + f[i + interval]) / 2)

    for i in range(max_ind - N - 1, max_ind):
        c.append(0)

    c = np.array(c)

    th = np.mean(c) + (p_factor * (np.max(c) - np.mean(c)))

    th_line = th * np.ones(len(c))

    c = c.clip(min=0)

    if (show_plot):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.subplots(1)
        ax.plot(c, linestyle='--', color='r')
        ax.plot(th_line, linestyle='--')
        plt.xlabel('Frame Number', fontsize=18)
        plt.ylabel('Difference Measure', fontsize=18)
        # ax.set_yticks([])
        # ax.set_xticks([])
        ax.grid(True)
        plt.show()

    if (constant):
        candidate = (sorted(range(len((c - th))), key=lambda i: (c - th)[i])[-N_frames:])
        return np.sort(candidate)
    else:
        ret_f = []
        for i in range(len(c)):
            if (c[i] >= th):
                ret_f.append(i)
        return np.array(ret_f)