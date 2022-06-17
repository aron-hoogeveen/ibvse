# fidelity measure based on semiHausdorff distance
import numpy as np
import cv2
import scipy
import os
import glob
import random
from main import *
from scipy.spatial import distance
from math import atan2

def fidelity_descriptors(path):
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    histnorm = width * height  # to normalize color histogram with
    downscale = 0.5  # downsize frame for lower computation for hogs
    fdnorm = histnorm * (downscale) ** 2  # to normalize edge detection histogram

    fd_data = []
    hist_data = []
    cnt = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        # print("Creating histograms for frame " + str(cnt))
        # fd, hog_image = hog(frame, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,channel_axis=-1)
        fd = calculateHOG(frame, downscale)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
        # hist = cv2.normalize(hist, None).flatten()
        hist_data.append(hist)
        fd_data.append(fd)
        cnt += 1
        if cnt % 50 == 0:
            print("created hog for frame: " + str(cnt))
        # if cnt > 10:
        #     break
    print("end of histogram creation")
    return fd_data, hist_data, fdnorm, histnorm

def fidelity(kf_indices, path, vseq_hists, vseq_hogs, fdnorm, histnorm):
    """
    Computes fidelity for a chosen selection of keyframes
    """
    # cap = cv2.VideoCapture(os.path.abspath(os.path.expanduser(sys.argv[1])))
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # video_fps = cap.get(cv2.CAP_PROP_FPS)
    # width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # histnorm = width*height # to normalize color histogram with
    # downscale = 0.1 # downsize frame for lower computation for hogs
    # fdnorm = histnorm*(downscale)**2 # to normalize edge detection histogram

    #decode keyframe indices from video into array
    #keyframes = keyframes_indices_to_array(kf_indices, path, video_fps, frame_count)

    #calculate histograms
    #[keyframes_hogs, vseq_hogs, keyframes_hists, vseq_hists] = calculateHists(keyframes, path, video_fps, cap, downscale)
    maxdiff = 1 # maximum value difference() can return
    maxdist = 0
    for i in range(0, len(vseq_hists)):
        distance = maxdiff
        for j in range(0, len(kf_indices)):
            diff = difference(vseq_hists[i], vseq_hists[kf_indices[j]], vseq_hogs[i], vseq_hogs[kf_indices[j]], fdnorm, histnorm)
            if diff < distance:
                distance = diff
        if (distance > maxdist):
            maxdist = distance

    return maxdiff - maxdist

# def distance_inner(selection, nf, path,  hist_sel, hist_data, maxdiff, fd_sel, fd_data):
#     difference_list = {}
#     for i in range(0, len(hist_sel)):
#         difference_list[i] = difference(hist_sel[i], hist_d, fd_sel[i], fd_d)
#     #print(difference_list)
#     #maxdiff = check_maxdifference(difference_list,maxdiff)
#     distance  = minimalvalue(difference_list)
#     return distance
#
# # def check_maxdifference(array, maxdiff):
# #     for i in range(1, len(array)):
# #         if array[i] > maxdiff:
# #             maxdiff = array[i]
# #     return maxdiff
#
# def minimalvalue(array):
#     min = array[0]
#     for i in range(1, len(array)):
#         if array[i] < min:
#             min = array[i]
#     return min
def calculateHOG(frame, downsize):
    #resize image for computational speed gain
    scale_percent = downsize  # percent of original size
    width = int(frame.shape[1] * scale_percent)
    height = int(frame.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    # Gaussian blur (kernel size = 3)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    # Convert to grayscale (= luminance)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gradientmax = 255 * 4  # maximum gradient value
    threshold = gradientmax * 0.3  # threshold to reduce background noise
    # apply sobel filters,
    grad_x = np.array(cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT))
    grad_y = np.array(cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT))

    # calculate absolute values
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Gradient magnitude calculated by 0.5(|G_x|+|G_y|) instead of sqrt(G_x^2 + G_y^2)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    histogram = np.zeros(72, dtype="float32")
    for i in range(0, grad.shape[0]):
        for j in range(0, grad.shape[1]):
            if grad[i][j]:
                angle = atan2(grad_y[i][j], grad_x[i][j]) / 3.14
                histogram[round(abs(angle * 71))] += int(1)
    return histogram

def calculateHists(keyframes, path, video_fps, cap, downscale):
    #parameters



    fd_sel = []
    hist_sel = []
    print("Creating hogs")
    for frame in keyframes:
        #print("Creating hog for frame")
        #fd, hog_image = hog(frame, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,channel_axis=-1)
        fd = calculateHOG(frame, downscale)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # fd = hog.compute(image)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
        #hist = cv2.normalize(hist, None).flatten()
        hist_sel.append(hist)
        fd_sel.append(fd)

    fd_data = []
    hist_data = []
    cnt = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        #print("Creating histograms for frame " + str(cnt))
        #fd, hog_image = hog(frame, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,channel_axis=-1)
        fd = calculateHOG(frame, downscale)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
        #hist = cv2.normalize(hist, None).flatten()
        hist_data.append(hist)
        fd_data.append(fd)
        cnt += 1
        if cnt%50 == 0:
            print("created hog for frame: " + str(cnt))
        # if cnt > 10:
        #     break
    print("end of histogram creation")
    return fd_sel, fd_data, hist_sel, hist_data


def difference(hist1, hist2, fd1, fd2, fd_norm, hist_norm):
    d_h = 1-cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)/hist_norm
    #print(d_h)
    #d_d = 1-cv2.compareHist(fd1, fd2, cv2.HISTCMP_INTERSECT)/fd_norm

    # normalize feature descriptors edges
    n_fd = fd1 / np.sqrt(np.sum(fd1 ** 2))
    n_fd2 = fd2 / np.sqrt(np.sum(fd2 ** 2))

    d_d = distance.euclidean(n_fd, n_fd2)


    return d_h*d_d #originally d_h*d_w + d_h*d_d + d_d*d_w