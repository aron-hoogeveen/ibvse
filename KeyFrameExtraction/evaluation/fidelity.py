# fidelity measure based on semiHausdorff distance
import numpy as np
import cv2
import scipy
import os
import glob
import random
from main import *
from scipy.spatial import distance
from skimage.feature import hog


def fidelity(selection, nf, path, skipSavingAllFrames):
    saveimages(path, skipSavingAllFrames, True, selection, nf) #saves selection in /summary and all frames of video in /data
    [hist_sel, hist_data] = calculateHists(selection)
    [fd_sel, fd_data] = calculateHOGS(selection)

    maxdist = 0
    maxdiff = 0
    for i in range(0, nf):
        [a, maxdiff] = distance_inner(selection, nf, path, i, hist_sel, hist_data, maxdiff, fd_sel, fd_data)
        if (a > maxdist):
            maxdist = a
    return maxdiff - maxdist

def distance_inner(selection, nf, path, framenumber, hist_sel, hist_data, maxdiff, fd_sel, fd_data):
    difference_list = {}
    for i in range(0, len(hist_sel)):
        difference_list[i] = difference(hist_sel[i], hist_data[framenumber], fd_sel[i], fd_data[framenumber])
    print(difference_list)
    maxdiff = check_maxdifference(difference_list,maxdiff)
    distance  = minimalvalue(difference_list)
    return distance, maxdiff

def check_maxdifference(array, maxdiff):
    for i in range(1, len(array)):
        if array[i] > maxdiff:
            maxdiff = array[i]
    return maxdiff

def minimalvalue(array):
    min = array[0]
    for i in range(1, len(array)):
        if array[i] < min:
            min = array[i]
    return min

def difference(hist1, hist2, fd1, fd2):
    d_h = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    # normalize feature descriptors edges
    n_fd = fd1 / np.sqrt(np.sum(fd1 ** 2))
    n_fd2 = fd2 / np.sqrt(np.sum(fd2 ** 2))
    d_d =  distance.euclidean(n_fd, n_fd2)

    return d_h * d_d

def calculateHOGS(sel):
    #hog = cv2.HOGDescriptor()
    fd_sel = {}
    cnt = 0
    print("Creating hogs for summary folder")
    for imagePath in glob.glob('./summary/*.jpg'):
        filename = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #scikit (slow):
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        #fd = hog.compute(image)
        fd_sel[cnt] = fd
        cnt += 1
    fd_data = {}
    cnt = 0
    print("Creating hogs for data folder")
    for imagePath in glob.glob('./data/*.jpg'):
        filename = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # scikit (slow):
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        #fd = hog.compute(image)
        fd_data[cnt] = fd
        print("created hog for frame: " + str(cnt))
        cnt += 1
    print("end of hogs creation")
    return fd_sel, fd_data

def calculateHists(sel):
    hist_sel = {}
    cnt = 0
    print("Creating histograms for summary folder")
    for imagePath in glob.glob('./summary/*.jpg'):
        filename = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        hist_sel[cnt] = hist
        cnt += 1
    hist_data = {}
    cnt = 0
    print("Creating histograms for data folder")
    for imagePath in glob.glob('./data/*.jpg'):
        filename = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        hist_data[cnt] = hist
        cnt += 1
    # for frame in range(0, len(sel)):
    #     if sel[frame]:
    #         image = cv2.imread(imagePath, 1)
    #         cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    #         print(frame)
    #hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    #hist = cv2.normalize(hist, None).flatten()
    #index[filename] = hist
    print("end of histogram creation")
    return hist_sel, hist_data



def histogram():
    images = {}
    index = {}
    for imagePath in glob.glob('./data/*.jpg'):
        filename = imagePath[imagePath.rfind("/") + 1:]
        print("filename: " + str(filename))

        image = cv2.imread(imagePath, 1)
        images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        index[filename] = hist
        OPENCV_METHODS = (
            (cv2.HISTCMP_CORREL),
            (cv2.HISTCMP_CHISQR),
            (cv2.HISTCMP_INTERSECT),
            (cv2.HISTCMP_BHATTACHARYYA))

        for method in OPENCV_METHODS:

            results = {}
            reverse = False

            if method in (cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT):
                reverse = True

        for (k, hist) in index.items():
            d = cv2.compareHist(index[k], hist, cv2.HISTCMP_INTERSECT)
            results[k] = d
            #print(d)

def saveimages(path, skipextractionofvideo, skipsummaryextraction, selection, nf):
    if not skipextractionofvideo:
        cap = cv2.VideoCapture(path)
        try:
            if not os.path.exists('../data'):
                os.makedirs('../data')
        except OSError:
            print("Error cant make directories")

        cframe = 0
        while (True):

            ret, frame = cap.read()
            if not ret:
                break
            name = './data/' + str(cframe) + '.jpg'
            print("creating" + name)
            cv2.imwrite(name, frame)
            cframe += 1


    if not skipsummaryextraction:
        extract_from_indices(selection, nf, path, 'summary')  # save summary to summary