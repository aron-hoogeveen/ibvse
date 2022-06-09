import numpy as np
import time
import matplotlib.pyplot as plt
import cv2


def histogram_summary(hists, shot_start_number):
    threshold = 0.3

    current_keyframe_hist = hists[0]
    keyframe_indices = [shot_start_number]
    #print(shot_start_number)
    for i in range(1, len(hists)):
        histdiff = cv2.compareHist(current_keyframe_hist, hists[i], cv2.HISTCMP_BHATTACHARYYA)

        if (histdiff) > threshold:

            keyframe_indices.append(i+shot_start_number)
            current_keyframe_hist = hists[i]

    #print(keyframe_indices)

    # summary = np.count_nonzero(sumsel == 1)/frame_count
    # if summary < 0.001:
    #     print("Compression too high: " + str(1-summary))
    #     sumsel = histogram_summary(path, threshold-0.1, frame_count)
    # if summary > 0.1:
    #     print("Compression too low: " + str(1 - summary))
    #     sumsel = histogram_summary(path, threshold + 0.09, frame_count)
    # print("compression: "+ str(1-summary))
    # print("--- %s seconds ---" % (time.time() - start_time))
    return keyframe_indices


def first_middle_last(descriptors, shot_start_number):
    print('Selecting first, middle and last frame from shot')
    middle = int((shot_start_number+len(descriptors)/2))
    indices = [shot_start_number, middle, shot_start_number+len(descriptors)]
    return indices

def first_last(descriptors, shot_start_number):
    print('Selecting first and last frame from shot')
    indices = [shot_start_number, shot_start_number+len(descriptors)]
    return indices

def first_only(descriptors, shot_start_number):
    print('Selecting first frame from shot')
    indices = [shot_start_number]
    return indices

def shotdependent_sampling(descriptors, shot_start_number):
    indices = [shot_start_number]
    return indices