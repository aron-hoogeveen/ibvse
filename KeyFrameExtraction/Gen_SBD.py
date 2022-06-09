#!/usr/bin/python

import cv2
import numpy as np
import os
import sys
import math
import copy
import time
from basicmethods import *
from histogramblockclustering import *
from VSUMM_KE import *
from VSUMM_combi import *

start_time = time.time()


__hist_size__ = 128             # how many bins for each R,G,B histogram
__min_duration__ = 10           # if a shot has length less than this, merge it with others

class shotDetector(object):
    def __init__(self, min_duration=__min_duration__, output_dir=None):
        self.min_duration = min_duration
        self.output_dir = output_dir
        self.factor = 5
        self.n_frames = 0
        self.method_descriptors = []
        
    def run(self, method, cap, presample, skip_num):

        self.scores = []
        hists = []

        self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        if presample:
            count = 0
            frame_count = 0
            while True:
                success = cap.grab()
                if not success:
                    break
                if int(count % skip_num) == 0:
                    ret, frame = cap.retrieve()
                    chists = [cv2.calcHist([frame], [c], None, [__hist_size__], [0, 256]) \
                              for c in range(3)]
                    chists = np.array([chist for chist in chists])
                    hists.append(chists.flatten())
                    self.method_descriptors.append(createDescriptor(method, frame))
                    self.n_frames += 1
                    frame_count += 1
                count += 1
        else:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                chists = [cv2.calcHist([frame], [c], None, [__hist_size__], [0,256]) \
                              for c in range(3)]
                chists = np.array([chist for chist in chists])
                hists.append(chists.flatten())
                self.method_descriptors.append(createDescriptor(method, frame))
                self.n_frames += 1

        # compute hist  distances
        self.scores = [np.ndarray.sum(abs(pair[0] - pair[1])) for pair in zip(hists[1:], hists[:-1])]

    def pick_frame(self, method):
        average_frame_div = sum(self.scores)/len(self.scores)
        self.frame_index = []
        for idx in range(len(self.scores)):
            if self.scores[idx] > self.factor * average_frame_div:
                self.frame_index.append(idx)
                

        tmp_idx = copy.copy(self.frame_index)
        for i in range(0, len(self.frame_index) - 1):
            if self.frame_index[i + 1] - self.frame_index[i] < __min_duration__:
                del tmp_idx[tmp_idx.index(self.frame_index[i])]
        self.frame_index = tmp_idx

        # the real index start from 1 but time 0 and end add to it
        idx_new = copy.copy(self.frame_index)
        idx_new.insert(0, -1)
        if self.n_frames - 1 - idx_new[-1] < __min_duration__:
            del idx_new[-1]
        #print(self.n_frames)
        idx_new.append(self.n_frames - 1)

        idx_new = list(map(lambda x : x + 1, idx_new))
        return idx_new, self.method_descriptors


def createDescriptor(method, frame):

    if method == "crudehistogram":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        descriptor = hist
    if method == "histogramblockclustering":
        frame_rgb = frame

        # dividing a frame into 3*3 i.e 9 blocks
        height, width, channels = frame_rgb.shape

        if height % 3 == 0:
            h_chunk = int(height / 3)
        else:
            h_chunk = int(height / 3) + 1
        if width % 3 == 0:
            w_chunk = int(width / 3)
        else:
            w_chunk = int(width / 3) + 1
        h = 0
        w = 0
        feature_vector = []
        for a in range(1, 4):
            h_window = h_chunk * a
            for b in range(1, 4):
                frame = frame_rgb[h: h_window, w: w_chunk * b, :]
                hist = cv2.calcHist(frame, [0, 1, 2], None, [6, 6, 6],
                                    [0, 256, 0, 256, 0, 256])  # finding histograms for each block
                hist1 = hist.flatten()  # flatten the hist to one-dimensinal vector
                feature_vector += list(hist1)
                w = w_chunk * b

            h = h_chunk * a
            w = 0

        descriptor = feature_vector # M = 1944 one dimensional feature vector for frame

    elif method == "VSUMM" or method == "VSUMM_combi":
        channels=['b','g','r']
        num_bins = 16
        feature_value=[cv2.calcHist([frame],[i],None,[num_bins],[0,256]) for i,col in enumerate(channels)]
        descriptor = np.asarray(feature_value).flatten()

    elif  method == "firstmiddlelast" or method == "firstonly" or method == "firstlast" or method == "uniformsampling" or method == "shotdependentsampling":
        descriptor = 1

    return descriptor

def KFE(presample, skip_num, method, method_descriptors, shot_frame_number):
    if method == "crudehistogram":
        keyframe_indices = histogram_summary(method_descriptors, shot_frame_number)
    elif method == "firstmiddlelast":
        keyframe_indices = first_middle_last(method_descriptors, shot_frame_number)
    elif method == "firstlast":
        keyframe_indices = first_last(method_descriptors, shot_frame_number)
    elif method == "firstonly":
        keyframe_indices = first_only(method_descriptors, shot_frame_number)
    elif method == "histogramblockclustering":
        keyframe_indices = blockclustering(method_descriptors, shot_frame_number)
    elif method == "shotdependentsampling":
        keyframe_indices = shotdependent_sampling(method_descriptors, shot_frame_number)
    elif method == "VSUMM":
        keyframe_indices = VSUMM(method_descriptors, shot_frame_number, presample, skip_num)
    elif method == "VSUMM_combi":
        keyframe_indices = VSUMM_combi(method_descriptors, shot_frame_number, presample, skip_num)

    if presample:
        keyframe_indices = [round(element * skip_num) for element in keyframe_indices]
    return keyframe_indices


def SBD(cap, method, performSBD, presample, video_fps):
    time_SBD = time.time()

    sampling_rate = 10 # presampling to decrease computation time for reading frames
    skip_num = video_fps/sampling_rate # retrieve every n-th frame

    if not performSBD:
        print("No SBD performed! Viewing video as one entire shot/segment")
        frame_count = 0
        method_descriptors = []

        if presample:
            count = 0
            while True:
                success = cap.grab()
                if not success:
                    break
                if int(count % skip_num) == 0:
                    ret, frame = cap.retrieve()
                    method_descriptors.append(createDescriptor(method, frame))
                    frame_count += 1
                count += 1
        else:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                method_descriptors.append(createDescriptor(method, frame))
                frame_count += 1


        shot_boundary = []
        shot_boundary.append(0)
        shot_boundary.append(frame_count)
        print(shot_boundary)
        print('\033[94m' + f'Time to read (presampled) video and  generate descriptors for chosen method: {time.time() - time_SBD}' + '\033[0m')

    else: #  perform SBD
        detector = shotDetector()
        detector.run(method, cap, presample, skip_num)

        shot_boundary, method_descriptors = detector.pick_frame(method)
        if presample:
            actual_shot_boundary = [round(element * skip_num) for element in shot_boundary]
            print(f'>>> There are {len(actual_shot_boundary) - 1} shots found at  {actual_shot_boundary}')
        else:
            print(f'>>> There are {len(shot_boundary) - 1} shots found at  {shot_boundary}')
        print('\033[94m' + f'Time to apply SBD and generate descriptors for chosen method: {time.time() - time_SBD}'+ '\033[0m')
        print(">>> Applying " + method + " to shots")

    keyframe_indices = []
    for i in range(0, len(shot_boundary) - 1):
        keyframe_indices.append(KFE(presample, skip_num, method, method_descriptors[int(shot_boundary[i]):int(shot_boundary[i + 1] - 1)], int(shot_boundary[i])))
    return keyframe_indices