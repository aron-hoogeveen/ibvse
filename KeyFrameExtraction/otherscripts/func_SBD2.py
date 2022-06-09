import cv2
import numpy as np
import os
import sys
import math
import copy
import time

'''
  A simple yet effective python implementation for video shot detection of abrupt transition
  based on python OpenCV
'''

__hist_size__ = 128  # how many bins for each R,G,B histogram
__min_duration__ = 10  # if a shot has length less than this, merge it with others
__absolute_threshold__ = 100000  # any transition must be no less than this threshold


class shotDetector(object):
    def __init__(self, cap=None, min_duration=__min_duration__, output_dir=None):
        self.cap = cap
        self.min_duration = min_duration
        self.output_dir = output_dir
        self.factor = 6

    def run(self, cap=None):
        if cap is not None:
            self.cap = cap
        assert (self.cap is not None), "add the cap! no cap"

        self.scores = []
        self.frames = []
        hists = []

        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            self.frames.append(frame)
            chists = [cv2.calcHist([frame], [c], None, [__hist_size__], [0, 256]) \
                      for c in range(3)]
            chists = np.array([chist for chist in chists])
            hists.append(chists.flatten())
        # compute hist  distances
        self.scores = [np.ndarray.sum(abs(pair[0] - pair[1])) for pair in zip(hists[1:], hists[:-1])]
        print("succes")
        return self.frames

    def pick_frame(self):
        average_frame_div = sum(self.scores) / len(self.scores)
        self.frame_index = []
        for idx in range(len(self.scores)):
            if self.scores[idx] > self.factor * average_frame_div:
                self.frame_index.append(idx + 1)

        tmp_idx = copy.copy(self.frame_index)
        for i in range(0, len(self.frame_index) - 1):
            if self.frame_index[i + 1] - self.frame_index[i] < __min_duration__:
                del tmp_idx[tmp_idx.index(self.frame_index[i])]
        self.frame_index = tmp_idx
        print("special frames have {0}".format(len(self.frame_index)))

        # the real index start from 1 but time 0 and end add to it
        idx_new = copy.copy(self.frame_index)
        idx_new.insert(0, -1)
        if len(self.frames) - 1 - idx_new[-1] < __min_duration__:
            del idx_new[-1]
        idx_new.append(len(self.frames) - 1)

        idx_new = list(map(lambda x: x + 1.0, idx_new))
        return idx_new