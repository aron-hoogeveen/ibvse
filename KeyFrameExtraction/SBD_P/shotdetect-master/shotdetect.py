#!/usr/bin/python

import cv2
import numpy as np
import os
import sys
import math
import copy
import time
start_time = time.time()
'''
  A simple yet effective python implementation for video shot detection of abrupt transition
  based on python OpenCV
'''

__hist_size__ = 128             # how many bins for each R,G,B histogram
__min_duration__ = 10           # if a shot has length less than this, merge it with others
__absolute_threshold__ = 100000   # any transition must be no less than this threshold

class shotDetector(object):
    def __init__(self, video_path = None, min_duration=__min_duration__, output_dir=None):
        self.video_path = video_path
        self.min_duration = min_duration
        self.output_dir = output_dir
        self.factor = 6
        
    def run(self, video_path=None):
        if video_path is not None:
            self.video_path = video_path    
        assert (self.video_path is not None), "you should must the video path!"

        self.scores = []
        self.frames = []
        hists = []
        frame_diffs = []
        cap = cv2.VideoCapture(self.video_path)
        self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        success, old_frame = cap.read()
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        frame_idx = 1
        if success:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # self.frames.append(frame)

                frame1 = np.array(old_frame,'int')
                frame2 = np.array(frame_gray,'int')

                frame_diff = np.subtract(frame1,frame2)
                frame_diff = np.abs(frame_diff)
                frame_sum = np.sum(frame_diff)

                frame_diff_rat = frame_sum/256/len(frame1)/len(frame1[0])
                if frame_diff_rat > 0.1:
                    print(frame_idx)
                frame_diffs.append(frame_diff_rat)
                frame_idx += 1
                old_frame = frame_gray

            return self.frames

                      
    def pick_frame(self):
        average_frame_div = sum(self.scores)/len(self.scores)
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

        idx_new = list(map(lambda x : x + 1.0, idx_new))
        return idx_new

                
if __name__ == "__main__":

    shots_array = []

    video_path = sys.argv[1]
    detector = shotDetector(video_path)
    frames = detector.run()

    # shot_boundary = detector.pick_frame()
    #
    #
    # for i in range(len(shot_boundary)-1):
    #     shots_array.append(frames[int(shot_boundary[i]):int(shot_boundary[i+1])])
    #
    # # i.e. shots_array[n] is the (n+1)th number shot
    # print(shot_boundary)
    print("--- %s seconds ---" % (time.time() - start_time))




