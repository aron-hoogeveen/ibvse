import cv2
import numpy as np
import os
import sys
import math
import copy
import time

print("Path:")
print(sys.argv[1])

threshold = 0.001

shot_frames = []
cap = cv2.VideoCapture(sys.argv[1])
success, old_frame = cap.read()
old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
shot_frames.append(old_frame)
width = len(old_frame)
height = len(old_frame[0])
quotient = 256*threshold*width*height #rows*cols of frame
print(quotient)
sum_old = 0;


if success:
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shot_frames.append(frame_gray)
        # self.frames.append(frame)
        frame1 = np.array(old_frame,'int')
        frame2 = np.array(frame_gray,'int')
        #frame_diff = np.subtract(frame1,frame2)
        #frame_diff = np.abs(frame_diff)
        # for x in range(0, height):
        #     for y in range(0, width):

        framesubstract = np.abs(np.sum(frame1) - np.sum(frame2))
        #frame_diff_rat = frame_sum/256/len(frame1)/len(frame1[0])

        print(framesubstract)

        if (framesubstract > quotient):
            print("shot!")
        #frame_diffs.append(frame_diff_rat)

        old_frame = frame_gray