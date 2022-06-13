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

cap = cv2.VideoCapture(sys.argv[1])


success, old_frame = cap.read()
old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


width = len(old_frame)
height = len(old_frame[0])
quotient = 256*threshold*width*height #rows*cols of frame

while success:

    success, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame1 = np.array(old_frame)
    frame2 = np.array(frame_gray)

    #frame_diff = np.subtract(frame1,frame2)
    #frame_diff = np.abs(frame_diff)
    # for x in range(0, height):
    #     for y in range(0, width):

    framesubstract = np.abs(np.sum(frame1) - np.sum(frame2))
    #frame_diff_rat = frame_sum/256/len(frame1)/len(frame1[0])

    # print(framesubstract)

    old_frame = frame_gray



    if (framesubstract > quotient):
        print("shot!")
    #frame_diffs.append(frame_diff_rat)