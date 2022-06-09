import cv2
import argparse
import numpy as np

import cv2
import numpy as np
import os
import sys
import math
import copy
import time

print("Path:")
print(sys.argv[1])

cap = cv2.VideoCapture(sys.argv[1])
success, old_frame = cap.read()
old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
width = len(old_frame)
height = len(old_frame[0])

print(old_frame.shape)

sum = 0;
for i in range(0,720):
    for j in range(0, 1280):
        sum += old_frame[i][j]

print(sum)
print(np.sum(old_frame))
frame1 = np.array(old_frame,'int')

#frame_diff = np.subtract(frame1,frame2)
#frame_diff = np.abs(frame_diff)
# for x in range(0, height):
#     for y in range(0, width):

#framesubstract = np.abs(np.sum(frame1) - np.sum(frame2))
#frame_diff_rat = frame_sum/256/len(frame1)/len(frame1[0])
