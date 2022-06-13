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
success, frame = cap.read()
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
        print(hist1.shape)
        feature_vector += list(hist1)
        w = w_chunk * b

    h = h_chunk * a
    w = 0

descriptor = feature_vector  # M = 1944 one dimensional feature vector for frame