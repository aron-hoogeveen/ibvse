import cv2
import numpy as np
import os
import sys
import math
import copy
import time
import matplotlib.pyplot as plt
import operator

print("Path:")
print(sys.argv[1])

threshold = 0.001
factor = 2
cap = cv2.VideoCapture(sys.argv[1])

success, old_frame = cap.read()
old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

width = len(old_frame)
height = len(old_frame[0])
quotient = 256*threshold*width*height #rows*cols of frame

framediff = []
framediff2 = []
success, frame = cap.read()
while success:

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_old = np.array(old_frame,'float')
    frame_new = np.array(frame_gray,'float')



    framediff.append(np.sum(np.abs(np.subtract(frame_new, frame_old))))

    frame_old = np.array(old_frame)/quotient
    frame_new = np.array(frame_gray)/quotient

    framediff2.append(np.sum(np.abs(np.subtract(frame_new, frame_old))))

    old_frame = frame_gray
    success, frame = cap.read()

mean_diff = sum(framediff)/len(framediff)
threshold_diff = factor * mean_diff

diffdiff = []
diffdiff.append(0)
for i in range(len(framediff)-1):
    diff_of_diff = (framediff[i]-framediff[i-1])
    diffdiff.append(diff_of_diff)
meane = []
meane.append(0)
pt = []
pt.append(threshold_diff)
pt.append(threshold_diff)

conv = np.ones(5)
diffconv = np.convolve(framediff,conv)/2

convthresh = []
for j in range(len(framediff)):
    if threshold_diff > diffconv[j]:
        convthresh.append(threshold_diff)
    else:
        convthresh.append(diffconv[j])

meane.append(len(diffdiff))

for i in range(len(framediff)):
    if framediff[i] >= convthresh[i]:
        print(i)


plt.plot(range(len(framediff)), framediff, label='Frame difference')#, range(len(diffconv)),diffconv, meane, pt)
plt.plot(range(len(diffconv)),diffconv, label='CFAR threshold')
plt.plot(meane, pt, label='average threshold')
plt.title("Plot of frame differences and threshold using modified method")
plt.xlabel("N-th frame")
plt.ylabel("Magnitude of difference")

# plt.axis([150, 220, 0, 40000000])
plt.legend(loc="upper left")
plt.show()
