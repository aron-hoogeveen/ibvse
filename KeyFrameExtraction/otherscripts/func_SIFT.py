import cv2
import sys
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import namedtuple
import time
import heapq

skipFrames = 1

ColorMoments = namedtuple('ColorMoments', ['mean', 'stdDeviation', 'skewness'])

def getColorMoments(histogram, totalPixels):
    sum = 0
    for pixels in histogram:
        sum += pixels
    mean = float(sum / totalPixels)
    sumOfSquares = 0
    sumOfCubes = 0
    for pixels in histogram:
        sumOfSquares += math.pow(pixels - mean, 2)
        sumOfCubes += math.pow(pixels - mean, 3)
    variance = float(sumOfSquares / totalPixels)
    stdDeviation = math.sqrt(variance)
    avgSumOfCubes = float(sumOfCubes / totalPixels)
    skewness = float(avgSumOfCubes ** (1. / 3.))
    return ColorMoments(mean, stdDeviation, skewness)


def getEuclideanDistance(currColorMoments, prevColorMoments):
    distance = math.pow(currColorMoments.mean - prevColorMoments.mean, 2) + math.pow(
        currColorMoments.stdDeviation - prevColorMoments.stdDeviation, 2) + math.pow(
        currColorMoments.skewness - prevColorMoments.skewness, 2)
    return distance


def save_keyframes(frame_indices):
    global skipFrames
    print("Saving frame indices")

    out_file = open(folderName + "frame_indices_" + str(skipFrames) + ".txt", 'w')
    for idx in frame_indices:
        out_file.write(str(idx * skipFrames) + '\n')
    print("Saved indices")