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

#fileName = sys.argv[1]
# folderName = sys.argv[2]

skipFrames = 1


ColorMoments = namedtuple('ColorMoments', ['mean', 'stdDeviation', 'skewness'])



def getColorMoments(histogram, totalPixels):
    sum = 0
    for i in range(len(histogram)):
        sum += i*histogram[i]

    mean = float (sum / totalPixels)
    sumOfSquares = 0
    sumOfCubes = 0
    for pixels in histogram:
        sumOfSquares += math.pow(pixels-mean, 2)
        sumOfCubes += math.pow(pixels-mean, 3)
    variance = float (sumOfSquares / totalPixels)
    stdDeviation = math.sqrt(variance)
    avgSumOfCubes = float (sumOfCubes / totalPixels)
    skewness = float (avgSumOfCubes**(1./3.))
    return ColorMoments(mean, stdDeviation, skewness)


def getEuclideanDistance(currColorMoments, prevColorMoments):
    distance = math.pow(currColorMoments.mean - prevColorMoments.mean, 2) + math.pow(currColorMoments.stdDeviation - prevColorMoments.stdDeviation, 2) + math.pow(currColorMoments.skewness - prevColorMoments.skewness, 2)
    return distance


def colormoments(descriptor, shot_frame_number, totalpixels):

    euclideanDistance = []

    prevColorMoments = getColorMoments(descriptor[0], totalpixels)

    for i in range(1, len(descriptor)):

        colorMoments = getColorMoments(descriptor[i], totalpixels)
        euclideanDistance.append( getEuclideanDistance(colorMoments, prevColorMoments) )
        prevColorMoments = colorMoments



    perc = 2
    factor = 6
    keyFramesIndices = []

    meanEuclideanDistance = sum(euclideanDistance[1:]) / float(len(euclideanDistance) - 1)
    thresholdEuclideanDistance = factor * meanEuclideanDistance

    for i in range(len(euclideanDistance)):
        if euclideanDistance[i] >= thresholdEuclideanDistance:
            keyFramesIndices.append(i)


    if len(keyFramesIndices) > i*perc/100:
        keyFramesIndices = []
        keyFramesIndices = sorted(np.argsort(euclideanDistance)[::-1][:int(max(i*perc/100,1))])

    keyframe_indices = np.array(keyFramesIndices)
    keyframe_indices = [(element + shot_frame_number) for element in keyframe_indices]


    return keyframe_indices