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

fileName = sys.argv[1]
# folderName = sys.argv[2]

skipFrames = 1


ColorMoments = namedtuple('ColorMoments', ['mean', 'stdDeviation', 'skewness'])



def getColorMoments(histogram, totalPixels):
    sum = 0
    for i in range(len(histogram)):
        sum += i*histogram[i]
    test = np.array(histogram)
    test = np.sum(test)

    mean = float (sum / totalPixels)
    print(mean)


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


def save_keyframes(frame_indices):
    global skipFrames
    print ("Saving frame indices")
    
    out_file=open(folderName+"frame_indices_"+str(skipFrames)+".txt",'w')
    for idx in frame_indices:
        out_file.write(str(idx*skipFrames)+'\n')
    print ("Saved indices")

def main():
    videoCap = cv2.VideoCapture(fileName)
    fps = videoCap.get(cv2.CAP_PROP_FPS)
    print ("Frames per second: ", fps)
    euclideanDistance = []

    t0 = time.perf_counter()

    i = 0
    success, image = videoCap.read()
    height = len(image)
    width = len(image[0])
    totalPixels = width * height
    print(totalPixels)
    while success:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([grayImage],[0],None,[256],[0,256])


        colorMoments = getColorMoments(histogram, totalPixels)

        if i==0:
            euclideanDistance.append(0)
        else:
            euclideanDistance.append( getEuclideanDistance(colorMoments, prevColorMoments) )

        prevColorMoments = colorMoments

        i += 1
        success, image = videoCap.read()
        # Uncomment this for breaking early i.e. 100 frames
        # if i==50:
        #     break
    factor = 5
    meanEuclideanDistance = sum(euclideanDistance[1:]) / float(len(euclideanDistance)-1)
    thresholdEuclideanDistance = max(factor * meanEuclideanDistance, euclideanDistance)

    keyFramesIndices = []
    for i in range(len(euclideanDistance)):
        if euclideanDistance[i] > thresholdEuclideanDistance:
            keyFramesIndices.append(i)

    # perc = 0.05
    # keyFramesIndices = sorted(np.argsort(euclideanDistance)[::-1][:int(i*perc)])
    print(keyFramesIndices)
    #save_keyframes(keyFramesIndices)
    
    print ('Time taken to run =', time.perf_counter() - t0, 'seconds' )

if __name__ == '__main__':
	    main()