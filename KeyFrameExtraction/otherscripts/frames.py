# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import argparse
import numpy

def extractImages(pathIn, pathOut, fps, idx):
    count = 1 # offset for not doing  double first frame
    vidcap = cv2.VideoCapture(pathIn)

    for i in idx:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, i / fps * 1000)
        success, image = vidcap.read()
        cv2.imwrite(pathOut + "\\frame%d.jpg" % count, image)  # save frame as JPEG file
        count += 1


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images")
    a.add_argument("--pathIdx", help="path to text file with index")
    args = a.parse_args()
    idx = open(args.pathIdx, 'r')
    idx_content = idx.readlines()

    vidcap = cv2.VideoCapture(args.pathIn)
    frame_fps = vidcap.get(cv2.CAP_PROP_FPS)

    idx_value = numpy.zeros(len(idx_content))
    for n in range(0, len(idx_content)):
        idx_value[n] = int(idx_content[n])
    print(idx_value)
    print(args)
    extractImages(args.pathIn, args.pathOut, frame_fps, idx_value)