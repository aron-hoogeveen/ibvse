import cv2
import argparse
import numpy



if __name__=="__main__":

    idx = open("C:\\Users\\Leo\\Desktop\\nad31_cut.txt", 'r')
    idx_content = idx.readlines()
    ref = []
    ref_tmp = []
    for i in range(len(idx_content)):
        ref_tmp = str(idx_content[i])
        ref_tmp = ref_tmp.split("\"")
        if ref_tmp[1] == "CUT":
            print("ja")
        print(ref_tmp[1])
    tt = idx_content.split("\"")
    print(tt)
