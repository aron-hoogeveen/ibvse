import numpy as np
from fidelity import *
from main import *
#import matplotlib.pyplot as plt
from math import atan2
from math import degrees
import os
import time


if __name__ == '__main__':
    filename = "results.txt"
    paths = ["C:\\Users\\Robert\\bin\\ewi-tudelft-2.mp4", "C:\\Users\\Robert\\bin\\mailbox-street.mp4",
             "C:\\Users\\Robert\\bin\\multishot.mp4", "C:\\Users\\Robert\\bin\\zheng-he.mp4"]
    paths = ["C:\\Users\\Robert\\bin\\multishot.mp4", "C:\\Users\\Robert\\bin\\zheng-he.mp4"]
    methods = ["uniform_sampling", "uniform_sampling_small", "crudehistogram", "histogramblockclustering", "VSUMM", "VSUMM_combi", "colormoments"]
    performSBD = True
    presample = True
    n = 10

    with open(filename, 'a') as f:
        #f.write(KE_method + "\n")
        for path in paths:
            f.write(path + "\n")
            cap = cv2.VideoCapture(path)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            video_fps = cap.get(cv2.CAP_PROP_FPS)

            print("Computing fidelity")
            [hogs, hists, fdnorm, histnorm] = fidelity_descriptors(path)


            for KE_method in methods:
                f.write(KE_method + "\n")
                # compute results for this video
                presamples = [False, True]
                performSBDs = [False, True]
                for performSBD in performSBDs:
                    for presample in presamples:
                        f.write("\tPerformSBD: " + str(performSBD) + "\n")
                        f.write("\tPresample: " + str(presample) +  "\n")
                        times = 0
                        for i in range(0,n):
                            kfe_time = time.time()
                            if KE_method == "uniform_sampling":
                                keyframes_data, keyframe_indices, video_fps = KE_uniform_sampling(path, 5, 0.85)
                            elif KE_method == "uniform_sampling_small":
                                keyframes_data, keyframe_indices, video_fps = KE_uniform_sampling(path, 0.6, 0.85)
                            else:
                                keyframes_data, keyframe_indices, video_fps = keyframe_extraction(path, KE_method, performSBD, presample)
                            times += time.time()-kfe_time
                        avgtime = times/n
                        print(f'>>> There are {len(keyframe_indices)} keyframes extracted (indices: {keyframe_indices}).')
                        duration = frame_count / video_fps
                        compression = 1 - len(keyframe_indices) / frame_count
                        frames_per_second = len(keyframe_indices) / duration
                        print("Duration of input video: " + str(duration))
                        print("Avg. keyframes per second: " + str(frames_per_second))
                        print("Compression Ratio: " + str(compression))
                        print("Average time: " + str(avgtime))

                        fid = fidelity(keyframe_indices, path, hists, hogs, fdnorm, histnorm)

                        #write results to file
                        print("Writing to file")
                        f.write("Comptime (n=" + str(n) + "): "+ str(round(avgtime, 2))+"\t")
                        f.write("Comptime/dur: " + str(round(avgtime/duration, 3)) + "\t")
                        f.write("KF: " + str(len(keyframe_indices)) + "\t")
                        f.write("CR: " + str(round(100*compression, 4)) + "\t")
                        f.write("Fidelity: " + str(round(fid, 4)) + "\n\n")