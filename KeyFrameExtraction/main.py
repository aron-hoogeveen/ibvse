import sys
import os
import time
import numpy as np

from Gen_SBD import *

def save_keyframes(keyframe_indices, frames_data):
    print("Extracting keyframes")
    savepath = os.path.expanduser("~/bin/keyframes")
    try:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
    except OSError:
        print("Error can't make directory")
    for i in range (0, len(keyframe_indices)):
        # frame_rgb = cv2.cvtColor(frames_data[i], cv2.COLOR_BGR2RGB)
        frame_rgb = frames_data[i]
        #print("Extracting frame " + str(keyframe_indices[i]))
        name = savepath + '/' + str(keyframe_indices[i]) + '.jpg'
        cv2.imwrite(name, frame_rgb)

def keyframe_extraction(video_path, method, performSBD, presample):
    """
    Performs the extraction of keyframes of an input video and returns
    :param video_path: the path to the input video
    :param method: the method of keyframe extraction after performing shot detection
    :param performSBD: boolean, perform shot boundary detection or not
    :param presample: boolean, presample the input video for speed gain (default  = 10 fps)
    :return: Indices of keyframes, corresponding rgb-data and video_fps
    """


    print("Opening video: " + video_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    method_time = time.time()

    print('>>> Performing Shot Based Detection')
    keyframes_indices = SBD(cap, method, performSBD, presample, video_fps)
    # keyframes_idx = np.array([])
    # keyframes_indices = int(np.array(keyframes_indices))
    # for i in range(0, len(keyframes_indices)):
    #     keyframes_idx = np.concatenate((keyframes_idx,keyframes_indices[i]), axis=None)

    # # Convert [[x,x,x], [x,x,x,x], [x,x,x,x], ... ] to one axis array
    keyframes_idx = []
    for i in range(0, len(keyframes_indices)):
        for j in range(0, len(keyframes_indices[i])):
            keyframes_idx.append(keyframes_indices[i][j])

    print('\033[93m' + f'Time to select indices with method using descriptors: {time.time() - method_time}'+ '\033[0m')
    keyframes_data = keyframes_indices_to_array(keyframes_idx, video_path, video_fps, frame_count)

    print_statistics(frame_count, video_fps, keyframes_idx)

    return keyframes_data, keyframes_idx, video_fps



def KE_uniform_sampling(video_path, rate, CR):
    # Generates indices and gets the framedata from the video using CAP_PROP_POS
    # CAP_PROP_POS is only superior to regular reading ad discarding for uniformly sampling if every 21th frame or more
    # (=1.4fps for a 30fps video) is taken

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    method_time = time.time()

    choose_rate = True # set to False to use Compression Ratio instead
    if choose_rate:
        skip_num = video_fps / rate
    else:
        skip_num = 1/(1-CR)

    keyframes_idx = [i for i in range(frame_count-1) if int(i % skip_num) == 0]
    print('\033[93m' + f'Time to select indices with uniform sampling: {time.time()-method_time}'+ '\033[0m')
    keyframes_data = keyframes_indices_to_array(keyframes_idx, video_path, video_fps, frame_count)

    print_statistics(frame_count, video_fps, keyframes_idx)

    return keyframes_data, keyframes_idx, video_fps

def fast_uniform_sampling(video_path, rate, CR):
    # Reads video and stores frames uniformly and discards others
    # Reading and discarding is better for faster sampling (more than 1.4fps for a 30fps video) than setting the frame
    # location to be picked using CAP_PROP_POS
    print('>>> Uniformly sampling frames from video directly into array w/o selected keyframe indices')
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    choose_rate = True # set to False to use Compression Ratio instead
    if choose_rate:
        skip_num = video_fps / rate
    else:
        skip_num = 1/(1-CR)

    keyframes_data = []
    keyframes_idx = []

    count = 1
    while True:
        success = cap.grab()
        if not success:
            break
        if int(count%skip_num) == 0:
            ret, frame = cap.retrieve()
            keyframes_data.append(frame)
            keyframes_idx.append(count)
        count += 1
    print_statistics(frame_count, video_fps, keyframes_idx)

    return keyframes_data, keyframes_idx, video_fps


def keyframes_indices_to_array(indices, path, video_fps,frame_count):
    #decodes selected videoframes into array
    print('>>> Decoding selected frames from video into array')
    frames = []
    cap = cv2.VideoCapture(path)
    summary_ratio = 0.65/30 # about 0.22 of the amount of frames as threshold
    # for CR higher than 97.8%, CAP_PROP_POS_FRAMES is faster
    if (len(indices) < frame_count*summary_ratio):
        for i in range(0, len(indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, indices[i])
            _, frame = cap.read()
            frames.append(frame)
    else:
        count = 0
        idx = 0
        while True:
            success = cap.grab()
            if not success:
                break
            if (count == indices[idx]):
                ret, frame = cap.retrieve()
                frames.append(frame)
                idx += 1
            if idx > len(indices)-1:
                break
            count += 1
    return frames

def print_statistics(frame_count, video_fps, keyframes_idx):
    print(f'>>> There are {len(keyframes_idx)} keyframes extracted (indices: {keyframes_idx}).')
    print("STATISTICS:")
    duration = frame_count/video_fps
    compression = 1- len(keyframes_idx)/frame_count
    frames_per_second = len(keyframes_idx)/duration
    print("Duration of input video: " + str(duration))
    print("Avg. keyframes per second: " + str(frames_per_second))
    print("Compression Ratio: " + str(compression))


if __name__ == '__main__':
    print("Path:")
    print(sys.argv[1])
    kfe_time  = time.time()
    # methods: crudehistogram, firstmiddlelast, firstlast, firstonly, histogramblockclustering, shotdependentsampling
    #           VSUMM, VSUMM_combi, colormoments
    KE_method = "VSUMM_combi"
    performSBD = True
    presample = False
    keyframes_data, keyframe_indices, video_fps = keyframe_extraction(sys.argv[1], KE_method, performSBD, presample)
    #keyframes_data, keyframe_indices, video_fps = KE_uniform_sampling(sys.argv[1], 0.5, 0.85)
    #keyframes_data, keyframe_indices, video_fps = fast_uniform_sampling(sys.argv[1], 5, 0.85)
    print('\033[92m' + f' Total KeyframeExtraction time: {time.time()-kfe_time}'+ '\033[0m')
    #save_keyframes(keyframe_indices, keyframes_data)