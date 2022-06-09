import numpy as np
import cv2
import math

def random_summary(compression, n_frames):
    print("Generating random summary")
    summary_selections = np.random.random((n_frames, 1)) * 100
    val_percentile = np.percentile(summary_selections, compression)
    sumsel = np.zeros([n_frames, 1])
    for i in range(0, len(summary_selections)):
        if summary_selections[i] >= val_percentile:
            sumsel[i] = 1
    return sumsel

def uniform_sampling_summary(path, rate):
    print("Generating uniformly sampled keyframes")
    cap = cv2.VideoCapture(path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if rate > video_fps:
        raise ValueError("Sampling rate exceeds video fps")

    keyframe_amount = math.ceil(frame_count*rate/video_fps)

    count = 0
    ret, frame = cap.read()

    #create empty arrays
    indices = np.zeros(keyframe_amount)
    w = frame.shape[0]
    h = frame.shape[1]
    d = frame.shape[2]
    keyframes = np.empty((keyframe_amount, w, h, d), dtype=frame.dtype)
    indices = np.zeros(keyframe_amount)

    while ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keyframes[count] = img
        indices[count] = cap.get(cv2.CAP_PROP_POS_FRAMES)
        count += 1
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * count / rate)
        ret, frame = cap.read()
    print(keyframes.shape)
    return indices, keyframes, video_fps

