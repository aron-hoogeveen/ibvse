#from func_SBD import *
from KeyFrameExtraction.otherscripts.func_SBD2 import *


def noSBD(frame_loc, cap):
    ret, frame = cap.read()
    frames = []
    while ret:
        frame_loc += 1
        frames.append(frame)
        ret, frame = cap.read()
        if not ret:
            break
    print("Generated one shot for entire video")
    return frames, frame_loc

def SBDmain2(frame_loc, cap):

    return data, frame_loc, cap

def SBDmain(frame_loc, cap):
    shots_array = []
    detector = shotDetector(cap)
    video_fps = 30
    frames = detector.run()

    shot_boundary = detector.pick_frame()

    for i in range(len(shot_boundary)-1):
        shots_array.append(frames[int(shot_boundary[i]):int(shot_boundary[i+1])])

    # i.e. shots_array[n] is the (n+1)th number shot
    #print(shot_boundary)
    #return shot_boundary, shots_array
    frame_loc += shot_boundary[1]

    return shots_array[0] ,frame_loc, cap

# def SBD(video_path):
#
#     __hist_size__ = 128  # how many bins for each R,G,B histogram
#     __min_duration__ = 10  # if a shot has length less than this, merge it with others
#     __absolute_threshold__ = 100000  # any transition must be no less than this threshold
#
#
#     shots = []
#     min_duration = __min_duration__
#     detector = shotDetector(video_path)
#     scores, hists, frames = detector.run()
#
#     # detector.pick_frame(sys.argv[2], sys.argv[3])
#
#     print("hoi")
#     print(len(detector.scores))
#     print("frame_count is {0}".format(detector.frame_count))
#
#     average_frame_div = sum(detector.scores) / len(detector.scores)
#     print("average divergence = {0}".format(average_frame_div))
#
#     special_frame = [sp_frame for sp_frame in detector.scores if sp_frame > average_frame_div * detector.factor]
#
#     print("special frames have {0}".format(len(special_frame)))
#
#     print("max diff:", max(scores), "min diff:", min(scores))
#     # compute automatic threshold
#     mean_score = np.mean(scores)
#     std_score = np.std(scores)
#     threshold = max(__absolute_threshold__, mean_score + 3 * std_score)
#     print("thresh")
#     print(threshold)
#     # decide shot boundaries
#     prev_i = 0
#     prev_score = scores[0]
#     for i, score in enumerate(scores[1:]):
#         if score >= threshold and abs(score - prev_score) >= threshold / 2.0:
#             shots.append((prev_i, i + 2))
#             prev_i = i + 2
#         prev_score = score
#     video_length = len(hists)
#     shots.append((prev_i, video_length))
#     assert video_length >= min_duration, "duration error"
#
#     #merge_short_shots(shots, min_duration)
#
#     # save key frames
#     out_array = list(zip(*shots))
#     out_array = out_array[0]
#     return out_array