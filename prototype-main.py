"""

"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('featureextraction/solar/solar_global/')))
sys.path.append(os.path.dirname(os.path.abspath('KeyFrameExtraction/SBD')))
sys.path.append(os.path.dirname(os.path.abspath('nearestneighbor/main.py')))
from nearestneighbor.main import nns
from featureextraction.fe_main import extract_features_global
import numpy as np
from PIL import Image
import argparse
from KeyFrameExtraction.main import save_keyframes, keyframe_extraction
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_video',
        type=str,
        help='the path to the input video'
    )
    parser.add_argument(
        'input_image',
        type=str,
        help='the path to the input image'
    )
    args = parser.parse_args()

    # expand the paths
    args.input_video = os.path.expanduser(args.input_video)
    args.input_image = os.path.expanduser(args.input_image)

    return args


def main():
    args = parse_args()

    start_time = time.time()  # to be sure we also save the ultimate start time

    input_image = Image.open(args.input_image)
    input_image = input_image.convert('RGB')
    input_image = np.array(input_image)
    if input_image.shape[2] == 4:  # if there is an alpha channel: disregard it
        input_image = input_image[...,:3]

    kfe_start_time = time.time()
    keyframes_data, indices, fps = keyframe_extraction(args.input_video, 'VSUMM_combi', True, True)
    indices = np.array(indices)
    kfe_end_time = time.time()

    size = 576  # as determined by our timing/performance measurements
    fe_start_time = time.time()
    search_features = extract_features_global(np.array([input_image]), size)
    frame_features = extract_features_global(keyframes_data, size)
    fe_time_end = time.time()
    fe_time = fe_time_end - fe_start_time

    print('>>> Performing Nearest Neighbour Search')
    nn_start_time = time.time()
    idx, dist, _ = nns(frame_features, search_features)
    nn_end_time = time.time()
    nn_time = nn_end_time - nn_start_time
    print('>>> Done performing Nearest Neighbour Search')

    frame_idx = indices[idx[0]]  # The frames in which the object is most likely present
    dist = dist[0]
    timestamps = frame_idx / fps
    end_time = time.time()
    total_time = end_time - start_time

    print('-' * 80)
    print(f'- Results obtained in {total_time} seconds')
    print('-' * 80)
    print('-')
    print(f'- The following timestamps were returned:')
    print(*[f'-\t {t} (frame {f}, dist {d})' for f, t, d in zip(frame_idx, timestamps, dist)], sep='\n')
    print('')
    # print(f'The linear calculated distances are: {lin_dist}')

    kfe_time = kfe_end_time - kfe_start_time
    print()
    print('Timings:')
    print(f'\tKFE: {kfe_time}')
    print(f'\tFE : {fe_time}')
    print(f'\tNN : {nn_time}')

    save_keyframes(indices, keyframes_data)  # save the individual frames to disk for manual checking


if __name__ == '__main__':
    main()
