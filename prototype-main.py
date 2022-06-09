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

    # TODO make utility function for reading in images
    input_image = Image.open(args.input_image)
    input_image = input_image.convert('RGB')
    input_image = np.array(input_image)
    if input_image.shape[2] == 4:  # if there is an alpha channel: disregard it
        input_image = input_image[...,:3]

    kfe_start = time.time()
    keyframes_data, indices, fps = keyframe_extraction(args.input_video, 'VSUMM_combi', True, True)
    kfe_end = time.time()

    fe_time_start = time.time()
    size = 480
    search_features = extract_features_global(np.array([input_image]), size)
    frame_features = extract_features_global(keyframes_data, size)
    fe_time_end = time.time()
    fe_time = fe_time_end - fe_time_start
    print('>>> Done performing feature extraction')
    print('>>> Performing Nearest Neighbour Search')
    nn_time_start = time.time()
    idx, dist, _ = nns(frame_features, search_features)
    nn_time_end = time.time()
    nn_time = nn_time_end - nn_time_start
    print('>>> Done performing Nearest Neighbour Search')

    indices = np.array(indices)
    print(f'The indices are:\n{indices[idx]}.\nThe distances are:\n{dist}')

    save_keyframes(indices, keyframes_data)
    kfe_time = kfe_end - kfe_start
    print()
    print('Timings:')
    print(f'\tKFE: {kfe_time}')
    print(f'\tFE : {fe_time}')
    print(f'\tNN : {nn_time}')


if __name__ == '__main__':
    main()
