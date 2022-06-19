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
        'input_videos',
        type=str,
        nargs='+',
        help='the path to the input video'
    )
    parser.add_argument(
        'input_images',
        type=str,
        nargs='+',
        help='the path to the input image'
    )
    args = parser.parse_args()

    # expand the paths
    args.input_videos = os.path.expanduser(args.input_videos)
    args.input_images = os.path.expanduser(args.input_images)

    return args


def main(use_args, input_videos: [str], input_images: [str]):
    """
    The Image Based Video Search Engine main body according to the pipeline of:
    Keyframe Extraction => Feature Extraction => Nearest Neighbour Search
    :param use_args: Should parser arguments be used
    :param input_videos: A list of the paths to the input videos
    :param input_images: A list of the paths to the input images
    :return: The timestamp and distance per video and per query image. The structure is a list that contains a list per
    video. This list contains lists with the data for each image corresponding to that video.
    Example: [[[Video1_Image1_data],[Video1_Image2_data]], [[Video2_Image1_data],[Video2_Image2_data]]]
    """
    #
    # Check if the input files exist, otherwise throw error.
    #
    if use_args:
        args = parse_args()
        input_videos = args.input_videos
        input_images = args.input_images

    start_time = time.time()  # to be sure we also save the ultimate start time
    print(input_images)
    # check if all input images exist
    for i_i, input_image in enumerate(input_images):
        input_image = os.path.expanduser(input_image)
        if not os.path.isfile(input_image):
            raise ValueError(f'Input image "{input_image}" is not an existing file.¨')
        input_images[i_i] = input_image

    for i_i, input_video in enumerate(input_videos):
        input_video = os.path.expanduser(input_video)
        if not os.path.isfile(input_video):
            raise ValueError(f'Input video "{input_video}" is not an existing file.¨')
        input_videos[i_i] = input_video

    query_images = []
    for input_image in input_images:
        img = Image.open(input_image)
        img = img.convert('RGB')
        img = np.array(img)
        if img.shape[2] == 4:
            img = img[..., :3]  # disregard the alpha layer if it is present
        query_images.append(img)


    res = []
    size = 576  # as determined by our timing/performance measurements
    query_features = extract_features_global(query_images, size)
    query_features = query_features.numpy()
    query_features = np.ascontiguousarray(query_features, dtype=np.float32)

    for input_video in input_videos:
        kfe_start_time = time.time()
        keyframes_data, indices, fps = keyframe_extraction(input_video, 'VSUMM_combi', True, True)
        indices = np.array(indices)
        kfe_end_time = time.time()

        fe_start_time = time.time()
        frame_features = extract_features_global(keyframes_data, size)
        fe_time_end = time.time()
        fe_time = fe_time_end - fe_start_time

        print('>>> Performing Nearest Neighbour Search')
        nn_start_time = time.time()


        frame_features = frame_features.numpy()
        frame_features = np.ascontiguousarray(frame_features, dtype=np.float32)

        idx, dist, _ = nns(frame_features, query_features)
        nn_end_time = time.time()
        nn_time = nn_end_time - nn_start_time
        print('>>> Done performing Nearest Neighbour Search')

        # frame_idx = indices[idx[0]]  # The frames in which the object is most likely present
        # dist = dist[0]
        output_data = []
        for image_idx, image_dist in zip(idx, dist):
            frame_idx = indices[image_idx]
            dist = image_dist
            timestamps = frame_idx/fps
            output_data.append([timestamps, dist])

        # timestamps = frame_idx/fps
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

        # save_keyframes(indices, keyframes_data)  # save the individual frames to disk for manual checking
        res.append(output_data)
    return res


if __name__ == '__main__':
    main(True)
