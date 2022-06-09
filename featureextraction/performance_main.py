import argparse
import os
import sys
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath('.')))
sys.path.append(os.path.dirname(os.path.abspath('./solar/solar_global')))
from featureextraction.fe_main import extract_features_global, extract_features


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dirs',
        type=str,
        nargs='+',
        help='the directories containing the extracted frames, the labels.csv and search-images.csv'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='performance.csv',
        help='path for the output csv (comma separated values) file'
    )
    parser.add_argument(
        '--img_dir',
        type=str,
        default='.',
        help='absolute or relative path to directory containing search images'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=720,
        help='the size at which to process the images'
    )
    return parser.parse_args()


def measure_perf(dirs: [str], img_dir: str, size: int = 256):
    """Measure the performance of our extraction system.

    Note: the search-images.csv file should contain full relative or absolute paths to the files

    Arguments:
        dirs:       a list of paths-like objects to the directories containing the frames and labels
        img_dir:    relative or absolute path to the directory containing the search images
        size:       the size to resize the image to. After resizing the shortest side of the images
                    will equal `size`

    Returns:
        data:   a dict with the following keys: `name`, `search_image`, `num_of_frames`, `ap`,
                `recall`, `k` and `size`
    """
    # make sure all directories exist and have the appropriate files
    for i, d in enumerate(dirs):
        d = os.path.expanduser(d)
        dirs[i] = d
        if not os.path.isdir(d):
            raise ValueError(f'The path "{d}" does not exist')
        if not os.path.isfile(d + '/labels.csv'):
            raise ValueError(f'Directory "{d}" does not contain the "labels.csv" file')
        if not os.path.isfile(d + '/search-images.csv'):
            raise ValueError(f'Directory "{d}" does not contain the "search-images.csv" file')
        # base_name = os.path.basename(d)
        # if not os.path.isfile(d + '/' + base_name + '.jpg'):
        #     raise ValueError(f'Directory "{d}" does not contain the "{base_name}.jpg" file (check'
        #                      f' the extension for jpg)')

    img_dir = os.path.expanduser(img_dir)
    if not os.path.isdir(img_dir):
        raise ValueError(f'The path "{img_dir}" does not exist')

    print('------------------------------ Settings ----------------------------------------')
    print(f'- Size: {size}.')
    print('- Directories:')
    for d in dirs:
        print(f'-     * "{d}"')

    data = {
        'name': [],
        'search_image': [],
        'num_of_frames': [],
        'ap': [],
        'recall': [],
        'k': [],
        'size': []
    }

    for d in dirs:
        # read in the search images
        df = pd.read_csv(d + '/search-images.csv')
        search_images = []
        for img_name in df['name']:
            img_name = os.path.expanduser(img_dir + '/' + img_name)
            img = Image.open(img_name)
            img = img.convert('RGB')
            img = np.array(img)
            if img.shape[2] == 4:
                img = img[..., :3]
            search_images.append(img)
        search_features = extract_features_global(search_images, size)

        # extract the features and labels of the frames
        labels = d + '/labels.csv'
        frame_features, frame_labels, frame_names = extract_features(d, labels, size)
        k = int(sum(frame_labels))  # the number of theoretical hits

        for idx, search_f in enumerate(search_features):
            # calculate the distances and order the frames according to the distances
            dist = frame_features - search_f
            dist = np.linalg.norm(dist, axis=1)
            res = np.argsort(dist)

            # calculate the average precision
            ap = 0
            hits = 0
            for i in range(0, k):
                if frame_labels[res[i]] == 1:
                    hits = hits + 1
                    ap = ap + hits / (i+1)
            ap = ap / k

            data['name'].append(os.path.basename(d))
            data['search_image'].append(df['name'][idx])
            data['num_of_frames'].append(len(frame_names))
            data['ap'].append(ap)
            data['recall'].append(hits / k)
            data['k'].append(k)
            data['size'].append(size)

    return data


def main():
    """
    Let the size of the images be a parameter.

    For each of the directories:
        1. Create a DatasetUtils.DataLoader for each of the input directories.
        2. For each of the directories:
            a. extract the features of the search image
            b. extract the features and labels of the frame images
            c. perform linear comparison using euclidean distance
            d. compute the average precision

    Returns:
        for each of the directories:
            - the name of the video
            - the number of frames
            - the average precision
            - the number of theoretical positive matches
            - the size
    """
    args = parse_args()
    # we do not want to overwrite earlier measurements. TODO implement merging of measurements
    out = os.path.expanduser(args.out)
    if os.path.isfile(out):
        print(f'ERROR >>> The file "{out}" already exists. Pls specify another name.')
        exit(1)
    if not (os.path.dirname(out) == '' or os.path.isdir(os.path.dirname(out))):
        print(f'Error >>> The directory for the output file "{os.path.dirname(out)}" does not exist.')
        exit(1)

    data = measure_perf(args.dirs, args.img_dir, args.size)
    df = pd.DataFrame(data)
    df.to_csv(out)


if __name__ == '__main__':
    main()
