banner = """
This script creates the labeling file (csv) that will be used by the DataSet
class.

Author: Aron Hoogeveen <aron.hoogeveen@gmail.com>
"""

import pandas as pd
import argparse
from glob import glob
import os


def parse_args():
    parser = argparse.ArgumentParser(description=banner)
    parser.add_argument(
        'dir',
        type=str,
        metavar='INPUT_DIR',
        help='the directory that contains all directories with extracted frames'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    args.dir = os.path.normpath(os.path.expanduser(args.dir))
    if not os.path.isdir(args.dir):
        print(f'!!! Directory "{args.dir}" does not exist')
        exit(1)

    for path in os.listdir(args.dir):
        full_path = args.dir + '/' + path
        if os.path.isdir(full_path):
            files = os.listdir(full_path)
            if 'labels-ranges.txt' not in files:
                continue  # this is not a directory with manual labeled ranges

            labels = {}
            try:
                frame_names = list(map(lambda x: os.path.basename(x),
                                       glob(full_path + '/frame_*.jpg')))
                list.sort(frame_names)  # glob does not guarantee ordering
                for name in frame_names:
                    labels[name] = 0

                with open(full_path + '/labels-ranges.txt') as fin:
                    for line in fin:
                        my_range = line.split('-')
                        if len(my_range) == 1:
                            labels[frame_names[int(my_range[0])]] = 1
                        else:
                            for i in range(int(my_range[0]), int(my_range[1])+1):
                                labels[frame_names[i]] = 1
            except ValueError as ve:
                print('!!! file "labels-ranges.txt" has an unsupported layout.')
                print(f'Exception: {ve}')
                exit(1)

            # Write the dict to a csv file

            s = pd.Series(labels, name='label')
            s.index.name = 'frame'
            s = s.reset_index()
            df = pd.DataFrame(s)
            df.to_csv(full_path + '/labels.csv')


if __name__ == '__main__':
    main()
