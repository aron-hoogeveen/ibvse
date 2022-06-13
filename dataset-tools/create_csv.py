#!/usr/bin/env python
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

    errors_occurred = False

    for path in os.listdir(args.dir):
        full_path = args.dir + '/' + path
        if os.path.isdir(full_path):
            files = os.listdir(full_path)
            if 'labels2img.csv' not in files:
                continue  # this is not a directory with manual labeled ranges
            print(f'Trying to process directory "{full_path}"')

            #
            # Loop over each of the search images
            # Create csv file per search image
            #
            df_labels_txt = pd.read_csv(full_path + '/labels2img.csv', index_col=0)  # TODO move fixed filenames to file with constants
            df_labels = df_labels_txt.copy()

            for i in range(len(df_labels_txt)):
                labels_file = full_path + '/' + df_labels_txt.at[i, 'labels']
                if not os.path.isfile(labels_file):
                    print(f'!!! ERROR file {labels_file} does not exist. Skipping directory {full_path}.')
                    errors_occurred = True
                    continue

                labels = {}
                try:
                    frame_names = list(map(lambda x: os.path.basename(x),
                                           glob(full_path + '/frame_*.jpg')))
                    list.sort(frame_names)  # glob does not guarantee ordering
                    for name in frame_names:
                        labels[name] = 0

                    with open(labels_file) as fin:
                        for line in fin:
                            my_range = line.split('-')
                            if len(my_range) == 1:
                                labels[frame_names[int(my_range[0])]] = 1
                            else:
                                for j in range(int(my_range[0]), int(my_range[1])+1):
                                    labels[frame_names[j]] = 1
                except ValueError as ve:
                    print(f'!!! file "{labels_file}" has an error.')
                    print(f'Exception: {ve}')
                    errors_occurred = True
                    continue

                # Write the dict to a csv file

                s = pd.Series(labels, name='label')
                s.index.name = 'frame'
                s = s.reset_index()
                df = pd.DataFrame(s)
                file_name = os.path.splitext(os.path.basename(labels_file))[0] + '.csv'
                df_labels.at[i, 'labels'] = file_name
                df.to_csv(full_path + '/' + file_name)
                df_labels.to_csv(full_path + '/' + 'labels.csv')
                print(f'\tProcessed {os.path.basename(full_path) + "/" + df_labels_txt.at[i, "labels"]}')

    if errors_occurred:
        print('!!! There were errors. Please check the commandline log.')


if __name__ == '__main__':
    main()
