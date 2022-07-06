import os
from zipfile import ZipFile
import prototype_main
import pickle
import numpy as np
from mega import Mega

def main():
    """
    Demo script for the Image-based Video Search Engine. It downloads a small dataset and runs the engine.
    The obtained output is compared to the reference output and the deviations are listed
    :return: 2 files with the information on the results of the demo
    """
    np.set_printoptions(linewidth=np.inf)
    base_dir = os.path.abspath('Demo-images-and-videos')
    try:
        # Downloads the dataset stored on MEGA
        print(">>> Downloading dataset for demo")
        assert not os.path.exists(os.path.join(base_dir,'Demo_dataset_ibvse.zip'))
        mega = Mega()
        m = mega.login()
        m.import_public_url('https://mega.nz/file/ejICQS5A#zunX-XdB_-V5e6MgoCcr6frrH44Yds_lPVYXuquQlzw')
        file = m.find('Demo_dataset_ibvse.zip')
        m.download(file, os.path.abspath('Demo-images-and-videos'))

    except PermissionError: # except the permission error since it still downloads the file, but otherwise stops running
        # unzip the downloaded folder
        with ZipFile(os.path.abspath('Demo-images-and-videos/Demo_dataset_ibvse.zip'), 'r') as zipObj:
            zipObj.extractall(os.path.abspath('Demo-images-and-videos'))

    except AssertionError: # If already downloaded then stop downloading
        print('Content is already downloaded. Aborting the download')

    print(">>> Starting demo")
    # Find all videos and images
    videos = [os.path.join(base_dir, 'Battuta1.mp4'),
              os.path.join(base_dir, 'He1.mp4'),
              os.path.join(base_dir, 'Polo1.mp4')]
    images1 = [os.path.join(base_dir, os.path.join('Batutta1', file)) for file in os.listdir(os.path.join(base_dir, 'Batutta1'))]
    images2 = [os.path.join(base_dir, os.path.join('He1', file)) for file in os.listdir(os.path.join(base_dir, 'He1'))]
    images3 = [os.path.join(base_dir, os.path.join('Polo1', file)) for file in os.listdir(os.path.join(base_dir, 'Polo1'))]
    all_images = images1 + images2 + images3

    # using histogramblockclustering to get the same extracted keyframes every run.
    # Normally VSUMM combi is used, but for that the keyframes may differ across runs
    res = prototype_main.main(False, videos, all_images, kfe_method='histogramblockclustering')  # perform the search

    # load the reference results
    with open(os.path.join(base_dir,'demo_ref.pkl'), 'rb') as file:
        ref_res = pickle.load(file)

    mistakes = 0
    # Compare the reference results to the obtained results
    with open(os.path.join(base_dir,'demo_deviations.txt'),'w') as file_deviations, \
            open(os.path.join(base_dir,'demo_results.txt'),'w') as file_results :
        file_results.write("These are the timestamps and distances returned by the engine. The results are listed per video and query combination.\n"
                           "For each of these combinations the obtained result and reference result are listed.\n"
                           "The first list contains the timestamp and the second one the distances\n\n")
        file_deviations.write("These are the deviations that occured from the reference results\n\n")
        for i in range(len(ref_res)):  # for every query video
            for j in range(len(ref_res[i])):  # for every query image
                file_results.write(f'Video {os.path.split(videos[i])[-1]} and Image {os.path.split(all_images[j])[-1]}'
                                   f'\n Result:\t{res[i][j][0]}\t{res[i][j][1]}'
                                   f'\n Reference:\t{ref_res[i][j][0]}\t{ref_res[i][j][1]}\n\n')
                try:
                    assert (ref_res[i][j][0].astype(int) == res[i][j][0].astype(int)).all()  # check of the results
                except AssertionError:
                    mistakes += 1
                    file_deviations.write(f'A deviation from the reference result was found for Video {os.path.split(videos[i])[-1]} '
                          f'and Image {os.path.split(all_images[j])[-1]}:\n{ref_res[i][j][0]} != {res[i][j][0]}\n\n')


    print(f'>>> Results of demo\n'
          f'A total of {mistakes} deviations were found out of the {len(ref_res)*len(ref_res[0])} evaluations\n'
          f'The file "Demo-Images-and-videos/demo_deviations.txt" holds the information on the deviations\n'
          f'The file "Demo-Images-and-videos/demo_results.txt" holds all the information of the results\n\n')



if __name__ == '__main__':
    main()
