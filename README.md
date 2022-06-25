# Image Based Video Search Engine 

The Image Based Video Search Engine was developed for the BSc graduation project of group H. The project conists of a search engine that finds an instance of an object in query image(s) in a set of query videos. 

# Running the system
## Installations 
To run the system, first clone the repository. To install the correct packages, change directory to the folder that contains the enviroment.yml file and install the enviroment by using the following command
```
conda env create
```
Next, download the solar global model that can be found [here](https://imperialcollegelondon.box.com/shared/static/fznpeayct6btel2og2wjjgvqw0ziqnk4.pth). Move the downloaded model to the following folder: 
```
../ibvse/featureextraction/solar/data/networks/
``` 

Finally, if you want to make use of the videoplayer in the GUI, make sure you have the correct codecs installed. For Windows users, the K-Lite Codec Pack Basic can be downloaded [here](https://codecguide.com/download_k-lite_codec_pack_basic.htm) and provides the codecs for all common audio and video formats. The  

## Running the code
### Terminal
The system can be ran in terminal by calling the file prototype_main.py with the following command
```
usage: prototype_main.py [-h] input_videos [input_videos ...] input_images [input_images ...]

```
Where the arguments are the corresponding paths to the video and image files. <br /> 
The system prints the timings of the system in terminal and returns the timestamp and distances corresponding to the search. 

### GUI
To run the program with the GUI, run python gui_main.py without any argyments. <br /> 
After inputting the video(s) and image(s) the system will open a new window that shows the results for each video and image. By selecting an image and timestamp and pressing the play button a videoplayer will open to show the found result (If the correct codecs are installed). Additionally a slider is provided to tweak the results.


# Repository organization 
## Keyframe Extraction 
The Keyframe Extraction module aims to reduce the amount of frames that need to be evaluated by the rest of the system. This allows for drastically reducing the overall time for the Feature Extraction module since it only needs to extract the features of a small set of images. <br />
The code for the Keyframe Extraction module can be found in the folder KeyFrameExtraction

## Feature Extraction 
The Feature Extraction module uses the SOLAR global model to extract the features of the keyframes and query images, so that they can be compared to one another. <br />
The code for the Feature Extraction  module can be found in the folder featurextraction

## Data Compression and Nearest Neighbour Search
The Data Compression and Nearest Neighbour Search module finds the closest matches between the feature vectors of the keyframes and query images. Multiple methods are implemented and a selector selects the most optimal method (of the implemented methods) based on the amount of keyframes and query images. To test the module as a standalone, extract the feautures of a dataset (the one used for this project was [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)) using any model (the one used for this project was resnet-18). The extracted feautres should be saved to numpy files with the following names: 
```
frames.npy          # The training data (majority of the dataset)
frame_labels.npy    # The labels of the training data (majority of the dataset)
images.npy          # The test data (minority of the dataset)
image_labels.npy    # The labels of the test data (minority of the dataset)
```
The code for the Data Compression and Nearest Neighbour Search module can be found in the folder nearestneighbor
