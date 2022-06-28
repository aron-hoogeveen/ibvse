# Image Based Video Search Engine 

The search engine was developed for the BSc graduation project of group H. The project is a search engine that finds an instance of an object in query image(s) in a set of query videos. 

# Running the system
To run the system, first clone the repository. To install the correct packages, change directory to the folder that contains the enviroment.yml file and install the enviroment by using the following command
```
conda env create
```
The system can be ran by calling the file main.py in terminal with the following command
```
usage: prototype-main.py [-h] input_video input_image
```
Where the arguments are the corresponding paths to the video and image files.

## output
The system outputs the time stamps, distances and the frame number for the found matches. The keyframes are stored as images in the bin folder of the user directory. The frame number output of the system corresponds to the image with the same number.


# Repository organization 
![alt text](https://github.com/aron-hoogeveen/ibvse/blob/main/pipeline.png?raw=true)
## Keyframe Extraction 
The code for the Keyframe Extraction module can be found in the folder KeyFrameExtraction

## Feature Extraction 
The code for the Feature Extraction  module can be found in the folder featurextraction

## Data Compression and Nearest Neighbour Search
The code for the Data Compression and Nearest Neighbour Search module can be found in the folder nearestneighbor
