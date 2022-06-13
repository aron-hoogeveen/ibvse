# k means clustering to generate video summary
import sys
import imageio
import numpy as np
import cv2
import scipy.io
# k-means
from sklearn.cluster import KMeans

# System Arguments
# Argument 1: Location of the video
# Argument 2: Sampling rate (k where every kth frame is chosed)
# Argument 3: Percentage of frames in the keyframe summany (Hence the number of cluster)
# NOTE: pass the number of clusters as -1 to choose 1/50 the number of frames in original video
# Only valid for SumMe dataset

# optional arguments 
# Argument 4: 1: if 3D Histograms need to be generated and clustered, else 0
# Argument 5: 1: if want to save keyframes 
# Argument 6: 1: if want to save the frame indices
# Argument 7: directory where keyframes will be saved
def VSUMM(descriptors, shot_frame_number):
	# defines the number of bins for pixel values of each type {r,g,b}
	#D

	# size of values in each bin
	#range_per_bin = 256 / num_bins

	# number of centroids
	percent = int(2)
	#global num_bins, sampling_rate, percent, num_centroids
	# print ("Opening video!")
	# video=imageio.get_reader(sys.argv[1])
	# vidcap = cv2.VideoCapture(sys.argv[1])
	# frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
	# print ("Video opened\nChoosing frames")
	# print(len(video))
	# print(frame_count)
	# #choosing the subset of frames from which video summary will be generateed
	# frames=[video.get_data(i*sampling_rate) for i in range(int(frame_count/sampling_rate))]
	# print ("Frames chosen")
	# print ("Length of video %d" % frame_count)

	# converting percentage to actual number
	num_centroids=int(percent*len(descriptors)/100)
	if num_centroids == 0:
		num_centroids = 1
	# if (frame_count/sampling_rate) < num_centroids:
	# 	print ("Samples too less to generate such a large summary")
	# 	print ("Changing to maximum possible centroids")
	# 	num_centroids=frame_count/sampling_rate
		


	# #opencv: generates 3 histograms corresponding to each channel for each frame
	# print ("Generating linear Histrograms using OpenCV")
	# channels=['b','g','r']
	# hist=[]
	# for frame in frames:
	# 	feature_value=[cv2.calcHist([frame],[i],None,[num_bins],[0,256]) for i,col in enumerate(channels)]
	# 	hist.append(np.asarray(feature_value).flatten())
	# hist=np.asarray(hist)


	# # choose number of centroids for clustering from user required frames (specified in GT folder for each video)
	# if percent==-1:
	# 	video_address=sys.argv[1].split('/')
	# 	gt_file=video_address[len(video_address)-1].split('.')[0]+'.mat'
	# 	video_address[len(video_address)-1]=gt_file
	# 	video_address[len(video_address)-2]='GT'
	# 	gt_file='/'.join(video_address)
	# 	num_frames=int(scipy.io.loadmat(gt_file).get('user_score').shape[0])
	# 	# automatic summary sizing: summary assumed to be 1/100 of original video
	# 	num_centroids=int(0.1*num_frames)

	kmeans=KMeans(n_clusters=num_centroids).fit(descriptors)
	kmeans2 = KMeans(n_clusters=num_centroids).fit_predict(descriptors)
	print(kmeans2)

	summary_frames=[]

	# transforms into cluster-distance space (n_cluster dimensional)
	hist_transform=kmeans.transform(descriptors)

	frame_indices=[]
	for cluster in range(hist_transform.shape[1]):
		#print ("Frame number: %d" % (np.argmin(hist_transform.T[cluster])*sampling_rate))
		frame_indices.append(np.argmin(hist_transform.T[cluster])+shot_frame_number)
	
	# frames generated in sequence from original video
	frame_indices=sorted(frame_indices)
	#summary_frames=[frames[i] for i in frame_indices]

	return frame_indices

