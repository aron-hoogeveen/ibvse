# k means clustering to generate video summary
import sys
import imageio
import numpy as np
import cv2
import scipy.io
# k-means
from sklearn.cluster import KMeans

# combines histogram and vsumm
def VSUMM_combi(descriptors, shot_frame_number, skip_num):

	percent = int(5*skip_num)

	# converting percentage to actual number
	num_centroids=int(percent*len(descriptors)/100)
	if num_centroids == 0:
		num_centroids = 1
	kmeans=KMeans(n_clusters=num_centroids).fit(descriptors)
	kmeans2 = KMeans(n_clusters=num_centroids).fit_predict(descriptors)
	print(kmeans2)

	summary_frames=[]

	# transforms into cluster-distance space (n_cluster dimensional)
	hist_transform=kmeans.transform(descriptors)

	frame_indices=[]
	for cluster in range(hist_transform.shape[1]):
		#print ("Frame number: %d" % (np.argmin(hist_transform.T[cluster])*sampling_rate))
		frame_indices.append(np.argmin(hist_transform.T[cluster]))
	
	# frames generated in sequence from original video
	frame_indices=sorted(frame_indices)

	# apply histogram threshold to extracted keyframes
	threshold = 0.2
	deleted_amnt = 0
	if len(frame_indices) > 1:
		for i in range(1, len(frame_indices)):
			histdiff = cv2.compareHist(descriptors[frame_indices[i-deleted_amnt]], descriptors[frame_indices[i-1-deleted_amnt]], cv2.HISTCMP_BHATTACHARYYA)
			print(histdiff)
			if (histdiff) < threshold:
				frame_indices.pop(i-deleted_amnt) # delete frame because it is too similar to previous
				deleted_amnt += 1
			# if i >= len(frame_indices)-deleted_amnt:
			# 	break
			print(i)
	print("Removed amount of frames after clustering: " + str(deleted_amnt))


	frame_indices = [element + shot_frame_number for element in frame_indices]
	#summary_frames=[frames[i] for i in frame_indices]

	return frame_indices

