import sys
import imageio
import numpy as np
import cv2
import shlex
import subprocess
import os, sys, glob
from scc import strongly_connected_components_tree

# System Arguments
# Argument 1: Location of the video
# Argument 2: Location of save

# defines the number of bins for pixel values of each type as used the original work
num_bins_H=32
num_bins_S=4
num_bins_V=2

# manual function to generate histogram on HSV values
def generate_histogram_hsv(frame):
	hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hsv_frame = hsv_frame
	global num_bins_H, num_bins_S, num_bins_V
	hist = cv2.calcHist([frame], [0, 1, 2], None, [int(256/num_bins_H), int(256/num_bins_S), int(256/num_bins_V)],
		[0, 256, 0, 256, 0, 256])
	
	#norm = np.zeros((800,800))
	#hist = cv2.normalize(hist, norm).flatten()
	return hist;

# function to calculate the distance matrix for bhattacharyya_distance
def bhattacharyya_distance(color_histogram):
	distance_matrix=np.zeros((len(color_histogram),len(color_histogram)))
	for i in range(len(color_histogram)):
		temp_list = []
		for j in range(len(color_histogram)):
			if i != j:
				distance_matrix[i][j] = cv2.compareHist(color_histogram[i],color_histogram[j],cv2.HISTCMP_BHATTACHARYYA)
			else:
				distance_matrix[i][j] = float("inf")
	return distance_matrix

def save_keyframes(frame_indices, summary_frames):
	print ("Saving frame indices")
	video_name = sys.argv[1]
	video_name = video_name.split('.')
	video_name = video_name[0].split('\\')
	out_file=open("frame_indices_"+video_name[len(list(video_name))-1]+".txt",'w')
	for idx in frame_indices:
		out_file.write(str(idx)+'\n')
	print ("Saved indices")

def main():
	if len(sys.argv) < 2:
		print ("Incorrect no. of arguments, Halting !!!!")
		return
	print ("Opening video!")

	video=imageio.get_reader(sys.argv[1]);
	print ("Video opened\nChoosing frames")

	# delete scenes.txt if it already exists
	print ("Detecting different shots")
	if os.path.exists("C:\\Users\\Leo\\Desktop\\SumMe\\scenes.txt"):
		os.remove("C:\\Users\\Leo\\Desktop\\SumMe\\scenes.txt")
	# use the parameter currently set as "0.1" to control the no. of frames to be selected
	video_name = sys.argv[1]
	video_name = video_name.split('.')
	video_name = video_name[0].split('\\')
	video_name = video_name[len(list(video_name)) - 1]+'.mp4'
	dir = sys.argv[1][:-len(video_name)-1]
	print("hoiiii")
	print (video_name)
	os.chdir(dir)
	print(dir)
	# value after scene\, is the threshold
	cmd = 'ffprobe -show_frames -of compact=p=0 -f lavfi "movie='+str(video_name)+',select=gt(scene\,0.07)">> ../scenes.txt'
	print(cmd)
	os.system(cmd)
	os.chdir('..')
	seginfo = 'scenes.txt'
	frame_index_list = []
	for line in open(seginfo,'r'):
		line = line.replace("|"," ")
		line = line.replace("="," ")
		parts = line.split()
		frame_index_list.append(int(parts[11])) #appending the frame no. in the list of selected frames
	print (frame_index_list, len(video))
	frames = []
	for i in range(len(frame_index_list)):
		if frame_index_list[i] >= 0 and frame_index_list[i] < len(video):
			frames.append(np.array(video.get_data(frame_index_list[i])))

	if len(frames) <= 0:
		print ("unable to detect any shot, Halting !!!!")
		return
	print ("Frames chosen: ", len(frame_index_list))

	#extracting color features from each representative frame
	print ("Generating Histrograms")
	color_histogram=[generate_histogram_hsv(frame) for frame in frames]
	print ("Color Histograms generated")

	#to-do (optional): extract texture features for each frame

	#calculate distance between each pair of feature histograms
	print ("Evaluating the distance matirix for feature hitograms")
	distance_matrix = bhattacharyya_distance(color_histogram)
	print ("Done Evalualting distance matrix")

	#constructing NNG (nearest neighbour graph) based of distance_matrix
	print ("Constructing NNG")
	eps_texture_NN = [None]*len(distance_matrix[0])
	for i in range(0,len(distance_matrix[0])):
		temp = float(0)
		for j in range(len(distance_matrix[i])):
			if distance_matrix[i][j] >= temp:
				eps_texture_NN[i] = j
				temp = distance_matrix[i][j]

	#constructing RNNG(reverse nearest neighbour graph) for the above NNG
	print ("Constructing RNNG")
	eps_texture_RNN = {}
	for i in range(len(eps_texture_NN)):
		if eps_texture_NN[i] in eps_texture_RNN.keys():
			eps_texture_RNN[eps_texture_NN[i]].append(i)
		else:
			eps_texture_RNN[eps_texture_NN[i]] = [i]
		if i not in eps_texture_RNN.keys():
			eps_texture_RNN[i] = []

	#calculating the SCCs(strongly connected components) for RNNG
	print ("Finiding the strongly connected components of RNNG")
	vertices = [i for i in range(0,len(frames))]
	scc_graph = strongly_connected_components_tree(vertices, eps_texture_RNN)

	#choosing one frame per SCC in summary
	print ("Evaluating final summary")
	summary = []
	summary_frames = []
	skim_length = 10
	for scc in scc_graph:
		frame_to_add = frame_index_list[next(iter(scc))]
		print(frame_to_add)
		for i in range(-skim_length,skim_length):
			if frame_to_add + i > 0 and frame_to_add + i < len(video):
				if frame_to_add+i not in summary:
					summary.append(frame_to_add+i)
				# summary_frames.append(video.get_data(frame_to_add + i))

	# writing the summary in a file 
	os.chdir(sys.argv[2])
	print("hoi")
	print(summary)
	save_keyframes(summary, summary_frames)

if __name__ == '__main__':
	main()