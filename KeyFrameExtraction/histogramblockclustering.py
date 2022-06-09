import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import cv2
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

def blockclustering(feature_vectors, shot_frame_number):
    minf = 63
    if (len(feature_vectors) < minf):
        minf = len(feature_vectors)-1
    strminf = "v" + str(minf+1)
    arr = np.empty((0, 1944), int)  # initializing 1944 dimensional array to store 'flattened' color histograms

    for feature_vector in feature_vectors:
        arr = np.vstack((arr, feature_vector))  # appending each one-dimensinal vector to generate N*M matrix (where N is number of frames
        # and M is 1944)

    count = len(feature_vectors)

    final_arr = arr.transpose()  # transposing so that i will have all frames in columns i.e M*N dimensional matrix
    # where M is 1944 and N is number of frames
    #print(final_arr.shape)
    #print("Original frame amount:" + str(count))

    A = csc_matrix(final_arr, dtype=float)

    #top 63 singular values from 76082 to 508
    u, s, vt = svds(A, k = minf)

    #print(u.shape, s.shape, vt.shape)

    #print(list(s))

    v1_t = vt.transpose()

    projections = v1_t @ np.diag(s) #the column vectors i.e the frame histogram data has been projected onto the orthonormal basis
    #formed by vectors of the left singular matrix u .The coordinates of the frames in this space are given by v1_t @ np.diag(s)
    #So we can see that , now we need only 63 dimensions to represent each column/frame
    #print(projections.shape)

    # dynamic clustering of projected frame histograms to find which all frames are similar i.e make shots
    f = projections
    C = dict()  # to store frames in respective cluster
    for i in range(f.shape[0]):
        C[i] = np.empty((0, minf), int)

    # adding first two projected frames in first cluster i.e Initializaton
    C[0] = np.vstack((C[0], f[0]))
    C[0] = np.vstack((C[0], f[1]))

    E = dict()  # to store centroids of each cluster
    for i in range(projections.shape[0]):
        E[i] = np.empty((0, minf), int)

    E[0] = np.mean(C[0], axis=0)  # finding centroid of C[0] cluster

    count = 0
    for i in range(2, f.shape[0]):
        similarity = np.dot(f[i], E[count]) / (
                    (np.dot(f[i], f[i]) ** .5) * (np.dot(E[count], E[count]) ** .5))  # cosine similarity
        # this metric is used to quantify how similar is one vector to other. The maximum value is 1 which indicates they are same
        # and if the value is 0 which indicates they are orthogonal nothing is common between them.
        # Here we want to find similarity between each projected frame and last cluster formed chronologically.

        if similarity < 0.8:  # if the projected frame and last cluster formed  are not similar upto 0.9 cosine value then
            # we assign this data point to newly created cluster and find centroid
            # We checked other thresholds also like 0.85, 0.875, 0.95, 0.98
            # but 0.9 looks okay because as we go below then we get many key-frames for similar event and
            # as we go above we have lesser number of key-frames thus missed some events. So, 0.9 seems optimal.

            count += 1
            C[count] = np.vstack((C[count], f[i]))
            E[count] = np.mean(C[count], axis=0)
        else:  # if they are similar then assign this data point to last cluster formed and update the centroid of the cluster
            C[count] = np.vstack((C[count], f[i]))
            E[count] = np.mean(C[count], axis=0)


    b = []  #find the number of data points in each cluster formed.

    #We can assume that sparse clusters indicates
    #transition between shots so we will ignore these frames which lies in such clusters and wherever the clusters are densely populated indicates they form shots
    #and we can take the last element of these shots to summarise that particular shot

    for i in range(f.shape[0]):
        b.append(C[i].shape[0])

    last = b.index(0)  #where we find 0 in b indicates that all required clusters have been formed , so we can delete these from C
    b1=b[:last ] #The size of each cluster.

    res = [idx for idx, val in enumerate(b1) if val >= 5] #so i am assuming any dense cluster with atleast 25 frames is eligible to
    #make shot.
    #print("Number of keyframes extracted: " + str(len(res))) #so total 25 shots with 46 (71-25) cuts



    GG = C #copying the elements of C to GG, the purpose of  the below code is to label each cluster so later
    #it would be easier to identify frames in each cluster
    for i in range(last):
        p1= np.repeat(i, b1[i]).reshape(b1[i],1)
        GG[i] = np.hstack((GG[i],p1))


    #the purpose of the below code is to append each cluster to get multidimensional array of dimension N*64, N is number of frames
    F=  np.empty((0,minf+1), int)
    for i in range(last):
        F = np.vstack((F,GG[i]))


    #converting F (multidimensional array)  to dataframe

    colnames = []
    for i in range(1, minf+2):
        col_name = "v" + str(i)
        colnames+= [col_name]
    #print(colnames)

    df = pd.DataFrame(F, columns= colnames)

    df[strminf]= df[strminf].astype(int)  #converting the cluster level from float type to integer type
    attribute = strminf
    df1 =  df[getattr(df, attribute).isin(res)]   #filter only those frames which are eligible to be a part of shot or filter those frames who are
    #part of required clusters that have more than 25 frames in it

    new = df1.groupby(strminf).tail(1)[strminf] #For each cluster /group take its last element which summarize the shot i.e key-frame

    new1 = new.index #finding key-frames (frame number so that we can go back get the original picture)
    indices = []
    for i in range(0, len(new1)):
        indices.append(new1[i]+shot_frame_number)
    return indices

