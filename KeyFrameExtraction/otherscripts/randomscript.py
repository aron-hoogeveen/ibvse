import numpy as np
import cv2
import scipy
import os
import glob
from matplotlib import pyplot as plt
import scipy.signal as sig
from scipy.spatial import distance
from skimage.feature import hog
from skimage import data, exposure

# Define Sobel operators for gradient of edges
kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

imagePath = '498.jpg'
imagePath2 = '147.jpg'
image = cv2.imread(imagePath,1)
image2 = cv2.imread(imagePath2,1)

#create HSV color space histogram with 64 bins
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
hist_1 = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
hist_2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
hist_1 = cv2.normalize(hist_1, None).flatten()
hist_2 = cv2.normalize(hist_2, None).flatten()
a = cv2.compareHist(hist_1,hist_2,cv2.HISTCMP_BHATTACHARYYA)
print(a)


# cvt_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to HSV
# h, s, v = cvt_image[:,:,0], cvt_image[:,:,1], cvt_image[:,:,2]
# hist_h = cv2.calcHist([h],[0],None,[64],[0,256])
# hist_s = cv2.calcHist([s],[0],None,[64],[0,256])
# hist_v = cv2.calcHist([v],[0],None,[64],[0,256])
# plt.plot(hist_h, color='r', label="h")
# plt.plot(hist_s, color='g', label="s")
# plt.plot(hist_v, color='b', label="v")
# plt.legend()
# plt.show()

#edge direction histogram
# grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# G_x = sig.convolve2d(grayscale_image, kernel_x, mode='same')
# G_y = sig.convolve2d(grayscale_image, kernel_y, mode='same')
#
# # Plot them!
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
#
# # Actually plt.imshow() can handle the value scale well even if I don't do
# # the transformation (G_x + 255) / 2.
# ax1.imshow((G_x + 255) / 2, cmap='gray'); ax1.set_xlabel("Gx")
# ax2.imshow((G_y + 255) / 2, cmap='gray'); ax2.set_xlabel("Gy")
# plt.show()



# fd, hog_image = hog(image, orientations=3, pixels_per_cell=(200, 200),cells_per_block=(1, 1), visualize=True, multichannel=True)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
#
# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')
#
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()


#hog_image = hog(image, orientations = 72, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, channel_axis=-1)
#print(hog_image)


fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),
                    cells_per_block=(2, 2), visualize=True, channel_axis=-1)

fd2, hog_image2 = hog(image2, orientations=8, pixels_per_cell=(32, 32),
                    cells_per_block=(2, 2), visualize=True, channel_axis=-1)

n_fd = fd / np.sqrt(np.sum(fd**2))
n_fd2 = fd2 / np.sqrt(np.sum(fd2**2))

dst = distance.euclidean(n_fd, n_fd2)
print(dst)

hog = cv2.HOGDescriptor()
h = hog.compute(image)
print(h)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')
#
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()

