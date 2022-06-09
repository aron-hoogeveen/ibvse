import numpy as np
import cv2

sift = cv2.SIFT_create()
img = cv2.imread("keyframes/666.jpg")
img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

keypoints, descriptors = sift.detectAndCompute(img_gray, None)


img2 = cv2.imread("keyframes/0.jpg")
img2_gray= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

print("computing repeatability")
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors, descriptors2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print(good)

