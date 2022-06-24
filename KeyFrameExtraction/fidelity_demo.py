import numpy as np
from fidelity import *
from main import *
#import matplotlib.pyplot as plt
from math import atan2
from math import degrees


if __name__ == '__main__':
    print(sys.argv[1])
    path = sys.argv[1]

    #[hogs, hists, fdnorm, histnorm] = fidelity_descriptors(path)
    #

    KE_method = "VSUMM_combi"
    performSBD = False
    presample = True
    keyframes_data, keyframe_indices, video_fps = keyframe_extraction(sys.argv[1], KE_method, performSBD, presample)
    save_keyframes(keyframe_indices, keyframes_data, "vsummcombimetpresample")

    #fid = fidelity(keyframe_indices, path, hists, hogs, fdnorm, histnorm)
    #print("Fidelity: " + str(fid))


    KE_method = "VSUMM_combi"
    performSBD = False
    presample = False
    keyframes_data, keyframe_indices, video_fps = keyframe_extraction(sys.argv[1], KE_method, performSBD, presample)
    save_keyframes(keyframe_indices, keyframes_data, "vsummcombizonderpresample")

    #fid = fidelity(keyframe_indices, path, hists, hogs, fdnorm, histnorm)
    #print("Fidelity: " + str(fid))

    # keyframes_data, keyframe_indices, video_fps = fast_uniform_sampling(sys.argv[1], 0.5, 0.85)
    # fid = fidelity(keyframe_indices, path, hists, hogs, fdnorm, histnorm)
    #print("Fidelity: " + str(fid))








    #DEMO histogram bins HOGS and edges
    # cap = cv2.VideoCapture(sys.argv[1])
    # success, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #
    # #resize image for computational speed gain
    # scale_percent = 5  # percent of original size
    # width = int(frame.shape[1] * scale_percent / 100)
    # height = int(frame.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    #
    # window_name = ('Sobel Demo - Simple Edge Detector')
    # scale = 1
    # delta = 0
    # ddepth = cv2.CV_16S
    #
    # # Gaussian blur (kernel size = 3)
    # blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    # # Convert to grayscale (= luminance)
    # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # gradientmax = 255*4 #maximum gradient value
    # threshold = gradientmax *0.3 # threshold to reduce background noise
    # # apply sobel filters,
    # grad_x = np.array(cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT))
    # grad_y = np.array(cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT))
    #
    # # calculate absolute values
    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)
    #
    # # Gradient magnitude calculated by 0.5(|G_x|+|G_y|) instead of sqrt(G_x^2 + G_y^2)
    # grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #
    # histogram = np.zeros(72, dtype="float32")
    # for i in range (0, grad.shape[0]):
    #      for j in range(0, grad.shape[1]):
    #         if grad[i][j]:
    #             angle = atan2(grad_y[i][j], grad_x[i][j])/3.14
    #             histogram[round(abs(angle*71))] += int(1)
    #
    #
    # # plt.plot(histogram, color = 'm')
    # # plt.title('72-bin Sobel histogram')
    # # plt.xlabel('bins')
    # # plt.ylabel('number of pixels')
    # # plt.legend(loc="upper right")
    # # plt.show()
    #
    # h = cv2.calcHist([frame], [0], None, [64], [0, 256])
    # intersection = cv2.compareHist(histogram, histogram, cv2.HISTCMP_INTERSECT) / (frame.shape[0] * frame.shape[1])
    #
    #
    #
    # print(intersection)
    #
    # histogram = histogram[:, np.newaxis]
    # print(h.dtype)
    # print(histogram.dtype)
    # #histogram= histogram.astype(np.uint8)
    #
    # intersection = cv2.compareHist(histogram, histogram, cv2.HISTCMP_INTERSECT)/(grad.shape[0] * grad.shape[1])
    # print(intersection)

    #
    #
    # cv2.imshow(window_name, abs_grad_x)
    # cv2.waitKey(0)
    #cv2.waitKey(0)


    # plt.imshow(gray, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    #64 bins, HSV, intersection
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.calcHist(images, channels, mask, histSize, ranges)
    # h = cv2.calcHist([frame], [0], None, [64], [0, 256])
    # s = cv2.calcHist([frame], [1], None, [64], [0, 256])
    # v = cv2.calcHist([frame], [2], None, [64], [0, 256])
    # print(v.shape)

   # hist = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])

    #cap = cv2.VideoCapture(sys.argv[1])
    #for i in range(0, 200):
    #    success, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #hist2 = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
    #print(cv2.compareHist(hist, hist, cv2.HISTCMP_INTERSECT))
    #intersection = cv2.compareHist(hist, hist, cv2.HISTCMP_INTERSECT) / (frame.shape[0] * frame.shape[1])
    #print(intersection)
    # hist = cv2.normalize(hist, None).flatten()
    # plt.plot(h, color = 'r', label='H')
    # plt.plot(s, color='g', label='S')
    # plt.plot(v, color='b', label='V')
    # plt.title('64-bin HSV histogram of "boxes.png"')
    # plt.xlabel('bins')
    # plt.ylabel('number of pixels')
    # plt.legend(loc="upper right")
    # plt.show()


    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # print("Creating hog for frame")
    # fd, hog_image = hog(frame, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
    #                     channel_axis=-1)
    # plt.imshow(hog_image)
    # plt.show()

    # image = gaussian_blur(frame, 9, verbose=True)
    # filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # sobel_edge_detection(image, filter, verbose=True)

    # def sobel_edge_detection(image, filter, verbose=False):
    #     new_image_x = convolution(image, filter, verbose)
    #
    #     if verbose:
    #         plt.imshow(new_image_x, cmap='gray')
    #         plt.title("Horizontal Edge")
    #         plt.show()
    #
    #     new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)
    #
    #     if verbose:
    #         plt.imshow(new_image_y, cmap='gray')
    #         plt.title("Vertical Edge")
    #         plt.show()
    #
    #     gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    #
    #     gradient_magnitude *= 255.0 / gradient_magnitude.max()
    #
    #     if verbose:
    #         plt.imshow(gradient_magnitude, cmap='gray')
    #         plt.title("Gradient Magnitude")
    #         plt.show()
    #
    #     return gradient_magnitude