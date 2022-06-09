def SIFT(fileName):

    videoCap = cv2.VideoCapture(fileName)
    fps = videoCap.get(cv2.CAP_PROP_FPS)
    print("Frames per second: ", fps)
    euclideanDistance = []


    i = 0
    success, image = videoCap.read()
    height = len(image)
    width = len(image[0])
    totalPixels = width * height
    while success:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([grayImage], [0], None, [256], [0, 256])
        colorMoments = getColorMoments(histogram, totalPixels)

        if i == 0:
            euclideanDistance.append(0)
        else:
            euclideanDistance.append(getEuclideanDistance(colorMoments, prevColorMoments))

        prevColorMoments = colorMoments

        i += 1
        success, image = videoCap.read()
        # Uncomment this for breaking early i.e. 100 frames
        # if i==50:
        #     break

    perc = 0.05
    keyFramesIndices = sorted(np.argsort(euclideanDistance)[::-1][:int(i * perc)])
    print(keyFramesIndices)
    return keyFramesIndices