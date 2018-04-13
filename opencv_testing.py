

# Standard imports
import cv2
import numpy as np;
from matplotlib import pyplot as plt

# Read image
im = cv2.imread("474.png", cv2.IMREAD_GRAYSCALE)

"""
cv2.imshow("im", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 300
params.maxArea = 300000

params.filterByCircularity = True
params.minCircularity=0
params.maxCircularity=1

params.filterByColor = True
params.blobColor = 255


params.filterByConvexity = True
params.minConvexity=0
params.maxConvexity=1


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
X_data = []


    
    
for j in range(len(keypoints)):
    print (j)
    size = int(keypoints[1].size /2)
    x = int(keypoints[j].pt[1])
    y = int(keypoints[j].pt[0])
    lowerx = x-size
    upperx = x+size
    
    if lowerx <0:
        lowerx=0
        
    if upperx > len(im[0]):
        upperx = len(im[0])
              
    lowery = y-size
    uppery = y+size
    
    if lowery <0:
        lowery=0
        
    if uppery > len(im):
        uppery = len(im)
    crop_img = im[lowerx:upperx, lowery:uppery]
    X_data.append (crop_img)
    plt.imshow(crop_img)
    plt.show()


print (j)

size = int(keypoints[0].size /2)
x = int(keypoints[0].pt[1])
y = int(keypoints[0].pt[0])
crop_img = im[y-size:y+size,x-size:x+size]
X_data.append (crop_img)
plt.imshow(crop_img)
plt.show()

