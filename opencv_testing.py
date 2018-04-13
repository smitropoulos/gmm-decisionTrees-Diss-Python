

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
blobStorageArray = []


def bigBlobExtractor(im):
    # Blob Detector parameters
    # Setup SimpleBlobDetector parameters.
    frameBlobs = []
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
        frameBlobs.append (crop_img)
        plt.imshow(frameBlobs[j])
        plt.show()
       
    return frameBlobs

blobs = bigBlobExtractor(im)
vis=blobs[0]
blobStorageArray.append(blobs[0].tolist())


def removePadding(image):
    #image=image[:,:,1]      #needed constraint for non binary images
    height, width = image.shape
    
    for i in range(0,height-1):
        if sum(image[i,:])>0:
            top = i
            break
        
    for i in range(height-1,0,-1):
        if sum(image[i,:])>0:
            bottom = i
            break
    
    for i in range(0,width-1):
        if sum(image[:,i])>0:
            left = i
            break
        
    for i in range(width-1,0,-1):
        if sum(image[:,i])>0:
            right = i
            break    
    
    return image[top:abs(bottom), left:right ]


def rotate_image(image, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    #image=image[:,:,1]  #Use if image is not binary
    height, width = image.shape
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origin) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_image

image=rotate_image(vis,200)
image = removePadding(image)

cv2.imshow("im", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
