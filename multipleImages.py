from os import listdir
from os.path import isfile, join
import numpy
import cv2

from opencv_testing import removePadding
from opencv_testing import rotationAngle
from opencv_testing import rotateImage

mypath='D:\Dropbox\Matlab\MSc\Results\human'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(shape = (len(onlyfiles),2), dtype=object)
for n in range(0, len(onlyfiles)):
  image = cv2.imread( join(mypath,onlyfiles[n]) )
  image = rotateImage(removePadding(image),rotationAngle(image))    #add the preprocessing
  image = numpy.reshape(image,image.shape[0]*image.shape[1])
  images[n,0] = image
  images[n,1] = 1 #1 is codename for person. Decision trees do not need dummy variables so 1 is enough.

mypath='D:\Dropbox\Matlab\MSc\Results\car'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
car = numpy.empty(shape = (len(onlyfiles),2), dtype=object)
for n in range(0, len(onlyfiles)):
  image = cv2.imread( join(mypath,onlyfiles[n]) )
  image = rotateImage(removePadding(image),rotationAngle(image))    #add the preprocessing
  image = numpy.reshape(image,image.shape[0]*image.shape[1])
  car[n,0] = image
  car[n,1] = 0 
  
independent = numpy.concatenate((images, car), axis=0)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',X_test[95,0])
cv2.waitKey(0)
cv2.destroyAllWindows()


