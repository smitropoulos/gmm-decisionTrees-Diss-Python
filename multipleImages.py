def multipleImageLoader(pathofclass1, pathofclass0):

#Loads multiple images and processes them. Returns a numpy array

    from os import listdir
    from os.path import isfile, join
    import numpy
    import cv2

    from functions import removePadding
    from functions import rotationAngle
    from functions import rotateImage

    #We need a constant size of the images so we introduce a resize mechanic.
    #Resizing the images does not drastically reduce the performance of the classifier
    dim =150     #init 150.

    mypath=pathofclass1
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    images = numpy.empty(shape = (len(onlyfiles),dim*dim +1), dtype=int)
    for n in range(0, len(onlyfiles)):
      image = cv2.imread( join(mypath,onlyfiles[n]),0 )
      image = cv2.medianBlur(image,7) #Always use int n , n%2 ==0
      ret2,image = cv2.threshold(image,0,1,cv2.THRESH_OTSU)
      image = rotateImage(removePadding(image),rotationAngle(image))    #add the preprocessing
      image = cv2.resize(image,(dim,dim))
      image = numpy.reshape(image,image.shape[0]*image.shape[1])
      images[n,:-1] = image
      images[n,dim*dim] = 1 #1 is codename for person. Decision trees do not need dummy variables so 1 is enough.

    mypath=pathofclass0
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    car = numpy.empty(shape = (len(onlyfiles),dim*dim +1), dtype=int)
    for n in range(0, len(onlyfiles)):
      image = cv2.imread( join(mypath,onlyfiles[n]),0 )
      image = cv2.medianBlur(image,7) #Always use int n , n%2 ==0
      ret2,image = cv2.threshold(image,0,1,cv2.THRESH_OTSU)
      image = rotateImage(removePadding(image),rotationAngle(image))    #add the preprocessing
      image = cv2.resize(image,(dim,dim))
      image = numpy.reshape(image,image.shape[0]*image.shape[1])
      car[n,:-1] = image
      car[n,dim*dim] = 0 #1 is codename for person. Decision trees do not need dummy variables so 1 is enough.

    concat = numpy.concatenate((images, car), axis=0)
    return concat#