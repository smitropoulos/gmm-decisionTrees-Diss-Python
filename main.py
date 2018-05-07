# Decision Tree Classification
#can't be assed to use argv.
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from multipleImages import multipleImageLoader
import cv2

reshapedImagesWithClassInTheEndColumn = multipleImageLoader('human', 'car')

X = reshapedImagesWithClassInTheEndColumn[:,:-1]
y = reshapedImagesWithClassInTheEndColumn[:,reshapedImagesWithClassInTheEndColumn.shape[1]-1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


df_cm = pd.DataFrame(cm, index = [i for i in ("Car","Human")],
                  columns = [i for i in ("Car","Human")])
plt.figure(figsize = (5,3))
sn.heatmap(df_cm, annot=True)



import cv2
import numpy
from functions import removePadding
from functions import rotationAngle
from functions import rotateImage
dim = 150

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    images = numpy.zeros(shape = (1,dim*dim), dtype=int)

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret2,imageGray = cv2.threshold(imageGray,0,1,cv2.THRESH_OTSU)
    imageGray = cv2.medianBlur(imageGray,7) #Always use int n , n%2 ==0
    imageGray = rotateImage(removePadding(imageGray),rotationAngle(imageGray))    #add the preprocessing
    imageGray = cv2.resize(imageGray,(dim,dim))   #Resizing the image
    imageGray = numpy.reshape(imageGray,imageGray.shape[0]*imageGray.shape[1])    #reshaping the image from 2d to a vector
    images[0,:] = imageGray


    y_pred = classifier.predict(images)
    print (y_pred)
    # Display the resulting frame
    cv2.imshow('image',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

