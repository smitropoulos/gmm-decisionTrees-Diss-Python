height, width = image.shape
flag = 0;
for i in range(0,height-1):
    for j in range(0,width-1):
        if image[i,j] > 0:
            topPixel =[j,i]  #x,y tradition
            flag=1
            break
    if flag==1:
        break
y_diff = round(height/2) - topPixel[0]
x_diff = round(width/2) - topPixel[1]
tanofdiff = math.tan(y_diff / x_diff)
atan = math.atan(tanofdiff)
tanofdiff=math.degrees(atan)

temp = image[top:abs(bottom), left:right ]

cv2.imshow("im", image)
cv2.waitKey(0)
cv2.destroyAllWindows()