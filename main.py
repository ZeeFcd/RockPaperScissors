import cv2
import matplotlib.pyplot as plt
import numpy as np

kep = cv2.imread("paper/0a3UtNzl5Ll3sq8K.png")

kep2=cv2.cvtColor(kep, cv2.COLOR_BGR2RGB)
plt.imshow(kep2)
plt.show()

hsv = cv2.cvtColor(kep, cv2.COLOR_BGR2HSV)

Hue, Val, Sat = cv2.split(hsv)
intensityValues, occurences=np.unique(Hue, return_counts=True)

plt.hist(intensityValues, intensityValues, weights=occurences)
plt.show()
#kezgray2 = cv2.cvtColor(kep2, cv2.COLOR_RGB2GRAY)
#for i in range(0,kezgray2.shape[0]):
#    for j in range(0,kezgray2.shape[1]):
#        if Hue[i][j] < 40:
#            kezgray2[i][j]=255
#        else:
#            kezgray2[i][j]=0

kezgray3 =  cv2.threshold(Hue, 40, 255, cv2.THRESH_BINARY_INV)[1]
plt.imshow(kezgray3, cmap='gray')
plt.show()

kernel = np.ones((5, 5), np.uint8)
img_erosionk = cv2.erode(kezgray3, kernel, iterations=1)
img_dilationk = cv2.dilate(img_erosionk, kernel, iterations=1)

szurt1= cv2.GaussianBlur(img_dilationk,(21,21),0)
feketef1 =  cv2.threshold(szurt1, 100, 255, cv2.THRESH_BINARY)[1]

plt.imshow(feketef1, cmap='gray')
plt.show()