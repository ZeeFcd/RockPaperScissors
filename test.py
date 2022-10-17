import cv2
import matplotlib.pyplot as plt
import numpy as np
import dataprovider as data
from sklearn.svm import LinearSVC
from joblib import dump, load

kep = cv2.imread("testO.jpg")
kep2 = cv2.cvtColor(kep, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(kep, cv2.COLOR_BGR2HSV)
Hue, Sat, Val = cv2.split(hsv)
intensityValues, occurences=np.unique(Hue, return_counts=True)
kezgray3 = cv2.threshold(Hue, 18, 255, cv2.THRESH_BINARY_INV)[1]


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)) #np.ones((21, 21), np.uint8)
img_erosionk = cv2.erode(kezgray3, kernel, iterations=7)
img_dilationk = cv2.dilate(img_erosionk, kernel, iterations=7)
szurt1 = cv2.GaussianBlur(img_dilationk,(21,21),0)
atmeretez = cv2.resize(szurt1, (300, 200), interpolation=cv2.INTER_AREA)



fvektor=np.array(atmeretez).flatten().reshape(1, -1)
#fvektor=np.array(kezgray3).flatten().reshape(1, -1)
clf = load('clfKNN.joblib')
pred = clf.predict(fvektor)
kopapirollo = {1: "Kő",  2: "Papír",  3:  "Olló"}

plt.imshow(kep2)
plt.show()
plt.hist(intensityValues, intensityValues, weights=occurences)
plt.show()
plt.imshow(kezgray3, cmap='gray')
plt.show()
plt.imshow(atmeretez, cmap='gray')
plt.show()
print(f"A képen látható: {kopapirollo[pred[0]]}")
