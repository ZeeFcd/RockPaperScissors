import cv2
import matplotlib.pyplot as plt
import numpy as np

from joblib import load

kep = cv2.imread("testO.jpg")
kep2 = cv2.cvtColor(kep, cv2.COLOR_BGR2RGB)

ycrcb = cv2.cvtColor(kep, cv2.COLOR_BGR2YCR_CB)
min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)
skinRegionYCrCb = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
img_erosionk = cv2.erode(skinRegionYCrCb, kernel, iterations=3)
img_dilationk = cv2.dilate(img_erosionk, kernel, iterations=3)
szurt1 = cv2.GaussianBlur(img_dilationk, (21, 21), 0)
atmeretez = cv2.resize(img_dilationk, (300, 200), interpolation=cv2.INTER_AREA)
atmeretez[atmeretez > 0] = 255

fvektor = np.array(atmeretez).flatten().reshape(1, -1)

clf = load('clfKNN.joblib')

pred = clf.predict(fvektor)
kopapirollo = {1: "Kő",  2: "Papír",  3:  "Olló"}

plt.imshow(kep2)
plt.show()

plt.imshow(skinRegionYCrCb, cmap='gray')
plt.show()

plt.imshow(atmeretez, cmap='gray'),  plt.title(kopapirollo[pred[0]])
plt.show()
print(kopapirollo[pred[0]])

