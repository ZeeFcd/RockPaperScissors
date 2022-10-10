import cv2
import matplotlib.pyplot as plt
import numpy as np

kep = cv2.imread("testP.jpg")

kep2=cv2.cvtColor(kep, cv2.COLOR_BGR2RGB)
plt.imshow(kep2)
plt.show()

hsv = cv2.cvtColor(kep, cv2.COLOR_BGR2HSV)

Hue, Val, Sat = cv2.split(hsv)
intensityValues, occurences=np.unique(Hue, return_counts=True)

plt.hist(intensityValues, intensityValues, weights=occurences)
plt.show()

kezgray3 =  cv2.threshold(Hue, 40, 255, cv2.THRESH_BINARY_INV)[1]
plt.imshow(kezgray3, cmap='gray')
plt.show()
