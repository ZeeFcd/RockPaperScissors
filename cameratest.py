import requests
import cv2
import numpy as np
from joblib import load
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "666"

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
clf = load('clfKNN.joblib')
kopapirollo = {1: "Kő", 2: "Papír", 3: "Olló"}

# While loop to continuously fetching data from the Url
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
    img_erosionk = cv2.erode(skinRegionYCrCb, kernel, iterations=3)
    img_dilationk = cv2.dilate(img_erosionk, kernel, iterations=3)


    if img_dilationk.shape[1] > img_dilationk.shape[0]:

        img = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)
        atmeretez = cv2.resize(img_dilationk, (300, 200), interpolation=cv2.INTER_AREA)
    else:

        img = cv2.resize(img, (200, 300), interpolation=cv2.INTER_AREA)
        kep3 = cv2.resize(img_dilationk, (200, 300), interpolation=cv2.INTER_AREA)
        kivagott = np.zeros(np.array(kep3[:100]).transpose().shape)
        maradek = np.array(kep3[100:])
        atmeretez = np.concatenate((kivagott, maradek), axis=1)


    fvektor = np.array(atmeretez).flatten().reshape(1, -1)

    pred = clf.predict(fvektor)
    print(kopapirollo[pred[0]])

    cv2.imshow("Android Cam", img)
    cv2.imshow("Kez Cam", atmeretez)
    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()