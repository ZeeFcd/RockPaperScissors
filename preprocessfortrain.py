import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

def process_image(image):

    Hue, Val, Sat = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    cut_out_by_Hue = cv2.threshold(Hue, 40, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((5, 5), np.uint8)
    morph_opened_im = cv2.dilate(cv2.erode(cut_out_by_Hue, kernel, iterations=1), kernel, iterations=1)
    blurred = cv2.GaussianBlur(morph_opened_im, (21, 21), 0)

    return cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

def import_images(path):

    image_list = []
    for filename in glob.glob(path):
        image_to_process = cv2.imread(filename)
        image_list.append(process_image(image_to_process))

    return image_list


processed_paper_list = import_images('paper/*.png')
processed_rock_list = import_images('rock/*.png')
processed_scissors_list = import_images('scissors/*.png')

plt.subplot(131), plt.imshow(processed_paper_list[0], cmap = 'gray'), plt.title("Paper")
plt.subplot(132), plt.imshow(processed_rock_list[0], cmap = 'gray'), plt.title("Rock")
plt.subplot(133), plt.imshow(processed_scissors_list[0], cmap = 'gray'), plt.title("Scissors")
plt.show()
