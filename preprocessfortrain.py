import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

paper_list = []
processed_paper_list=[]
for filename in glob.glob('paper/*.png'):
    image = cv2.imread(filename)
    paper_list.append(image)

kernel = np.ones((5, 5), np.uint8)
for paper in paper_list:

    Hue, Val, Sat = cv2.split(cv2.cvtColor(paper, cv2.COLOR_BGR2HSV))
    cut_out_by_Hue = cv2.threshold(Hue, 40, 255, cv2.THRESH_BINARY_INV)[1]
    morph_opened_im = cv2.dilate(cv2.erode(cut_out_by_Hue, kernel, iterations=1), kernel, iterations=1)
    blurred = cv2.GaussianBlur(morph_opened_im, (21, 21), 0)
    output_im = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
    processed_paper_list.append(output_im)


kep = cv2.cvtColor(paper_list[0], cv2.COLOR_BGR2RGB)
plt.subplot(121), plt.imshow(kep), plt.title("Original")
plt.subplot(122), plt.imshow(processed_paper_list[0], cmap='gray'), plt.title("Processed")
plt.show()
