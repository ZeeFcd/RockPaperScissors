import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

image_list = []

image_list = []
for filename in glob.glob('paper/*.png'):
    image = cv2.imread(filename)
    image_list.append(image)

print(image_list[0])