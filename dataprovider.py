import cv2
import numpy as np
import glob

class ImageTrainDataProvider:

    def __init__(self):
        # processed list contains the images as feature vectors
        self.processed_paper_list = self.import_and_process('paper/*.png')
        self.processed_rock_list = self.import_and_process('rock/*.png')
        self.processed_scissors_list = self.import_and_process('scissors/*.png')

    # gets and image and returns the processed image as a feature vector
    def process_image(self, image):

        Hue, Val, Sat = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        cut_out_by_Hue = cv2.threshold(Hue, 40, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = np.ones((5, 5), np.uint8)
        morph_opened_im = cv2.dilate(cv2.erode(cut_out_by_Hue, kernel, iterations=1), kernel, iterations=1)
        blurred = cv2.GaussianBlur(morph_opened_im, (21, 21), 0)

        return np.array(cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]).flatten()

    # reads the images and processes them
    def import_and_process(self, path):

        image_list = []
        for filename in glob.glob(path):
            image_to_process = cv2.imread(filename)
            image_list.append(self.process_image(image_to_process))

        return image_list
