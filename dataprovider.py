import cv2
import numpy as np
import glob

class ImageTrainDataProvider:

    def __init__(self):
        # processed list contains the images as feature vectors
        self.rock_list = None
        self.paper_list = None
        self.scissors_list = None
        self.rock_feature_vectors = None
        self.paper_feature_vectors = None
        self.scissors_feature_vectors = None
        self.processed_rock_list = None
        self.processed_paper_list = None
        self.processed_scissors_list = None

    def Read_All_Images(self):
        self.rock_list = self.import_images('rock/*.png')
        self.paper_list = self.import_images('paper/*.png')
        self.scissors_list = self.import_images('scissors/*.png')

    def Proccess_All_Images(self):
        self.processed_rock_list = self.get_process_images(self.rock_list)
        self.processed_paper_list = self.get_process_images(self.paper_list)
        self.processed_scissors_list = self.get_process_images(self.scissors_list)

    def Setup_Feature_Vectors(self):
        self.rock_feature_vectors = []
        self.paper_feature_vectors = []
        self.scissors_feature_vectors = []
        for rock in self.processed_rock_list:
            self.rock_feature_vectors.append(np.array(rock).flatten())
        for paper in self.processed_paper_list:
            self.paper_feature_vectors.append(np.array(paper).flatten())
        for scissors in self.processed_scissors_list:
            self.scissors_feature_vectors.append(np.array(scissors).flatten())

    # gets and image and returns the processed image
    def process_image(self, image):

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue = hsv_image[:, :, 0]
        cut_out_by_hue = cv2.threshold(hue, 40, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = np.ones((5, 5), np.uint8)
        morph_opened_im = cv2.dilate(cv2.erode(cut_out_by_hue, kernel, iterations=1), kernel, iterations=1)
        blurred = cv2.GaussianBlur(morph_opened_im, (21, 21), 0)

        return cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

    # processes a list of images
    def get_process_images(self, image_list):
        proccessed=[]
        for image in image_list:
            proccessed.append(self.process_image(image))

        return proccessed

    def import_images(self, path):

        image_list = []
        for filename in glob.glob(path):
            image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            image_list.append(image)

        return image_list
