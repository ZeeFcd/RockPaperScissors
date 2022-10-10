import cv2
import matplotlib.pyplot as plt
import numpy as np
import dataprovider as proc



if __name__ == "__main__":

    data_provider = proc.ImageTrainDataProvider()
    print(data_provider.processed_paper_list[0])
    print(data_provider.processed_rock_list[0])
    print(data_provider.processed_scissors_list[0])
