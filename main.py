import matplotlib.pyplot as plt
import numpy as np
import dataprovider as data
import sklearn as sk

if __name__ == "__main__":

    data_provider = data.ImageTrainDataProvider()
    data_provider.Read_All_Images()
    data_provider.Proccess_All_Images()
    data_provider.Setup_Feature_Vectors()

    plt.subplot(2, 3, 1), plt.imshow(data_provider.rock_list[0])
    plt.subplot(2, 3, 2), plt.imshow(data_provider.paper_list[0])
    plt.subplot(2, 3, 3), plt.imshow(data_provider.scissors_list[0])
    plt.subplot(2, 3, 4), plt.imshow(data_provider.processed_rock_list[0], cmap='gray')
    plt.subplot(2, 3, 5), plt.imshow(data_provider.processed_paper_list[0], cmap='gray')
    plt.subplot(2, 3, 6), plt.imshow(data_provider.processed_scissors_list[0], cmap='gray')
    plt.show()

